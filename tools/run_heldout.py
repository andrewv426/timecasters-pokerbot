"""Compare Fixed Equity and Adaptive Equity using the official duplicate judge."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import sys

from run_baselines import ROOT, run_match

PROFILES = ("value", "polarized", "trap", "switch")
SEEDS = (3107, 7919, 12239)


class DirectAgent:
    """Give the official match loop the same hook lifecycle without HTTP."""

    def __init__(self, name, config, start, action, end):
        self.name = name
        self.config = config
        self.start = start
        self.action = action
        self.end = end

    def start_half(self):
        from judge_engine import AgentStats

        self.memory = {}
        self.pending = []
        self.stats = AgentStats()
        self.start(self.config, self.memory)

    def act(self, obs):
        self.flush_hand_results()
        action = self.action(obs, self.memory)
        if action not in obs.legal_actions:
            raise ValueError(f"{self.name} returned an illegal action")
        self.stats.calls += 1
        return action

    def hand_end(self, result):
        self.pending.append(result)

    def flush_hand_results(self):
        for result in self.pending:
            self.end(result, self.memory)
        self.pending.clear()
        if self.memory.get("strategy_fallbacks"):
            raise RuntimeError(f"{self.name} used a strategy fallback")

    def describe(self):
        return self.name


def score_match(job):
    # This script lives in tools, while the official modules are at the root.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import agent
    from game import AgentConfig
    import heldout_opponent as opponent
    from judge_engine import combine_halves, play_half

    mode, profile, seed, deals, output = job
    agent.ADAPTATION = mode == "adaptive"
    config = AgentConfig(num_hands=deals)
    bot = DirectAgent(mode, config, agent.new_half, agent.choose_action, agent.on_hand_end)
    challenger = DirectAgent(profile, config, opponent.new_half,
                             lambda obs, memory: opponent.opponent_action(obs, memory, profile),
                             lambda result, memory: None)
    result = combine_halves(play_half(bot, challenger, seed, 1, deals),
                           play_half(bot, challenger, seed, 2, deals))
    output.mkdir(parents=True, exist_ok=True)
    (output / "match.json").write_text(result.to_json(include_timing=False))
    return mode, profile, seed, result.mean_bb, result.verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", action="append", choices=("fixed", "adaptive"))
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--profile", action="append", choices=PROFILES)
    parser.add_argument("--fixed-entry", type=Path, default=ROOT / "agent.py")
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--direct-jobs", type=int, default=0,
                        help="Run score-only matches in this many workers, without HTTP timing")
    args = parser.parse_args()
    jobs = [(mode, profile, str(seed), args.deals,
             args.output.resolve() / mode / f"seed-{seed}" / profile)
            for seed in args.seed or SEEDS
            for profile in args.profile or PROFILES
            for mode in args.mode or ("fixed", "adaptive")]
    if args.direct_jobs:
        if args.fixed_entry != ROOT / "agent.py" or args.direct_jobs < 1:
            parser.error("direct mode requires positive workers and the default fixed entry")
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[variable] = "1"
        with ProcessPoolExecutor(max_workers=args.direct_jobs) as pool:
            for mode, profile, seed, score, verdict in pool.map(score_match, jobs):
                print(f"{mode:8} {profile:9} seed {seed}: {score:+.4f} bb/deal ({verdict})",
                      flush=True)
        return 0
    healthy = True
    for mode, profile, seed, deals, output in jobs:
        print(f"\n=== {mode} vs {profile}, seed {seed} ===", flush=True)
        entry = args.fixed_entry.resolve() if mode == "fixed" else ROOT / "agent.py"
        healthy &= run_match(
            entry, ROOT / "heldout_opponent.py", seed, deals, output,
            bot_env={"POKER_ADAPTATION": "1" if mode == "adaptive" else "0"},
            opponent_env={"POKER_HOLDOUT": profile},
        )
    return int(not healthy)


if __name__ == "__main__":
    raise SystemExit(main())
