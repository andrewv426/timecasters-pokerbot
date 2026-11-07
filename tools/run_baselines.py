"""Run the official duplicate judge against the four published baseline bots."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("random", "calling_station", "always_half", "always_pot")


def local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def running_bot(script: Path, env: dict, log_path: Path):
    port = local_port()
    token = secrets.token_urlsafe(24)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(script)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**env, "PORT": str(port), "AGENT_TOKEN": token},
        )
        try:
            started = time.monotonic()
            while process.poll() is None and time.monotonic() - started < 15:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/", timeout=0.2) as response:
                        if response.status == 200:
                            break
                except (OSError, URLError):
                    time.sleep(0.05)
            else:
                raise RuntimeError(f"{script.name} failed startup. See {log_path}")
            yield port, token
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def run_match(bot: Path, opponent: Path, seed: str, deals: int, output: Path, *,
              bot_env: dict | None = None, opponent_env: dict | None = None) -> bool:
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    judge = [sys.executable, str(ROOT / "judge_engine.py")]
    for half in (1, 2):
        with running_bot(bot, {**env, **(bot_env or {})}, output / f"bot-h{half}.log") as (port1, token1):
            with running_bot(opponent, {**env, **(opponent_env or {})},
                             output / f"baseline-h{half}.log") as (port2, token2):
                subprocess.run(
                    judge + ["--half", str(half), "--seed", seed, "--deals", str(deals),
                             "--agent1", str(port1), "--agent2", str(port2),
                             "--json", str(output / f"h{half}.json")],
                    cwd=ROOT, env={**env, "AGENT1_TOKEN": token1, "AGENT2_TOKEN": token2},
                    check=True,
                )
    match_path = output / "match.json"
    subprocess.run(
        judge + ["combine", str(output / "h1.json"), str(output / "h2.json"),
                 "--json", str(match_path)],
        cwd=ROOT, env=env, check=True,
    )
    result = json.loads(match_path.read_text())
    healthy = all(
        not (half["errors"] or half["forfeited"] or half["timed_out"] or half["skipped_calls"])
        for halves in result["agent_stats"].values() for half in halves
    )
    fallback = any(
        "strategy fell back" in (output / f"bot-h{half}.log").read_text()
        for half in (1, 2)
    )
    if fallback:
        print(f"Strategy fallback logged in {output}", file=sys.stderr)
    return healthy and not fallback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot", type=Path, default=ROOT / "agent.py")
    parser.add_argument("--baseline", choices=BASELINES, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    healthy = True
    for seed in args.seed or [7]:
        for baseline in args.baseline or BASELINES:
            print(f"\n=== {baseline}, seed {seed}, {args.deals} duplicate deals ===", flush=True)
            match_dir = args.output.resolve() / f"seed-{seed}" / baseline
            opponent = ROOT / "baseline-bots" / f"baseline_{baseline}.py"
            healthy &= run_match(args.bot.resolve(), opponent, str(seed), args.deals, match_dir)
    return int(not healthy)


if __name__ == "__main__":
    raise SystemExit(main())
