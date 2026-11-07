"""Independent equity-based opponents for local duplicate matches."""

import os

import numpy as np
import uvicorn

import agent
from bot_equity import combination_index, draw_worlds, rank_tables, unseen_cards
from evaluator import NUM_CARDS
from game import Action, Street, default_action

PROFILES = ("value", "polarized", "trap", "switch")
_RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


def new_half(config, memory):
    memory["equity_cache"] = None
    memory["num_hands"] = config.num_hands


def opponent_action(obs, memory, profile):
    if profile == "switch":
        profile = "polarized" if obs.hand_index < memory["num_hands"] // 2 else "trap"
    raises = [action for action in _RAISES if action in obs.legal_actions]
    price = min(obs.my_stack, obs.to_call) / (obs.pot + min(obs.my_stack, obs.to_call))
    # A common per-deal stream keeps decisions reproducible across policy matchups.
    rng = np.random.default_rng((419, obs.hand_index, int(obs.street), int(obs.is_button)))
    mix = rng.random()
    if obs.street == Street.PREFLOP:
        uniform, opened, tight, features = agent.PREFLOP[combination_index(np.asarray(obs.hole_cards))]
        if not obs.raises_this_street:
            threshold = {"value": 0.535, "polarized": 0.484, "trap": 0.564}[profile]
            if not obs.is_button:
                threshold += 0.031
            if raises and uniform + 0.035 * (features - 0.5) > threshold:
                return raises[-1] if profile == "polarized" or uniform > 0.59 else raises[0]
            return (default_action(obs.to_call) if obs.is_button and uniform < 0.385
                    and (profile != "polarized" or mix < 0.30) else Action.CHECK_CALL)
        equity = 0.22 * uniform + 0.78 * (opened if obs.raises_this_street == 1 else tight)
        if raises and tight > {"value": 0.560, "polarized": 0.539, "trap": 0.588}[profile]:
            if profile != "trap" or mix < 0.60:
                return raises[-1] if tight > 0.584 or profile == "polarized" else raises[0]
        premium = 0.035 if obs.is_button else 0.073
        if min(obs.my_stack - obs.to_call, obs.opp_stack) == 0:
            premium = 0
        return Action.CHECK_CALL if equity >= price + premium else default_action(obs.to_call)

    key = (obs.hand_index, obs.hole_cards, obs.board)
    cached = memory["equity_cache"]
    if cached is None or cached[0] != key:
        tables = rank_tables()
        if obs.street == Street.RIVER:
            estimate = tables.river(obs.hole_cards, obs.board)
        else:
            worlds = draw_worlds(unseen_cards(obs.hole_cards, obs.board), 5 - len(obs.board), 256, rng)
            estimate = tables.sample(obs.hole_cards, obs.board, worlds)
        memory["equity_cache"] = key, estimate
    else:
        estimate = cached[1]

    q = float(estimate.frequency @ estimate.shares / estimate.trials)
    if obs.to_call:
        # Consider a range tilted toward value when facing action, with more
        # polarization on the river and more pressure after a reraise.
        strength = estimate.opponent_strength
        weights = estimate.frequency * (0.46 + (0.9 + 1.8 * price) * strength ** 2.7)
        defended = float(weights @ estimate.shares / weights.sum())
        cushion = 0.0
        if obs.street != Street.RIVER and min(obs.my_stack - obs.to_call, obs.opp_stack) > 0:
            cushion = 0.038 if obs.is_button else 0.068
        raise_at = {"value": 0.69, "polarized": 0.64, "trap": 0.73}[profile]
        if raises and (estimate.nuts or defended >= raise_at):
            return raises[-1] if defended >= 0.80 or profile != "value" else raises[0]
        if profile == "polarized":
            cushion -= 0.023
        return Action.CHECK_CALL if defended >= price + cushion else default_action(obs.to_call)

    if not raises:
        return Action.CHECK_CALL
    if profile == "trap":
        if q >= 0.77 and mix < (0.76 if obs.is_button else 0.30):
            return raises[-1]
        if q >= 0.61 and mix < 0.22:
            return raises[0]
        return Action.CHECK_CALL
    if profile == "polarized":
        blocker = any(
            obs.board.count(card) == 0
            and sum((c & 3) == (card & 3) for c in obs.board) >= 3
            for card in obs.hole_cards if card >> 2 == NUM_CARDS // 4 - 1
        )
        bluff = (q < 0.31 and mix < (0.34 if blocker else 0.13))
        if estimate.nuts or q >= 0.72 or bluff:
            return raises[-1]
        if (q >= 0.55 or (q >= 0.40 and estimate.redraw > 0.32)) and mix < 0.65:
            return raises[0]
        return Action.CHECK_CALL
    if q >= 0.79:
        return raises[-1]
    if q >= (0.57 if obs.is_button else 0.64):
        return raises[0]
    return Action.CHECK_CALL


if __name__ == "__main__":
    profile = os.environ.get("POKER_HOLDOUT")
    if profile not in PROFILES:
        raise ValueError(f"Set POKER_HOLDOUT to one of {PROFILES}")
    agent.AGENT_NAME = f"Held-out {profile}"
    agent.new_half = new_half
    agent.choose_action = lambda obs, memory: opponent_action(obs, memory, profile)
    agent.on_hand_end = lambda result, memory: None
    port = int(os.environ.get("PORT", "5009"))
    uvicorn.run(agent.app, host="127.0.0.1", port=port, access_log=False)
