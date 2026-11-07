"""Equity-based preflop and postflop decisions for the tournament API."""

from collections import Counter
import logging
from pathlib import Path
from time import perf_counter

import numpy as np

from bot_equity import Equity, combination_index, draw_worlds, rank_tables, unseen_cards
from game import Action, AgentConfig, Observation, Street, default_action
from opponent_model import OpponentModel, phase, size_bucket

_LOG = logging.getLogger(__name__)
_PREFLOP_PATH = Path(__file__).with_name("data") / "preflop.npz"
_RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


def load_preflop() -> np.ndarray:
    with np.load(_PREFLOP_PATH, allow_pickle=False) as asset:
        equity = asset["equity"]
    if equity.shape != (376_992, 4) or not np.isfinite(equity).all():
        raise ValueError("Invalid preflop equity asset")
    equity.setflags(write=False)
    return equity


def legal_raises(obs: Observation) -> list[tuple[Action, int]]:
    sizes = []
    for action in _RAISES:
        if action in obs.legal_actions:
            target = obs.raise_to[action]
            if not sizes or target != sizes[-1][1]:
                sizes.append((action, target))
    return sizes


def preflop_action(obs: Observation, table: np.ndarray, rng: np.random.Generator,
                   model: OpponentModel | None = None) -> Action:
    uniform, open_equity, tight, playable = table[combination_index(np.asarray(obs.hole_cards))]
    raises = legal_raises(obs)
    if obs.raises_this_street == 0:
        score = uniform + 0.045 * (playable - 0.5)
        if raises:
            pot_threshold = 0.568 if obs.is_button else 0.579
            half_threshold = 0.510 if obs.is_button else 0.539
            if score >= pot_threshold and rng.random() < 0.87:
                return raises[-1][0]
            if score >= half_threshold and rng.random() < 0.78:
                return raises[0][0]
        if obs.is_button and uniform < 0.34 and playable < 0.27:
            return default_action(obs.to_call)
        return Action.CHECK_CALL

    call = min(obs.to_call, obs.my_stack)
    price = call / (obs.pot + call)
    spr = min(obs.my_stack - call, obs.opp_stack) / (obs.pot + call)
    range_equity = open_equity if obs.raises_this_street == 1 else tight
    equity = 0.32 * uniform + 0.68 * range_equity
    event = model.last_event(Street.PREFLOP) if model else None
    if event and event.action in _RAISES:
        key = event.bet_context()
        prior = {"open": 0.49, "isolate": 0.28, "reraise": 0.16}[key[0]]
        rate, trust = model.estimate(key, obs.hand_index, prior)
        if trust:
            range_weight = min(0.94, max(0.30, 0.68 - 1.35 * (rate - prior)))
            equity = (1 - range_weight) * uniform + range_weight * range_equity
    position_cost = (0.021 if obs.is_button else 0.056) * min(1.0, spr / 2)
    if raises and tight >= (0.573 if obs.raises_this_street == 1 else 0.592):
        if tight >= 0.615 and rng.random() < 0.83:
            return raises[-1][0]
        if rng.random() < 0.76:
            return raises[0][0]
    if equity >= price + position_cost:
        return Action.CHECK_CALL
    return default_action(obs.to_call)


def continuation_cost(obs: Observation, pot: int, added: int, opponent_added: int,
                      redraw: float) -> float:
    """Equity cushion for exposed future betting and position."""
    if obs.street == Street.RIVER:
        return 0.0
    remaining = min(obs.my_stack - added, obs.opp_stack - opponent_added)
    in_position = 0.022 if obs.is_button else 0.061
    return max(0.0, in_position - 0.026 * redraw) * min(1.0, remaining / (2 * pot))


def raise_value(obs: Observation, target: int, config: AgentConfig, estimate: Equity,
                weights: np.ndarray, fold_shift: float = 0.0, reraise_shift: float = 0.0) -> float:
    paid = target - obs.my_bet
    matched = min(obs.opp_stack, target - obs.opp_bet)
    final_pot = obs.pot + paid + matched
    strength = estimate.opponent_strength
    continue_rate = np.clip(
        0.18 + 0.90 * strength - 0.66 * matched / final_pot - fold_shift, 0.07, 0.96
    )
    reraise_rate = 0.0
    reraise_value = 0.0
    if (obs.raises_this_street + 1 < config.raise_cap
            and obs.my_stack > paid and obs.opp_stack > matched):
        reraise_rate = 0.025 + 0.24 * strength ** 3
        if reraise_shift:
            reraise_rate = np.clip(reraise_rate + reraise_shift, 0.01, 0.40)
        reraise_to = min(obs.opp_bet + obs.opp_stack, target + final_pot)
        extra = min(obs.my_stack - paid, reraise_to - target)
        reraised_pot = obs.pot + paid + reraise_to - obs.opp_bet + extra
        reraise_weights = weights * continue_rate * reraise_rate
        equity = float(reraise_weights @ estimate.shares / reraise_weights.sum())
        equity -= continuation_cost(obs, reraised_pot, paid + extra,
                                    reraise_to - obs.opp_bet, estimate.redraw)
        reraise_value = reraise_weights.sum() * max(
            -paid, equity * reraised_pot - paid - extra
        )
    call_weights = weights * continue_rate * (1 - reraise_rate)
    fold_weight = weights @ (1 - continue_rate)
    cushion = continuation_cost(obs, final_pot, paid, matched, estimate.redraw)
    call_value = call_weights @ ((estimate.shares - cushion) * final_pot - paid)
    return float((fold_weight * obs.pot + call_value + reraise_value) / weights.sum())


def betting_weights(obs: Observation, estimate: Equity,
                    model: OpponentModel | None) -> np.ndarray:
    strength = estimate.opponent_strength
    if obs.to_call:
        # A fixed, deliberately broad betting prior keeps bluff catchers in range.
        pressure = min(1.20, 0.47 + 0.18 * obs.raises_this_street + 0.65 * obs.to_call / obs.pot)
        weights = estimate.frequency * (0.78 + pressure * strength ** 2)
    else:
        weights = estimate.frequency
    event = model.last_event(obs.street) if model else None
    if not event or not event.can_raise:
        return weights
    bet = obs.to_call and event.action in _RAISES
    checked = not obs.to_call and event.action == Action.CHECK_CALL and not event.due
    if not (bet or checked):
        return weights

    key = event.bet_context()
    prior = 0.13 if key[0] == "reraise" else (0.36 if event.button else 0.27)
    rate, trust = model.estimate(key, obs.hand_index, prior)
    if not trust:
        return weights
    weak, weak_trust = model.estimate(
        ("weak_bet", phase(obs.street), int(event.button)), obs.hand_index, 0.16
    )
    # Keep a broad bluff component and add weight only for affirmative weak
    # showdown evidence. Showdown samples favor hands strong enough to continue.
    floor = 0.13 + 0.8 * weak_trust * max(0.0, weak - 0.16)
    tail = np.clip((strength - (1 - rate) + 0.16) / 0.32, 0.0, 1.0)
    likelihood = floor + (1 - floor) * tail
    if checked:
        likelihood = 1.04 - likelihood
    learned = estimate.frequency * likelihood
    blend = (0.76 if bet else 0.46) * trust
    return (1 - blend) * weights + blend * (weights.sum() / learned.sum()) * learned


def response_shifts(obs: Observation, target: int,
                    model: OpponentModel | None) -> tuple[float, float]:
    if model is None:
        return 0.0, 0.0
    due = min(obs.opp_stack, target - obs.opp_bet)
    price = due / (obs.pot + target - obs.my_bet + due)
    lane = size_bucket(price)
    stage = phase(obs.street)
    fold_prior = 0.54 + 0.06 * lane + 0.015 * (obs.street == Street.RIVER)
    fold, _ = model.estimate(("fold", stage, lane), obs.hand_index, fold_prior)
    reraise, _ = model.estimate(("reraise", stage, lane), obs.hand_index, 0.105)
    return min(0.22, max(-0.22, fold - fold_prior)), min(0.12, max(-0.075, reraise - 0.105))


def has_river_blocker(obs: Observation) -> bool:
    suits = Counter(card & 3 for card in obs.board)
    ranks = Counter(card >> 2 for card in obs.board)
    return any((card >> 2 == 8 and suits[card & 3] >= 3)
               or ranks[card >> 2] >= 2 for card in obs.hole_cards)


def postflop_action(obs: Observation, config: AgentConfig, estimate: Equity,
                    mix: float, model: OpponentModel | None = None) -> tuple[Action, float]:
    weights = betting_weights(obs, estimate, model)
    equity = float(weights @ estimate.shares / weights.sum())
    call = min(obs.to_call, obs.my_stack)
    pot_if_called = obs.pot + call
    cushion = continuation_cost(obs, pot_if_called, call, 0, estimate.redraw)
    passive_value = max(0.0, equity - cushion) * pot_if_called - call
    candidates = [(passive_value, Action.CHECK_CALL)]
    if Action.FOLD in obs.legal_actions:
        candidates.append((0.0, Action.FOLD))

    value_threshold = 0.56 if obs.to_call else 0.53
    if obs.street == Street.RIVER:
        value_threshold += 0.025
    # Mix some strong hands into the checking range and allow selective redraw bets.
    value_hand = equity >= value_threshold and (obs.to_call > 0 or mix >= 0.045)
    semi_bluff = (not obs.to_call and obs.street != Street.RIVER
                  and equity >= 0.43 and estimate.made_equity < 0.40
                  and estimate.redraw >= 0.35 and mix > 0.92)
    for action, target in legal_raises(obs):
        fold_shift, reraise_shift = response_shifts(obs, target, model)
        thin_value = (fold_shift < -0.10 and not obs.to_call and mix >= 0.045
                      and equity >= value_threshold - 0.035)
        extra_bluff = (fold_shift > 0.095 and not obs.to_call and mix > 0.87
                       and ((obs.street != Street.RIVER and equity >= 0.32 and estimate.redraw > 0.28)
                            or (obs.street == Street.RIVER and equity < 0.35 and has_river_blocker(obs))))
        if value_hand or semi_bluff or estimate.nuts or thin_value or extra_bluff:
            risk = (0.045 if obs.to_call else 0.025) * (target - obs.my_bet)
            if extra_bluff and not (value_hand or semi_bluff or estimate.nuts):
                risk += 0.06 * obs.pot
            value = raise_value(obs, target, config, estimate, weights, fold_shift, reraise_shift)
            candidates.append((value - risk, action))
    candidates.sort(key=lambda choice: choice[0], reverse=True)
    margin = candidates[0][0] - candidates[1][0] if len(candidates) > 1 else float("inf")
    return candidates[0][1], margin


def policy_action(obs: Observation, memory: dict, preflop: np.ndarray) -> Action:
    config = memory["config"]
    rng = memory["rng"]
    model = memory.get("opponent_model")
    if obs.street == Street.PREFLOP:
        return preflop_action(obs, preflop, rng, model)

    tables = rank_tables()
    key = (obs.hand_index, obs.hole_cards, obs.board)
    if memory["equity_key"] != key:
        memory["equity_key"] = key
        memory["equity"] = None
    estimate = memory["equity"]
    headroom = memory["strategy_seconds"] < 0.60 * config.time_bank_seconds
    mix = rng.random()
    if estimate is None:
        if obs.street == Street.RIVER and headroom:
            estimate = tables.river(obs.hole_cards, obs.board)
        else:
            remaining = unseen_cards(obs.hole_cards, obs.board)
            sample_size = (192 if obs.street == Street.FLOP else 256) if headroom else 64
            worlds = draw_worlds(remaining, 5 - len(obs.board), sample_size, rng)
            estimate = tables.sample(obs.hole_cards, obs.board, worlds)
        memory["equity"] = estimate

    action, margin = postflop_action(obs, config, estimate, mix, model)
    if (headroom and obs.street != Street.RIVER and estimate.trials < 384
            and obs.pot >= 12 and margin < 0.032 * obs.pot):
        remaining = unseen_cards(obs.hole_cards, obs.board)
        worlds = draw_worlds(remaining, 5 - len(obs.board), 384, rng)
        estimate = tables.sample(obs.hole_cards, obs.board, worlds)
        memory["equity"] = estimate
        action, _ = postflop_action(obs, config, estimate, mix, model)
    return action


def choose_legal_action(obs: Observation, memory: dict, preflop: np.ndarray) -> Action:
    start = perf_counter()
    try:
        model = memory.get("opponent_model")
        if model:
            model.observe(obs)
        action = policy_action(obs, memory, preflop)
        if action not in obs.legal_actions:
            raise ValueError(f"Policy returned illegal action {action}")
        if model:
            model.remember(obs, action)
        return action
    except Exception:
        memory["strategy_fallbacks"] += 1
        if memory["strategy_fallbacks"] == 1:
            _LOG.exception("Equity strategy fell back to check or fold")
        return default_action(obs.to_call)
    finally:
        memory["strategy_seconds"] += perf_counter() - start
