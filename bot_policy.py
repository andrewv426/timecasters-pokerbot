"""Equity-based preflop and postflop decisions for the tournament API."""

from collections import Counter
import logging
from pathlib import Path
from time import perf_counter

import numpy as np

from big_blind import BigBlindRange, realized_equity
from bot_equity import Equity, combination_index, draw_worlds, rank_tables, unseen_cards
from game import Action, AgentConfig, Observation, Street, default_action
from late_defense import defensive_weights, pressure, river_pressure_cost
from opponent_model import OpponentModel, phase, size_bucket
from turn_lookahead import turn_call_value

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
                   model: OpponentModel | None = None, improve_bb: bool = False,
                   flop_pressure: bool = False) -> Action:
    uniform, open_equity, tight, playable = table[combination_index(np.asarray(obs.hole_cards))]
    raises = legal_raises(obs)
    if obs.raises_this_street == 0:
        score = uniform + 0.045 * (playable - 0.5)
        if raises:
            pot_threshold = 0.568 if obs.is_button else 0.579
            half_threshold = 0.510 if obs.is_button else 0.539
            if improve_bb and flop_pressure:
                pot_threshold, half_threshold = 0.545, 0.527
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
    rate = 0.49
    event = model.last_event(Street.PREFLOP) if model else None
    if event and event.action in _RAISES:
        key = event.bet_context()
        prior = {"open": 0.49, "isolate": 0.28, "reraise": 0.16}[key[0]]
        rate, trust = model.estimate(key, obs.hand_index, prior)
        if trust:
            range_weight = min(0.94, max(0.30, 0.68 - 1.35 * (rate - prior)))
            equity = (1 - range_weight) * uniform + range_weight * range_equity
    position_cost = (0.021 if obs.is_button else 0.056) * min(1.0, spr / 2)
    if improve_bb and flop_pressure and raises and obs.raises_this_street == 1:
        # Give playable hands with equity against the continuing range a
        # chance to take the initiative against an opening raise.
        value = 0.48 * equity + 0.52 * tight + 0.06 * (playable - 0.5)
        threshold = 0.546 - 0.055 * (rate - 0.49)
        if value >= threshold and rng.random() < 0.84:
            if value >= threshold + 0.014:
                return raises[-1][0]
            return raises[0][0]
    if raises and tight >= (0.573 if obs.raises_this_street == 1 else 0.592):
        if tight >= 0.615 and rng.random() < 0.83:
            return raises[-1][0]
        if rng.random() < 0.76:
            return raises[0][0]
    if improve_bb:
        can_call = realized_equity(float(equity), spr, Street.PREFLOP, float(playable)) >= price
    else:
        can_call = equity >= price + position_cost
    if can_call:
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


def flop_equity(equity: float, spr: float, redraw: float, is_button: bool) -> float:
    """Realize less with marginal hands and more with strong hands in deep pots."""
    if spr <= 0:
        return equity
    position_cost = 0.07 if is_button else 0.15
    friction = position_cost + 1.65 * (0.50 - equity) - 0.16 * (redraw - 0.30)
    friction = min(0.52, max(-0.28, friction))
    exposure = spr / (spr + 1.1)
    return min(1.0, equity * (1 - exposure * friction))


def raise_value(obs: Observation, target: int, config: AgentConfig, estimate: Equity,
                weights: np.ndarray, fold_shift: float = 0.0, reraise_shift: float = 0.0,
                improve_bb: bool = False, flop_model: OpponentModel | None = None) -> float:
    paid = target - obs.my_bet
    matched = min(obs.opp_stack, target - obs.opp_bet)
    final_pot = obs.pot + paid + matched
    strength = estimate.opponent_strength
    continue_rate = np.clip(
        0.18 + 0.90 * strength - 0.66 * matched / final_pot - fold_shift, 0.07, 0.96
    )
    if improve_bb:
        cheap_call = max(0.0, 1 - (matched / final_pot) / 0.20)
        # Discount projected folds when the big blind would build a pot
        # out of position, with a larger adjustment for cheap calls.
        continue_rate += (0.12 + 0.73 * cheap_call) * (1 - continue_rate)
    if flop_model is not None and not obs.to_call:
        projected_fold = float(weights @ (1 - continue_rate) / weights.sum())
        lane = size_bucket(matched / final_pot)
        fold, _ = flop_model.estimate(("flop_fold", int(obs.is_button), lane),
                                      obs.hand_index, projected_fold)
        continue_rate = np.clip(continue_rate + projected_fold - fold, 0.04, 0.98)
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
        if improve_bb or flop_model is not None:
            left = min(obs.my_stack - paid - extra, obs.opp_stack - reraise_to + obs.opp_bet)
            if flop_model is not None:
                equity = flop_equity(equity, left / reraised_pot, estimate.redraw, obs.is_button)
            else:
                equity = realized_equity(equity, left / reraised_pot, obs.street, estimate.redraw)
        else:
            equity -= continuation_cost(obs, reraised_pot, paid + extra,
                                        reraise_to - obs.opp_bet, estimate.redraw)
        reraise_value = reraise_weights.sum() * max(
            -paid, equity * reraised_pot - paid - extra
        )
    call_weights = weights * continue_rate * (1 - reraise_rate)
    fold_weight = weights @ (1 - continue_rate)
    if improve_bb or flop_model is not None:
        call_mass = float(call_weights.sum())
        called_equity = float(call_weights @ estimate.shares / call_mass)
        left = min(obs.my_stack - paid, obs.opp_stack - matched)
        if flop_model is not None:
            equity = flop_equity(called_equity, left / final_pot, estimate.redraw, obs.is_button)
        else:
            equity = realized_equity(called_equity, left / final_pot, obs.street, estimate.redraw)
        call_value = call_mass * (equity * final_pot - paid)
    else:
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
    rate, trust = model.estimate_for(obs, key, prior)
    if not trust:
        return weights
    weak, weak_trust = model.estimate_for(
        obs, ("weak_bet", phase(obs.street), int(event.button)), 0.16
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
    fold, _ = model.estimate_for(obs, ("fold", stage, lane), fold_prior)
    reraise, _ = model.estimate_for(obs, ("reraise", stage, lane), 0.105)
    return min(0.22, max(-0.22, fold - fold_prior)), min(0.12, max(-0.075, reraise - 0.105))


def has_river_blocker(obs: Observation) -> bool:
    suits = Counter(card & 3 for card in obs.board)
    ranks = Counter(card >> 2 for card in obs.board)
    return any((card >> 2 == 8 and suits[card & 3] >= 3)
               or ranks[card >> 2] >= 2 for card in obs.hole_cards)


def postflop_action(obs: Observation, config: AgentConfig, estimate: Equity,
                    mix: float, model: OpponentModel | None = None,
                    improve_bb: bool = False, guard_late_calls: bool = False,
                    flop_pressure: bool = False,
                    turn_hands: np.ndarray | None = None) -> tuple[Action, float]:
    # The big-blind range already includes the current action.
    weights = estimate.frequency if improve_bb else betting_weights(obs, estimate, model)
    severity = pressure(obs, model) if guard_late_calls else 0.0
    if (model and model.turn_bet_strength and model.trace and obs.street == Street.RIVER
            and obs.to_call and obs.raises_this_street == 1):
        turn_size = next((e.turn_bet_size for e in model.trace.events
                          if e.turn_bet_size is not None), None)
        if turn_size is not None:
            signal = model.turn_size_signal(obs.hand_index)
            price = obs.to_call / (obs.pot + obs.to_call)
            locked = obs.my_stack <= obs.to_call or not obs.opp_stack
            if turn_size == 0 and locked and price >= 0.12:
                severity = max(severity, signal)
            if turn_size == 1 and not locked and price >= 0.29 and signal:
                # A second full-sized bet can still be a bluff on this line.
                severity *= 1 - signal / 0.72
                weak = weights * (0.30 + (1 - estimate.opponent_strength) ** 2)
                weights = (1 - signal) * weights + signal * (weights.sum() / weak.sum()) * weak
    weights = defensive_weights(weights, estimate.opponent_strength, severity)
    equity = float(weights @ estimate.shares / weights.sum())
    call = min(obs.to_call, obs.my_stack)
    pot_if_called = obs.pot + call
    flop_model = model if flop_pressure and obs.street == Street.FLOP else None
    if flop_model is not None:
        left = min(obs.my_stack - call, obs.opp_stack)
        available = flop_equity(equity, left / pot_if_called, estimate.redraw, obs.is_button)
    elif improve_bb:
        left = min(obs.my_stack - call, obs.opp_stack)
        available = realized_equity(equity, left / pot_if_called, obs.street, estimate.redraw)
    else:
        cushion = continuation_cost(obs, pot_if_called, call, 0, estimate.redraw)
        available = max(0.0, equity - cushion)
    passive_value = available * pot_if_called - call - river_pressure_cost(
        obs, severity, estimate.redraw
    )
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
    pressure_lead = (flop_model is not None and not obs.to_call
                     and (equity >= 0.32 or mix < 0.24))
    for action, target in legal_raises(obs):
        fold_shift, reraise_shift = response_shifts(obs, target, model)
        thin_value = (fold_shift < -0.10 and not obs.to_call and mix >= 0.045
                      and equity >= value_threshold - 0.035)
        extra_bluff = (fold_shift > 0.095 and not obs.to_call and mix > 0.87
                       and ((obs.street != Street.RIVER and equity >= 0.32 and estimate.redraw > 0.28)
                            or (obs.street == Street.RIVER and equity < 0.35 and has_river_blocker(obs))))
        if value_hand or semi_bluff or estimate.nuts or thin_value or extra_bluff or pressure_lead:
            risk = (0.045 if obs.to_call else 0.025) * (target - obs.my_bet)
            if extra_bluff and not (value_hand or semi_bluff or estimate.nuts):
                risk += 0.06 * obs.pot
            elif pressure_lead and not (value_hand or semi_bluff or thin_value):
                risk += 0.055 * obs.pot
            value = raise_value(obs, target, config, estimate, weights, fold_shift, reraise_shift,
                                improve_bb, flop_model)
            candidates.append((value - risk, action))
    candidates.sort(key=lambda choice: choice[0], reverse=True)
    if (turn_hands is not None and model is not None and obs.street == Street.TURN
            and not obs.is_button and obs.raises_this_street == 1
            and call >= 3 * obs.big_blind and call / pot_if_called >= 0.23
            and min(obs.my_stack - call, obs.opp_stack) >= 2 * obs.big_blind
            and abs(passive_value) <= 0.07 * pot_if_called
            and {choice[1] for choice in candidates[:2]} == {Action.CHECK_CALL, Action.FOLD}):
        projected = turn_call_value(obs, model, turn_hands, weights)
        # The rollout assumes we check the river. Blend it with the baseline
        # to allow for river leads and uncertainty about future bet sizes.
        value = 0.35 * passive_value + 0.65 * projected
        candidates = [(value if action == Action.CHECK_CALL else ev, action)
                      for ev, action in candidates]
        candidates.sort(key=lambda choice: choice[0], reverse=True)
    margin = candidates[0][0] - candidates[1][0] if len(candidates) > 1 else float("inf")
    return candidates[0][1], margin


def policy_action(obs: Observation, memory: dict, preflop: np.ndarray) -> Action:
    config = memory["config"]
    rng = memory["rng"]
    model = memory.get("opponent_model")
    improve_bb = memory.get("improve_big_blind", False) and not obs.is_button and model is not None
    guard_late_calls = memory.get("guard_late_calls", False)
    flop_pressure = memory.get("flop_pressure", False)
    if obs.street == Street.PREFLOP:
        return preflop_action(obs, preflop, rng, model, improve_bb, flop_pressure)

    if improve_bb and memory["strategy_seconds"] < 0.52 * config.time_bank_seconds:
        key = (obs.hand_index, obs.hole_cards, obs.board)
        cache = memory.get("big_blind_range")
        mix = rng.random()
        if cache is None or cache[0] != key:
            remaining = unseen_cards(obs.hole_cards, obs.board)
            samples = {Street.FLOP: 224, Street.TURN: 288, Street.RIVER: 512}[obs.street]
            worlds = draw_worlds(remaining, 5 - len(obs.board), samples, rng)
            cache = key, BigBlindRange.build(obs, preflop, worlds)
            memory["big_blind_range"] = cache
        estimate = cache[1].condition(obs, model)
        turn_hands = None
        if (memory.get("turn_lookahead", False)
                and memory["strategy_seconds"] < 0.42 * config.time_bank_seconds):
            turn_hands = cache[1].turn_hands
        return postflop_action(obs, config, estimate, mix, model, improve_bb=True,
                               guard_late_calls=guard_late_calls, flop_pressure=flop_pressure,
                               turn_hands=turn_hands)[0]

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

    action, margin = postflop_action(obs, config, estimate, mix, model,
                                     guard_late_calls=guard_late_calls, flop_pressure=flop_pressure)
    if (headroom and obs.street != Street.RIVER and estimate.trials < 384
            and obs.pot >= 12 and margin < 0.032 * obs.pot):
        remaining = unseen_cards(obs.hole_cards, obs.board)
        worlds = draw_worlds(remaining, 5 - len(obs.board), 384, rng)
        estimate = tables.sample(obs.hole_cards, obs.board, worlds)
        memory["equity"] = estimate
        action, _ = postflop_action(obs, config, estimate, mix, model,
                                   guard_late_calls=guard_late_calls, flop_pressure=flop_pressure)
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
