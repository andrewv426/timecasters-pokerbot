"""Conservative defense against costly turn and river betting lines."""

import numpy as np

from game import Action, Observation, Street
from opponent_model import OpponentModel, phase

_RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


def pressure(obs: Observation, model: OpponentModel | None) -> float:
    """Strength of additional evidence that a late bet represents value."""
    if obs.street < Street.TURN or not obs.to_call:
        return 0.0
    call = min(obs.to_call, obs.my_stack)
    price = call / (obs.pot + call)
    earlier_bets = {a.street for a in obs.action_history
                    if not a.is_me and a.action in _RAISES and a.street < obs.street}
    if obs.street == Street.TURN:
        if min(obs.my_stack - call, obs.opp_stack) <= 0 or price < 0.29:
            return 0.0
        if not earlier_bets and obs.raises_this_street == 1:
            return 0.0
        strength = 0.52
    elif price >= 0.29:
        strength = 0.72 if Street.TURN in earlier_bets or obs.raises_this_street > 1 else 0.42
    elif price >= 0.21 and Street.TURN in earlier_bets:
        strength = 0.32
    else:
        return 0.0

    event = model.last_event(obs.street) if model is not None else None
    if event is None:
        return strength
    key = event.bet_context()
    prior = 0.13 if key[0] == "reraise" else (0.36 if event.button else 0.27)
    rate, trust = model.estimate(key, obs.hand_index, prior)
    weak, weak_trust = model.estimate(
        ("weak_bet", phase(obs.street), int(event.button)), obs.hand_index, 0.16
    )
    # Frequent betting and confirmed weak showdowns both temper the adjustment.
    aggression_discount = min(0.35, 1.15 * trust * max(0.0, rate - prior))
    bluff_discount = min(0.35, 1.40 * weak_trust * max(0.0, weak - 0.16))
    return strength * (1 - aggression_discount) * (1 - bluff_discount)


def defensive_weights(weights: np.ndarray, opponent_strength: np.ndarray,
                      severity: float) -> np.ndarray:
    if not severity:
        return weights
    value_range = weights * (0.20 + 1.60 * opponent_strength ** 4)
    value_range *= weights.sum() / value_range.sum()
    return (1 - severity) * weights + severity * value_range


def river_pressure_cost(obs: Observation, severity: float, redraw: float) -> float:
    """Reserve some of the turn equity when another meaningful bet can follow."""
    if obs.street != Street.TURN or not severity:
        return 0.0
    call = min(obs.to_call, obs.my_stack)
    left = min(obs.my_stack - call, obs.opp_stack)
    future_bet = min(left, 0.50 * (obs.pot + call))
    return max(0.0, future_bet) * 0.13 * severity * (1 - 0.6 * redraw)
