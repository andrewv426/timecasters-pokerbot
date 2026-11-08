"""A check/call-or-fold river rollout for marginal big-blind turn calls."""

import numpy as np

from big_blind import midranks
from bot_equity import card_sets, combination_index, rank_tables, showdown_share, unseen_cards
from game import Observation
from opponent_model import OpponentModel

_PAIRS = card_sets(5, 2)
_TRIPLES = card_sets(5, 3)


def river_check_value(pot: int, stack: int, weights: np.ndarray, shares: np.ndarray,
                      bet_probability: np.ndarray) -> float:
    """Unnormalized value of checking and then responding to a river bet."""
    bet_weights = weights * bet_probability
    check_value = float((weights - bet_weights) @ shares) * pot
    wins = float(bet_weights @ shares)
    mass = float(bet_weights.sum())
    half = min(stack, pot // 2)
    full = min(stack, pot)
    # Decide at the public bet size, averaging over all hands that could bet.
    half_call = max(0.0, wins * (pot + 2 * half) - mass * half)
    full_call = max(0.0, wins * (pot + 2 * full) - mass * full)
    return check_value + 0.5 * (half_call + full_call)


def turn_call_value(obs: Observation, model: OpponentModel, hands: np.ndarray,
                    weights: np.ndarray) -> float:
    call = min(obs.to_call, obs.my_stack)
    pot = obs.pot + call
    stack = min(obs.my_stack - call, obs.opp_stack)
    rivers = unseen_cards(obs.hole_cards, obs.board)
    board = np.column_stack((np.broadcast_to(obs.board, (len(rivers), 4)), rivers))
    triples = combination_index(board[:, _TRIPLES])
    tables = rank_tables()
    hero = tables.best(combination_index(np.asarray(obs.hole_cards)[_PAIRS]), triples)
    villain_pairs = combination_index(hands[:, _PAIRS])
    villain = tables.best(villain_pairs[None, :, :], triples[:, None, :])
    shares = showdown_share(hero[:, None], villain)
    legal = ~(hands[None, :, :] == rivers[:, None, None]).any(axis=2)
    river_weights = legal * weights
    bet_rate, _ = model.estimate(("bet", 2, 1), obs.hand_index, 0.36)

    value = 0.0
    total = 0.0
    for rank, mass, outcome in zip(villain, river_weights, shares):
        count = float(mass.sum())
        if not count:
            continue
        strength = midranks(rank, mass)
        tendency = 0.30 + 1.80 * strength ** 3 + 0.45 * (1 - strength) ** 4
        probability = np.minimum(0.96, bet_rate * tendency * count / (mass @ tendency))
        value += river_check_value(pot, stack, mass, outcome, probability)
        total += count
    return value / total - call
