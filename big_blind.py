"""Action-conditioned range and equity realization for the big blind."""

from dataclasses import dataclass, replace

import numpy as np

from bot_equity import Equity, card_sets, combination_index, rank_tables
from game import Action, BOARD_CARDS_VISIBLE, Observation, Street
from opponent_model import Event, OpponentModel, phase

_PAIRS = card_sets(5, 2)
_TRIPLES = {n: card_sets(n, 3) for n in (3, 4, 5)}
_RAISES = (Action.RAISE_HALF, Action.RAISE_POT)
_RIVER_BINS = 16


def realized_equity(equity: float, spr: float, street: Street, potential: float) -> float:
    """Discount future pressure on marginal hands, especially in deeper pots."""
    if street == Street.RIVER or spr <= 0:
        return equity
    if street == Street.PREFLOP:
        friction = 0.19 + 0.30 * (0.5 - potential) + 0.24 * (0.5 - equity)
    else:
        friction = 0.15 + 0.85 * (0.50 - equity) - 0.15 * (potential - 0.30)
    exposure = spr / (spr + 1.1)
    return min(1.0, equity * (1 - exposure * min(0.38, max(-0.10, friction))))


def midranks(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    prefix = np.concatenate(([0.0], weights[order].cumsum()))
    low = sorted_values.searchsorted(values, "left")
    high = sorted_values.searchsorted(values, "right")
    return (prefix[low] + prefix[high]) / (2 * prefix[-1])


def _soft_tail(percentile: np.ndarray, fraction: float, width: float) -> np.ndarray:
    if fraction <= 0:
        return np.zeros_like(percentile)
    return np.clip(0.5 + (percentile - (1.0 - fraction)) / width, 0.0, 1.0)


def _likelihood(event: Event, strengths: np.ndarray, weights: np.ndarray,
                model: OpponentModel, hand: int) -> np.ndarray:
    kind, stage, lane = event.bet_context()
    prior = {"open": 0.49, "isolate": 0.28, "reraise": 0.13,
             "bet": 0.36 if event.button else 0.27}[kind]
    aggressive, _ = model.estimate((kind, stage, lane), hand, prior)
    folded = 0.0
    if kind == "reraise":
        price = event.due / (event.pot + event.due)
        prior_fold = min(0.62, 1.5 * price)
        folded, _ = model.estimate(("fold", stage, lane), hand, prior_fold)
        aggressive *= 1 - folded
    if not event.can_raise:
        aggressive = 0.0

    # The soft tails are relative to the range that survived earlier actions.
    pct = midranks(strengths, weights)
    width = (0.38, 0.48, 0.32)[stage]
    bluff_share = (0.08, 0.19, 0.17)[stage]
    if aggressive >= 0.72:
        # A very frequent bet carries little information about this hand.
        return np.ones_like(weights)
    high = _soft_tail(pct, aggressive * (1 - bluff_share), width)
    low = _soft_tail(1 - pct, folded + aggressive * bluff_share, width)
    low_bluff = aggressive * bluff_share / max(0.01, folded + aggressive * bluff_share)
    raise_prob = np.minimum(0.97, high + low * low_bluff)
    fold_prob = np.minimum(0.97 - raise_prob, low * (1 - low_bluff))
    if event.action in _RAISES:
        base = aggressive
        likelihood = raise_prob
    elif event.action == Action.FOLD:
        base = folded
        likelihood = fold_prob
    else:
        base = 1 - aggressive - folded
        likelihood = 1 - raise_prob - fold_prob
    # Flat probability mass allows for unmodeled bluffs and weak draws.
    return 0.52 * likelihood + 0.48 * max(0.04, base)


@dataclass(frozen=True)
class BigBlindRange:
    estimate: Equity
    strengths: dict[Street, np.ndarray]
    river_bins: tuple[np.ndarray, np.ndarray] | None = None

    @classmethod
    def build(cls, obs: Observation, preflop: np.ndarray, worlds: np.ndarray):
        tables = rank_tables()
        hands = worlds[:, :5]
        sample_pairs = combination_index(hands[:, _PAIRS])
        pre = preflop[combination_index(hands)]
        strengths = {Street.PREFLOP: pre[:, 0] + 0.055 * (pre[:, 3] - 0.5)}
        for street in (Street.FLOP, Street.TURN, Street.RIVER):
            if street > obs.street:
                break
            board = np.asarray(obs.board[:BOARD_CARDS_VISIBLE[street]])
            triples = combination_index(board[_TRIPLES[len(board)]])
            strengths[street] = tables.best(sample_pairs, triples)
        if obs.street != Street.RIVER:
            return cls(tables.sample(obs.hole_cards, obs.board, worlds), strengths)
        ranks, exact = tables.river_profile(obs.hole_cards, obs.board)
        bins = (exact.opponent_strength * _RIVER_BINS).astype(int)
        sampled_bins = bins[ranks.searchsorted(strengths[Street.RIVER])]
        # Project earlier history onto exact rank counts. Today's river
        # betting can then update the exact distribution directly.
        strengths[Street.RIVER] = ranks
        return cls(exact, strengths, (bins, sampled_bins))

    def condition(self, obs: Observation, model: OpponentModel) -> Equity:
        events = model.trace.events if model.trace else ()
        weights = np.ones(len(self.strengths[Street.PREFLOP]))
        if self.river_bins is not None:
            return self._condition_river(obs, model, events, weights)
        for event in events:
            if event.street not in self.strengths:
                continue
            weights *= _likelihood(event, self.strengths[event.street], weights, model, obs.hand_index)
            weights /= weights.mean()
        # Bound concentration even after long, highly correlated action sequences.
        posterior = 0.83 * weights + 0.17
        return replace(self.estimate, frequency=posterior)

    def _condition_river(self, obs, model, events, weights) -> Equity:
        for event in events:
            if event.street < Street.RIVER:
                weights *= _likelihood(event, self.strengths[event.street], weights, model, obs.hand_index)
                weights /= weights.mean()
        bins, sampled_bins = self.river_bins
        count = np.bincount(sampled_bins, minlength=_RIVER_BINS)
        weighted = np.bincount(sampled_bins, weights=weights, minlength=_RIVER_BINS)
        factors = (weighted + 8) / (count + 8)
        prior = self.estimate.frequency.astype(float)
        posterior = prior * factors[bins]
        for event in events:
            if event.street == Street.RIVER:
                posterior *= _likelihood(event, self.strengths[Street.RIVER], posterior, model, obs.hand_index)
        posterior *= prior.sum() / posterior.sum()
        return replace(self.estimate, frequency=0.83 * posterior + 0.17 * prior)
