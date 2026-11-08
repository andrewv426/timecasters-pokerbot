"""Batch equity calculations using the referee's exact five-card ordering."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import chain, combinations
from math import comb

import numpy as np

from evaluator import NUM_CARDS, get_table


def card_sets(n: int, k: int) -> np.ndarray:
    cards = chain.from_iterable(combinations(range(n), k))
    return np.fromiter(cards, dtype=np.int8, count=k * comb(n, k)).reshape(-1, k)


_BINOMIAL = np.array([[comb(c, k) for c in range(NUM_CARDS)] for k in range(6)])
_CARD_BITS = np.left_shift(np.uint64(1), np.arange(NUM_CARDS, dtype=np.uint64))
_DECK = np.arange(NUM_CARDS, dtype=np.int8)
_HOLE_PAIRS = card_sets(5, 2)
_BOARD_TRIPLES = {n: card_sets(n, 3) for n in (3, 4, 5)}
_PAIR_SPLITS = [
    (pair, tuple(i for i in range(5) if i not in pair))
    for pair in combinations(range(5), 2)
]


def combination_index(cards: np.ndarray) -> np.ndarray:
    """Dense colex index, independent of card order."""
    sorted_cards = np.sort(cards, axis=-1)
    return _BINOMIAL[np.arange(1, cards.shape[-1] + 1), sorted_cards].sum(
        axis=-1, dtype=np.int32
    )


def unseen_cards(hole: tuple[int, ...], board: tuple[int, ...]) -> np.ndarray:
    available = np.ones(NUM_CARDS, dtype=bool)
    available[list(hole + board)] = False
    return _DECK[available]


def draw_worlds(
    remaining: np.ndarray, missing: int, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Each row is a uniform ordered draw of five holes and a legal runout."""
    keys = rng.random((n, remaining.size))
    return remaining[keys.argsort(axis=1)[:, :5 + missing]]


def showdown_share(hero: np.ndarray, villain: np.ndarray) -> np.ndarray:
    return (hero > villain).astype(float) + 0.5 * (hero == villain)


@dataclass(frozen=True)
class Equity:
    """Samples or river rank groups, with strength percentiles in [0, 1]."""

    shares: np.ndarray
    opponent_strength: np.ndarray
    frequency: np.ndarray
    made_equity: float
    redraw: float
    nuts: bool
    trials: int


@dataclass(frozen=True)
class RankTables:
    scores: np.ndarray
    category: np.ndarray
    river_pairs: np.ndarray

    def best(self, pairs: np.ndarray, triples: np.ndarray) -> np.ndarray:
        return self.scores[triples[..., :, None], pairs[..., None, :]].max(
            axis=(-2, -1)
        )

    def sample(self, hole: tuple[int, ...], board: tuple[int, ...],
               worlds: np.ndarray) -> Equity:
        hero_pairs = combination_index(np.array(hole)[_HOLE_PAIRS])
        villain_pairs = combination_index(worlds[:, :5][:, _HOLE_PAIRS])
        complete = np.concatenate((
            np.broadcast_to(board, (len(worlds), len(board))), worlds[:, 5:]
        ), axis=1)
        triples = combination_index(complete[:, _BOARD_TRIPLES[5]])
        hero = self.best(hero_pairs, triples)
        villain = self.best(villain_pairs, triples)

        visible = combination_index(np.array(board)[_BOARD_TRIPLES[len(board)]])
        current = self.best(hero_pairs, visible)
        opponent = self.best(villain_pairs, visible)
        ordered = np.sort(opponent)
        percentiles = (
            ordered.searchsorted(opponent, "left") + ordered.searchsorted(opponent, "right")
        ) / (2 * len(worlds))
        made_equity = float(showdown_share(current, opponent).mean())
        redraw = float((self.category[hero] > self.category[current]).mean())
        return Equity(
            showdown_share(hero, villain), percentiles, np.ones(len(worlds)),
            made_equity, redraw, False, len(worlds),
        )

    def river_profile(self, hole: tuple[int, ...], board: tuple[int, ...]) -> tuple[np.ndarray, Equity]:
        triples = combination_index(np.array(board)[_BOARD_TRIPLES[5]])
        hero_pairs = combination_index(np.array(hole)[_HOLE_PAIRS])
        hero = self.best(hero_pairs, triples)
        remaining = unseen_cards(hole, board)
        pair_ids = combination_index(remaining[_RIVER_AVAILABLE_PAIRS])
        pair_scores = self.scores[triples[:, None], pair_ids].max(axis=0)
        villain = pair_scores[self.river_pairs].max(axis=0)
        histogram = np.bincount(villain)
        ranks = histogram.nonzero()[0]
        frequency = histogram[ranks]
        percentiles = (frequency.cumsum() - 0.5 * frequency) / len(villain)
        shares = showdown_share(hero, ranks)
        equity = float(shares @ frequency / len(villain))
        return ranks, Equity(shares, percentiles, frequency, equity, 0.0,
                             bool(hero >= ranks[-1]), len(villain))

    def river(self, hole: tuple[int, ...], board: tuple[int, ...]) -> Equity:
        return self.river_profile(hole, board)[1]


_RIVER_AVAILABLE_PAIRS = card_sets(26, 2)


@lru_cache(maxsize=1)
def rank_tables() -> RankTables:
    """Pack the official ranks once at startup. Zero denotes overlapping cards."""
    five = card_sets(NUM_CARDS, 5)
    masks = np.bitwise_or.reduce(_CARD_BITS[five], axis=1)
    official = get_table()
    ranks = np.fromiter((official[int(mask)] for mask in masks), np.int32, len(five))
    unique, ordinal = np.unique(ranks, return_inverse=True)
    table = np.zeros((comb(NUM_CARDS, 3), comb(NUM_CARDS, 2)), dtype=np.uint16)
    for pair, triple in _PAIR_SPLITS:
        table[combination_index(five[:, triple]), combination_index(five[:, pair])] = ordinal + 1
    category = np.concatenate(([0], unique >> 20))
    river_hands = card_sets(26, 5)
    # The river's temporary pair ranks use lexicographic pair order.
    river_index = np.empty(comb(26, 2), dtype=np.uint16)
    river_index[combination_index(_RIVER_AVAILABLE_PAIRS)] = np.arange(len(river_index))
    river_pairs = np.ascontiguousarray(
        river_index[combination_index(river_hands[:, _HOLE_PAIRS])].T
    )
    for array in (table, category, river_pairs):
        array.setflags(write=False)
    return RankTables(table, category, river_pairs)
