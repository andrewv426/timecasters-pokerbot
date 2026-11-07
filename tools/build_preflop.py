"""Generate the fixed preflop asset with reproducible, blocker-aware simulations.

From the repository root: python -m tools.build_preflop
"""

import argparse
from itertools import permutations
from pathlib import Path

import numpy as np

from bot_equity import (
    card_sets, combination_index, draw_worlds, rank_tables, showdown_share,
    unseen_cards,
)
from evaluator import NUM_CARDS


def playability(cards: np.ndarray) -> np.ndarray:
    ranks = cards >> 2
    suits = cards & 3
    a, b = card_sets(5, 2).T
    lower, upper = ranks[:, a], ranks[:, b]
    paired = lower == upper
    pair_value = (paired * (0.12 + 0.23 * (upper / 8) ** 1.7)).max(axis=1)
    pair_count = paired.sum(axis=1)
    # Two useful pairs gain a little. Third and fourth copies block our own outs.
    pair_depth = 0.045 * (pair_count == 2) - 0.035 * np.maximum(0, pair_count - 2)
    gap = upper - lower
    wheel_gap = np.where((upper == 8) & (lower <= 3), lower + 1, 9)
    connected = np.maximum(0, 4 - np.minimum(gap, wheel_gap)) * (gap > 0)
    connectivity = 0.105 * connected.mean(axis=1)
    suit_value = np.zeros(len(cards))
    for suit in range(4):
        matches = suits == suit
        count = matches.sum(axis=1)
        high_rank = np.where(matches, ranks, 0).max(axis=1)
        suit_value += (count >= 2) * (0.065 + 0.13 * (high_rank / 8) ** 2)
        suit_value -= 0.035 * np.maximum(0, count - 2)
    quality = pair_value + pair_depth + connectivity + suit_value + 0.14 * ranks.mean(axis=1) / 8
    _, groups, counts = np.unique(quality.round(7), return_inverse=True, return_counts=True)
    return ((counts.cumsum() - 0.5 * counts) / len(cards))[groups]


def build(trials: int) -> np.ndarray:
    raw = card_sets(NUM_CARDS, 5)
    raw_ids = combination_index(raw)
    cards_by_id = np.empty_like(raw)
    cards_by_id[raw_ids] = raw
    percentile = np.empty(len(raw))
    percentile[raw_ids] = playability(raw)

    canonical = raw_ids.copy()
    rank_bits = raw & ~3
    for perm in permutations(range(4)):
        relabeled = rank_bits | np.array(perm, dtype=raw.dtype)[raw & 3]
        np.minimum(canonical, combination_index(relabeled), out=canonical)
    representatives, assignment = np.unique(canonical, return_inverse=True)
    assert len(representatives) == 19_998

    tables = rank_tables()
    hole_pairs = card_sets(5, 2)
    board_triples = card_sets(5, 3)
    rng = np.random.default_rng(20261107)
    representative_equity = np.empty((len(representatives), 3), dtype=np.float32)
    for i, card_id in enumerate(representatives):
        hole = cards_by_id[card_id]
        worlds = draw_worlds(unseen_cards(tuple(hole), ()), 5, trials, rng)
        villain_holes, boards = worlds[:, :5], worlds[:, 5:]
        hero_pairs = combination_index(hole[hole_pairs])
        villain_pairs = combination_index(villain_holes[:, hole_pairs])
        triples = combination_index(boards[:, board_triples])
        outcomes = showdown_share(tables.best(hero_pairs, triples),
                                  tables.best(villain_pairs, triples))
        strength = percentile[combination_index(villain_holes)]
        # Likelihoods for a broad opener and a stronger continuing range.
        opened = 0.30 + 0.70 * strength
        continued = 0.13 + 0.87 * strength ** 2.2
        representative_equity[i] = (
            outcomes.mean(), outcomes @ opened / opened.sum(),
            outcomes @ continued / continued.sum(),
        )
        if (i + 1) % 2000 == 0:
            print(f"{i + 1:,}/{len(representatives):,} suit classes", flush=True)

    result = np.empty((len(raw), 4), dtype=np.float32)
    result[raw_ids, :3] = representative_equity[assignment]
    result[:, 3] = percentile
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=Path("data/preflop.npz"))
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    equity = build(args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, equity=equity, trials=args.trials)
    print(f"Wrote {args.output} with {args.trials} trials per suit class")
