"""Parity tests for the bot's separate equity kernel and bundled asset."""

import numpy as np
import pytest

from bot_equity import card_sets, combination_index, draw_worlds, rank_tables, unseen_cards
from bot_policy import load_preflop
from evaluator import cards_from_str, get_table, omaha_score


def test_packed_rank_parity_for_all_five_card_hands():
    cards = card_sets(36, 5)
    masks = np.bitwise_or.reduce(np.uint64(1) << cards.astype(np.uint64), axis=1)
    official = get_table()
    raw_scores = np.fromiter((official[int(mask)] for mask in masks), np.int32, len(cards))
    _, expected = np.unique(raw_scores, return_inverse=True)
    expected += 1
    packed = rank_tables().scores
    assert np.array_equal(
        packed[combination_index(cards[:, :3]), combination_index(cards[:, 3:])],
        expected,
    )
    assert np.array_equal(
        packed[combination_index(cards[:, 2:]), combination_index(cards[:, :2])],
        expected,
    )


@pytest.mark.parametrize(("hero", "board", "wins", "ties", "losses"), [
    ("Th 7s 7d 8c 9c", "Ah Kh Qh Jh 6c", 13007, 564, 52209),
    ("Ah Kh 8d 9s 6d", "Qh Jh Th 6c 7d", 65780, 0, 0),
    ("Ac 6s Kd 7s 8d", "6c 7d 8h 9s Tc", 3408, 14502, 47870),
    ("6c 7h 8h 9h Tc", "Ad Ah As Kd Ks", 0, 243, 65537),
])
def test_exact_river(hero, board, wins, ties, losses):
    hole, community = (tuple(cards_from_str(cards)) for cards in (hero, board))
    estimate = rank_tables().river(hole, community)
    assert estimate.trials == 65_780
    assert tuple(estimate.frequency[estimate.shares == share].sum()
                 for share in (1, 0.5, 0)) == (wins, ties, losses)


@pytest.mark.parametrize("board_size", [3, 4])
def test_sampled_equity_uses_legal_deals_and_exact_scores(board_size):
    hole = (0, 4, 9, 22, 35)
    board = (2, 10, 16, 20)[:board_size]
    rng = np.random.default_rng(373)
    worlds = draw_worlds(unseen_cards(hole, board), 5 - len(board), 120, rng)
    estimate = rank_tables().sample(hole, board, worlds)
    expected = []
    for world in worlds.tolist():
        assert len(set(hole + board + tuple(world))) == 15
        villain, runout = world[:5], world[5:]
        final_board = board + tuple(runout)
        a = omaha_score(hole, final_board)
        b = omaha_score(villain, final_board)
        expected.append(1.0 if a > b else 0.5 if a == b else 0.0)
    assert estimate.shares.tolist() == expected


def test_preflop_asset_is_suit_invariant():
    asset = load_preflop()
    hole = np.array([2, 7, 8, 22, 35])
    remapped = (hole & ~3) | np.array([3, 0, 1, 2])[hole & 3]
    assert np.array_equal(asset[combination_index(hole)], asset[combination_index(remapped)])
