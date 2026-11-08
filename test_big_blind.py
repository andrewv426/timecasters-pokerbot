"""Regression checks for the big-blind range and realization math."""

import numpy as np
import pytest

from big_blind import BigBlindRange, midranks, realized_equity
from bot_equity import draw_worlds, rank_tables, unseen_cards
from bot_policy import load_preflop
from game import Action, AgentConfig, Game, Street, make_deal
from opponent_model import OpponentModel


def test_weighted_midranks_keep_tied_hand_values_together():
    result = midranks(np.array([40, 10, 40, 60]), np.array([2, 2, 4, 2]))
    np.testing.assert_allclose(result, [0.5, 0.1, 0.5, 0.9])


def test_realization_depends_on_depth_and_preserves_terminal_equity():
    assert realized_equity(0.40, 8, Street.PREFLOP, 0.15) < realized_equity(
        0.40, 1, Street.PREFLOP, 0.15
    )
    assert realized_equity(0.40, 8, Street.PREFLOP, 0.75) > realized_equity(
        0.40, 8, Street.PREFLOP, 0.15
    )
    assert realized_equity(0.40, 0, Street.FLOP, 0.15) == 0.40
    assert realized_equity(0.40, 8, Street.RIVER, 0.15) == 0.40


def test_opponent_open_still_affects_range_on_the_flop():
    game = Game(make_deal("big-blind", 0), button_seat=1)
    game.apply(Action.RAISE_POT)
    game.apply(Action.CHECK_CALL)
    obs = game.observation(0)
    assert obs.street == Street.FLOP and not obs.is_button
    model = OpponentModel(AgentConfig())
    model.observe(obs)
    worlds = draw_worlds(unseen_cards(obs.hole_cards, obs.board), 2, 256,
                        np.random.default_rng(73))
    sample = BigBlindRange.build(obs, load_preflop(), worlds)
    weights = sample.condition(obs, model).frequency
    preflop = sample.strengths[Street.PREFLOP]
    assert weights.mean() == pytest.approx(1.0)
    assert weights.min() >= 0.17
    assert np.average(preflop, weights=weights) > preflop.mean()


def test_river_conditioning_preserves_exact_outcomes_and_total_mass():
    game = Game(make_deal("big-blind-river", 0), button_seat=1)
    while game.street != Street.RIVER:
        game.apply(Action.CHECK_CALL)
    game.apply(Action.CHECK_CALL)
    game.apply(Action.RAISE_HALF)
    obs = game.observation(0)
    model = OpponentModel(AgentConfig())
    model.observe(obs)
    worlds = draw_worlds(unseen_cards(obs.hole_cards, obs.board), 0, 256,
                        np.random.default_rng(904))
    exact = rank_tables().river(obs.hole_cards, obs.board)
    posterior = BigBlindRange.build(obs, load_preflop(), worlds).condition(obs, model)
    np.testing.assert_array_equal(posterior.shares, exact.shares)
    assert posterior.frequency.sum() == pytest.approx(exact.trials)
    assert np.all(posterior.frequency >= 0.17 * exact.frequency)
    assert not np.allclose(posterior.frequency, exact.frequency)
