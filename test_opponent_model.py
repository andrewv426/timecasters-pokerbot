"""Regression cases for opponent evidence at the official hook boundary."""

from dataclasses import replace

import pytest

from game import Action, AgentConfig, Game, Street, make_deal
from opponent_model import OpponentModel


def play(game, model, action):
    if game.to_act == 0:
        obs = game.observation(0)
        model.observe(obs)
        model.remember(obs, action)
    game.apply(action)


def test_history_cursor_and_terminal_fold_are_counted_once():
    game = Game(make_deal("opponent-model", 0), button_seat=0)
    model = OpponentModel(AgentConfig())
    play(game, model, Action.RAISE_HALF)
    play(game, model, Action.RAISE_POT)
    obs = game.observation(0)
    model.observe(obs)
    model.observe(obs)
    assert len(model.trace.events) == 1
    play(game, model, Action.RAISE_POT)
    play(game, model, Action.FOLD)
    result = game.hand_result(0)
    model.finish(replace(result, hand_index=777))
    assert not model.rates
    model.finish(result)
    model.finish(result)
    assert model.trace is None
    assert model.rates["fold", 0, -1].at(0) == (1, 2)
    assert model.rates["reraise", 0, -1].at(0) == (1, 1)


def test_terminal_all_in_call_counts_as_a_defense_without_raise_opportunity():
    game = Game(make_deal("opponent-model", 1), button_seat=1)
    model = OpponentModel(AgentConfig())
    for action in (Action.RAISE_POT,) * 4 + (Action.CHECK_CALL,):
        play(game, model, action)
    result = game.hand_result(0)
    terminal = model.trace.infer_terminal(result, 4)
    assert terminal.action == Action.CHECK_CALL
    assert terminal.street == Street.PREFLOP
    assert not terminal.can_raise
    model.finish(result)
    assert model.rates["fold", 0, -1].at(0) == (0, 2)
    assert model.rates["reraise", 0, -1].at(0) == (1, 1)


def test_showdown_uses_a_real_opponent_bet_and_ignores_repeated_result():
    game = Game(make_deal("opponent-model", 2), button_seat=1)
    model = OpponentModel(AgentConfig())
    while game.street != Street.RIVER:
        play(game, model, Action.CHECK_CALL)
    play(game, model, Action.CHECK_CALL)
    play(game, model, Action.RAISE_HALF)
    play(game, model, Action.CHECK_CALL)
    result = game.hand_result(0)
    model.finish(result)
    model.finish(result)
    assert model.rates["bet", 2, 1].opportunities == 1
    assert model.rates["weak_bet", 2, 1].opportunities == 1


def test_rates_use_a_minimum_sample_and_discount_old_evidence():
    model = OpponentModel(AgentConfig())
    key = "fold", 2, 1
    for hand in range(10):
        model._add(*key, True, hand)
    assert model.estimate(key, 10, 0.55) == (0.55, 0.0)
    for hand in range(10, 110):
        model._add(*key, True, hand)
    old_rate, old_trust = model.estimate(key, 110, 0.55)
    assert old_rate > 0.9 and old_trust > 0.5
    for hand in range(450, 555):
        model._add(*key, False, hand)
    new_rate, _ = model.estimate(key, 555, 0.55)
    assert new_rate < 0.30
    assert model.estimate(("bet", 2, 1), 555, 0.36) == pytest.approx((0.36, 0.0))
