"""Regression cases for opponent evidence at the official hook boundary."""

from dataclasses import replace

import pytest

from game import Action, AgentConfig, Game, Street, make_deal
from opponent_model import Event, FlopLine, OpponentModel


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


@pytest.mark.parametrize("button", [0, 1])
@pytest.mark.parametrize("action, lane", [(Action.RAISE_HALF, 0), (Action.RAISE_POT, 1)])
def test_first_flop_bet_learns_terminal_fold_by_position_and_price(button, action, lane):
    game = Game(make_deal("flop-fold", 0), button_seat=button)
    model = OpponentModel(AgentConfig(), learn_flop_pressure=True)
    while game.street < Street.FLOP:
        play(game, model, Action.CHECK_CALL)
    if game.to_act != 0:
        play(game, model, Action.CHECK_CALL)
    play(game, model, action)
    play(game, model, Action.FOLD)
    model.finish(game.hand_result(0))
    assert model.rates["flop_fold", int(button == 0), lane].at(0) == (1, 1)


@pytest.mark.parametrize("line", list(FlopLine))
def test_big_blind_turn_read_tracks_last_flop_aggressor(line):
    game = Game(make_deal("split-turn", 0), button_seat=1)
    model = OpponentModel(AgentConfig(), split_bb_reads=True)
    while game.street < Street.FLOP:
        play(game, model, Action.CHECK_CALL)
    if line == FlopLine.HERO_AGGRESSOR:
        play(game, model, Action.RAISE_HALF)
        play(game, model, Action.CHECK_CALL)
    else:
        play(game, model, Action.CHECK_CALL)
        if line == FlopLine.OPPONENT_AGGRESSOR:
            play(game, model, Action.RAISE_HALF)
        play(game, model, Action.CHECK_CALL)
    assert game.street == Street.TURN
    play(game, model, Action.CHECK_CALL)
    play(game, model, Action.RAISE_HALF)
    while not game.is_terminal:
        play(game, model, Action.CHECK_CALL)
    model.finish(game.hand_result(0))
    assert model.bb_rates["bet", Street.TURN, 1, line].at(0) == (1, 1)
    assert model.bb_rates["bet", Street.TURN, 1, None].at(0) == (1, 1)
    assert ("bet", Street.FLOP, 1, line) not in model.bb_rates


def test_big_blind_read_backs_off_and_ignores_button_hands():
    model = OpponentModel(AgentConfig(), split_bb_reads=True)
    key = ("bet", 1, 1)
    flop = Event(Street.FLOP, Action.CHECK_CALL, 0, 4, 0, True, True)
    turn = replace(flop, street=Street.TURN, flop_line=FlopLine.HERO_AGGRESSOR)
    model._record_rate(turn, key, True, 0)
    assert model.bb_estimate(key, Street.TURN, 0, 0.36, turn.flop_line) == (0.36, 0.0)
    for hand in range(1, 50):
        model._record_rate(flop, key, False, hand)
        model._record_rate(turn, key, True, hand)
        model._record_rate(replace(turn, button=False), key, False, hand)
    broad, _ = model.estimate(key, 50, 0.36)
    assert model.bb_estimate(key, Street.FLOP, 50, 0.36, flop.flop_line)[0] < broad
    assert model.bb_estimate(key, Street.TURN, 50, 0.36, turn.flop_line)[0] > broad
    hits, opportunities = model.bb_rates["bet", Street.TURN, 1, None].at(50)
    assert hits == opportunities


@pytest.mark.parametrize("action, size", [(Action.RAISE_HALF, 0), (Action.RAISE_POT, 1)])
@pytest.mark.parametrize("revealed", [False, True])
def test_turn_size_is_learned_from_reveal_once(action, size, revealed):
    game = Game(make_deal("turn-sizing", 0), button_seat=1)
    model = OpponentModel(AgentConfig(), learn_turn_sizing=True)
    while game.street < Street.TURN:
        play(game, model, Action.CHECK_CALL)
    play(game, model, Action.CHECK_CALL)
    play(game, model, action)
    while not game.is_terminal:
        play(game, model, Action.CHECK_CALL)
    result = game.hand_result(0)
    if not revealed:
        result = replace(result, opp_hole=None)
    model.finish(result)
    model.finish(result)
    assert model.turn_bet_strength[size].opportunities == int(revealed)
    assert model.turn_bet_strength[1 - size].opportunities == 0


def test_turn_size_signal_needs_both_sizes_and_substantial_reversal():
    model = OpponentModel(AgentConfig(), learn_turn_sizing=True)
    half, pot = model.turn_bet_strength
    for hand in range(20):
        half.add(0.82, hand)
    assert model.turn_size_signal(20) == 0
    for hand in range(20):
        pot.add(0.40, hand)
    assert model.turn_size_signal(20) > 0.60
    assert model.turn_size_signal(3_000) == 0


@pytest.mark.parametrize("facing_raise", [False, True])
def test_big_blind_pot_frequency_counts_bets_and_reraises_separately(facing_raise):
    game = Game(make_deal("pot-frequency", 0), button_seat=1)
    model = OpponentModel(AgentConfig())
    while game.street < Street.FLOP:
        play(game, model, Action.CHECK_CALL)
    play(game, model, Action.RAISE_HALF if facing_raise else Action.CHECK_CALL)
    play(game, model, Action.RAISE_POT)
    obs = game.observation(0)
    model.observe(obs)
    event = model.last_event(Street.FLOP)
    assert event.full_size
    prior = model.bb_pot_rate(event, obs.hand_index)
    play(game, model, Action.FOLD)
    result = game.hand_result(0)
    model.finish(result)
    model.finish(result)
    assert model.bb_pot_rate(event, obs.hand_index) == pytest.approx((1 + 24 * prior) / 25)
    other = replace(event, raises=0, due=0) if facing_raise else replace(event, raises=1, due=2)
    assert model.bb_pot_rate(other, obs.hand_index) == (0.28 if facing_raise else 0.14)


def test_stack_capped_pot_action_does_not_count_as_full_sized():
    game = Game(make_deal("short-all-in", 0), button_seat=1)
    model = OpponentModel(AgentConfig())
    for action in (Action.RAISE_POT,) * 3 + (Action.CHECK_CALL,) * 2:
        play(game, model, action)
    assert game.street == Street.FLOP
    play(game, model, Action.RAISE_POT)
    obs = game.observation(0)
    model.observe(obs)
    assert obs.opp_stack == 0
    assert not model.last_event(Street.FLOP).full_size
