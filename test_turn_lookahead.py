from dataclasses import replace

import numpy as np
import pytest

import bot_policy
from bot_equity import Equity, unseen_cards
from evaluator import omaha_score
from game import Action, AgentConfig, Game, Street, make_deal
from opponent_model import OpponentModel
from turn_lookahead import river_check_value, turn_call_value


def turn_spot() -> Game:
    game = Game(make_deal("turn-lookahead", 0), button_seat=1)
    game.apply(Action.RAISE_POT)
    game.apply(Action.CHECK_CALL)
    while game.street < Street.TURN:
        game.apply(Action.CHECK_CALL)
    game.apply(Action.CHECK_CALL)
    game.apply(Action.RAISE_POT)
    return game


def test_river_choice_uses_bet_range_and_stack_cap():
    shares = np.array([0.0, 1.0])
    bets = np.ones(2)
    # Seeing a bet does not tell us which of these two holdings the bettor has.
    assert river_check_value(20, 20, np.array([0.9, 0.1]), shares, bets) == 0.0
    assert river_check_value(20, 20, np.array([0.7, 0.3]), shares, bets) == pytest.approx(1.0)
    assert river_check_value(20, 4, np.array([0.7, 0.3]), shares, bets) == pytest.approx(4.4)


def test_check_through_rollout_matches_legal_river_showdowns(monkeypatch):
    game = turn_spot()
    obs = game.observation(0)
    model = OpponentModel(AgentConfig())
    monkeypatch.setattr(model, "estimate", lambda *args: (0.0, 0.0))
    opponent = game.deal.hole_cards[1]
    outcomes = []
    for river in unseen_cards(obs.hole_cards, obs.board):
        if river in opponent:
            continue
        board = obs.board + (int(river),)
        ours = omaha_score(obs.hole_cards, board)
        theirs = omaha_score(opponent, board)
        outcomes.append(0.5 if ours == theirs else float(ours > theirs))
    expected = np.mean(outcomes) * (obs.pot + obs.to_call) - obs.to_call
    assert turn_call_value(obs, model, np.array([opponent]), np.ones(1)) == pytest.approx(expected)


def test_rollout_can_change_a_close_turn_call_and_skips_out_of_scope(monkeypatch):
    obs = turn_spot().observation(0)
    config = AgentConfig()
    model = OpponentModel(config)
    outcomes = np.r_[np.ones(41), np.zeros(59)]
    estimate = Equity(outcomes, np.linspace(0, 1, 100), np.ones(100), 0.41, 0.35, False, 100)
    hands = np.empty((100, 5), dtype=int)
    calls = []

    def rollout(*args):
        calls.append(args[0])
        return -obs.pot

    monkeypatch.setattr(bot_policy, "turn_call_value", rollout)
    assert bot_policy.postflop_action(obs, config, estimate, 0.5, model, improve_bb=True)[0] == Action.CHECK_CALL
    assert bot_policy.postflop_action(obs, config, estimate, 0.5, model, improve_bb=True,
                                    turn_hands=hands)[0] == Action.FOLD
    for out_of_scope in (replace(obs, is_button=True), replace(obs, opp_stack=0)):
        bot_policy.postflop_action(out_of_scope, config, estimate, 0.5, model, improve_bb=True,
                                   turn_hands=hands)
    assert calls == [obs]
