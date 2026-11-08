"""Regression coverage for the expensive call lines in the replay audit."""

from dataclasses import replace

import numpy as np
import pytest

from game import Action, Game, Street, make_deal
from late_defense import defensive_weights, pressure, river_pressure_cost


def turn_spot() -> Game:
    game = Game(make_deal("late-defense", 0), button_seat=1)
    game.apply(Action.RAISE_POT)
    game.apply(Action.CHECK_CALL)
    while game.street < Street.TURN:
        game.apply(Action.CHECK_CALL)
    game.apply(Action.CHECK_CALL)
    game.apply(Action.RAISE_POT)
    return game


def test_large_turn_bet_after_earlier_aggression_reserves_river_cost():
    obs = turn_spot().observation(0)
    severity = pressure(obs, None)
    assert severity > 0
    assert river_pressure_cost(obs, severity, 0.1) > river_pressure_cost(obs, severity, 0.8)
    assert pressure(replace(obs, opp_stack=0), None) == 0


def test_river_pressure_targets_barrels_and_leaves_cheap_calls_alone():
    game = turn_spot()
    game.apply(Action.CHECK_CALL)
    game.apply(Action.CHECK_CALL)
    game.apply(Action.RAISE_HALF)
    obs = game.observation(0)
    assert pressure(obs, None) > 0
    assert pressure(replace(obs, pot=10 * obs.to_call), None) == 0
    assert river_pressure_cost(obs, pressure(obs, None), 0.0) == 0


def test_defense_tilt_preserves_range_mass_and_cuts_weak_equity():
    weights = np.array([1.0, 2.0, 1.0])
    strength = np.array([0.12, 0.43, 0.91])
    tilted = defensive_weights(weights, strength, 0.6)
    assert tilted.sum() == pytest.approx(weights.sum())
    assert tilted[0] < weights[0]
    assert tilted[-1] > weights[-1]
    assert defensive_weights(weights, strength, 0) is weights
