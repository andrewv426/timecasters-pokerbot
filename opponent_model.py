"""Half-local opponent estimates from public actions and revealed showdowns."""

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from bot_equity import card_sets, combination_index, draw_worlds, rank_tables, unseen_cards
from game import Action, AgentConfig, BOARD_CARDS_VISIBLE, HandResult, Observation, Street

_RAISES = (Action.RAISE_HALF, Action.RAISE_POT)
_DECAY = 2 ** (-1 / 220)
_PAIRS = card_sets(5, 2)
_TRIPLES = {n: card_sets(n, 3) for n in (3, 4, 5)}


def phase(street: Street) -> int:
    if street == Street.PREFLOP:
        return 0
    return 2 if street == Street.RIVER else 1


def size_bucket(price: float) -> int:
    return int(price >= 0.29)


class FlopLine(IntEnum):
    CHECKED_THROUGH = 0
    HERO_AGGRESSOR = 1
    OPPONENT_AGGRESSOR = 2


@dataclass(frozen=True)
class Event:
    street: Street
    action: Action
    due: int
    pot: int
    raises: int
    can_raise: bool
    button: bool
    turn_bet_size: int | None = None
    flop_line: FlopLine = FlopLine.CHECKED_THROUGH
    full_size: bool = False

    @property
    def lane(self) -> int:
        return size_bucket(self.due / (self.pot + self.due))

    def bet_context(self) -> tuple[str, int, int]:
        if self.raises and self.due:
            return "reraise", phase(self.street), self.lane
        if self.street == Street.PREFLOP:
            return ("open" if self.button else "isolate"), 0, 0
        return "bet", phase(self.street), int(self.button)


@dataclass
class Frequency:
    hits: float = 0.0
    opportunities: float = 0.0
    hand: int = 0
    decay: float = _DECAY

    def at(self, hand: int) -> tuple[float, float]:
        decay = self.decay ** max(0, hand - self.hand)
        return self.hits * decay, self.opportunities * decay

    def add(self, hit: float, hand: int) -> None:
        now = max(hand, self.hand)
        hits, opportunities = self.at(now)
        weight = self.decay ** (now - hand)
        self.hits = hits + hit * weight
        self.opportunities = opportunities + weight
        self.hand = now


@dataclass
class HandTrace:
    key: tuple[int, tuple[int, ...], bool]
    bets: list[int]
    stacks: list[int]
    pot: int
    street: Street = Street.PREFLOP
    raises: int = 0
    cursor: int = 0
    events: list[Event] = field(default_factory=list)
    pending: tuple[Observation, Action] | None = None
    flop_line: FlopLine = FlopLine.CHECKED_THROUGH

    def consume(self, obs: Observation, cap: int) -> None:
        for record in obs.action_history[self.cursor:]:
            if record.street != self.street:
                self.street = record.street
                self.bets = [0, 0]
                self.raises = 0
            player = 0 if record.is_me else 1
            other = 1 - player
            due = max(0, self.bets[other] - self.bets[player])
            if not record.is_me:
                turn_bet_size = None
                if (self.street == Street.TURN and not self.raises and record.action in _RAISES
                        and record.amount < min(self.stacks)):
                    turn_bet_size = int(record.amount / self.pot >= 0.70)
                self.events.append(Event(
                    self.street, record.action, due, self.pot, self.raises,
                    self.raises < cap and self.stacks[1] > due and self.stacks[0] > 0,
                    not obs.is_button, turn_bet_size, self.flop_line,
                    # Subtract the call before measuring the raise. The pot
                    # action can be a much smaller bet when stacks are short.
                    full_size=(record.action == Action.RAISE_POT
                               and record.amount - due >= 0.70 * (self.pot + due)),
                ))
            if self.street == Street.FLOP and record.action in _RAISES:
                self.flop_line = (FlopLine.HERO_AGGRESSOR if record.is_me
                                  else FlopLine.OPPONENT_AGGRESSOR)
            self.pot += record.amount
            self.bets[player] = record.street_total
            self.stacks[player] -= record.amount
            self.raises += record.action in _RAISES
        self.cursor = len(obs.action_history)

    def infer_terminal(self, result: HandResult, cap: int) -> Event | None:
        if self.pending is None:
            return None
        obs, action = self.pending
        if action in _RAISES:
            target = obs.raise_to[action]
            paid = target - obs.my_bet
            due = target - obs.opp_bet
            after_raise = obs.pot + paid
            is_all_in = obs.my_stack == paid or obs.opp_stack <= due
            can_raise = not is_all_in and obs.raises_this_street + 1 < cap
            if (not result.went_to_showdown and result.winner == "me"
                    and result.board == obs.board and result.pot == after_raise):
                reply = Action.FOLD
            elif (result.went_to_showdown and (is_all_in or obs.street == Street.RIVER)
                  and result.pot == after_raise + min(obs.opp_stack, due)):
                reply = Action.CHECK_CALL
            else:
                return None
            return Event(obs.street, reply, due, after_raise, obs.raises_this_street + 1,
                         can_raise, not obs.is_button, flop_line=self.flop_line)
        if (action == Action.CHECK_CALL and obs.street == Street.RIVER and not obs.is_button
                and not obs.to_call and result.went_to_showdown and result.pot == obs.pot):
            return Event(obs.street, Action.CHECK_CALL, 0, obs.pot, 0,
                         bool(obs.my_stack and obs.opp_stack), True)
        return None


class OpponentModel:
    def __init__(self, config: AgentConfig, learn_turn_sizing: bool = False,
                 learn_flop_pressure: bool = False, split_bb_reads: bool = False):
        self.config = config
        self.trace: HandTrace | None = None
        self.rates: dict[tuple[str, int, int], Frequency] = {}
        self.finished: set[int] = set()
        self.showdown_rng = np.random.default_rng(852781)
        self.turn_bet_strength = (
            Frequency(decay=2 ** (-1 / 650)), Frequency(decay=2 ** (-1 / 650))
        ) if learn_turn_sizing else ()
        self.turn_showdown_rng = np.random.default_rng(421073)
        self.learn_flop_pressure = learn_flop_pressure
        self.split_bb_reads = split_bb_reads
        self.bb_pot_rates: dict[tuple[Street, bool], Frequency] = {}
        self.bb_rates: dict[tuple[str, Street, int, FlopLine | None], Frequency] = {}

    def observe(self, obs: Observation) -> None:
        key = (obs.hand_index, obs.hole_cards, obs.is_button)
        if self.trace is None or self.trace.key != key:
            small, big = self.config.small_blind, self.config.big_blind
            bets = [small, big] if obs.is_button else [big, small]
            self.trace = HandTrace(key, bets, [self.config.starting_stack - b for b in bets],
                                   small + big)
        self.trace.consume(obs, self.config.raise_cap)

    def remember(self, obs: Observation, action: Action) -> None:
        if self.trace is not None:
            self.trace.pending = obs, action

    def last_event(self, street: Street) -> Event | None:
        if self.trace and self.trace.events and self.trace.events[-1].street == street:
            return self.trace.events[-1]
        return None

    def turn_size_signal(self, hand: int) -> float:
        """Confidence in half-pot bets showing more strength than pot bets."""
        if not self.turn_bet_strength:
            return 0.0
        (half, n_half), (pot, n_pot) = (r.at(hand) for r in self.turn_bet_strength)
        support = min(n_half, n_pot)
        if support < 6:
            return 0.0
        gap = half / n_half - pot / n_pot
        # Reveals are selective. Require both sizes and a sizeable reversal.
        return min(0.72, max(0.0, 3.6 * (gap - 0.14))) * min(1.0, support / 12)

    def _add(self, kind: str, stage: int, lane: int, hit: bool, hand: int) -> None:
        for key in ((kind, stage, lane), (kind, stage, -1)):
            self.rates.setdefault(key, Frequency()).add(hit, hand)

    def estimate(self, key: tuple[str, int, int], hand: int, prior: float) -> tuple[float, float]:
        kind, stage, _ = key
        hits, total = self.rates[key].at(hand) if key in self.rates else (0.0, 0.0)
        pool_key = (kind, stage, -1)
        if pool_key in self.rates:
            pool_hits, pool_total = self.rates[pool_key].at(hand)
            other = max(0.0, pool_total - total)
            if other:
                share = min(0.35, 18 / other)
                hits += max(0.0, pool_hits - hits) * share
                total += other * share
        if total < 12:
            return prior, 0.0
        return (hits + 18 * prior) / (total + 18), total / (total + 24)

    def bb_estimate(self, key: tuple[str, int, int], street: Street, hand: int, prior: float,
                    flop_line: FlopLine) -> tuple[float, float]:
        rate, trust = self.estimate(key, hand, prior)
        if not self.split_bb_reads or street not in (Street.FLOP, Street.TURN):
            return rate, trust
        kind, _, lane = key
        # Back off from the turn line to the street, then to the pooled read.
        contexts = (None, flop_line) if street == Street.TURN else (None,)
        for context in contexts:
            local = self.bb_rates.get((kind, street, lane, context))
            if local is not None:
                hits, total = local.at(hand)
                if total >= 6:
                    rate = (hits + 18 * rate) / (total + 18)
                    trust = max(trust, total / (total + 24))
        return rate, trust

    def bb_pot_rate(self, event: Event, hand: int) -> float:
        """Estimate pot-sized aggression per legal opportunity in this context."""
        reraise = bool(event.raises and event.due)
        prior = 0.14 if reraise else 0.28
        rate = self.bb_pot_rates.get((event.street, reraise))
        if rate is None:
            return prior
        hits, total = rate.at(hand)
        return (hits + 24 * prior) / (total + 24)

    def estimate_for(self, obs: Observation, key: tuple[str, int, int],
                     prior: float) -> tuple[float, float]:
        if obs.is_button:
            return self.estimate(key, obs.hand_index, prior)
        line = self.trace.flop_line if self.trace else FlopLine.CHECKED_THROUGH
        return self.bb_estimate(key, obs.street, obs.hand_index, prior, line)

    def _record_rate(self, event: Event, key: tuple[str, int, int], hit: bool, hand: int) -> None:
        self._add(*key, hit, hand)
        if self.split_bb_reads and event.button and event.street in (Street.FLOP, Street.TURN):
            kind, _, lane = key
            contexts = (None, event.flop_line) if event.street == Street.TURN else (None,)
            for context in contexts:
                local = self.bb_rates.setdefault((kind, event.street, lane, context), Frequency())
                local.add(hit, hand)

    def finish(self, result: HandResult) -> None:
        trace = self.trace
        if trace is None or trace.key[0] != result.hand_index or result.hand_index in self.finished:
            return
        if trace.pending is None:
            return
        obs, _ = trace.pending
        if result.board[:len(obs.board)] != obs.board:
            return
        terminal = trace.infer_terminal(result, self.config.raise_cap)
        events = trace.events if terminal is None else [*trace.events, terminal]
        hand = result.hand_index
        for event in events:
            stage = phase(event.street)
            if event.button and event.can_raise and event.street in (Street.FLOP, Street.TURN):
                key = (event.street, bool(event.raises and event.due))
                self.bb_pot_rates.setdefault(key, Frequency()).add(event.full_size, hand)
            if event.due and event.raises:
                folded = event.action == Action.FOLD
                self._record_rate(event, ("fold", stage, event.lane), folded, hand)
                if self.learn_flop_pressure and event.street == Street.FLOP and event.raises == 1:
                    self._add("flop_fold", int(not event.button), event.lane, folded, hand)
                if not folded and event.can_raise:
                    self._record_rate(event, ("reraise", stage, event.lane),
                                      event.action in _RAISES, hand)
            elif event.can_raise:
                self._record_rate(event, event.bet_context(), event.action in _RAISES, hand)
        self._record_showdown(result, obs.hole_cards, events)
        self.finished.add(hand)
        self.trace = None

    def _record_showdown(self, result: HandResult, hero: tuple[int, ...],
                         events: list[Event]) -> None:
        villain = result.opp_hole
        if not result.went_to_showdown or villain is None or len(villain) != 5:
            return
        if len(set(hero + result.board + villain)) != 15:
            return
        if self.turn_bet_strength:
            turn = next((e for e in events if e.turn_bet_size is not None), None)
            if turn is not None:
                strength = _shown_strength(hero, villain, result.board[:4], self.turn_showdown_rng)
                self.turn_bet_strength[turn.turn_bet_size].add(strength, result.hand_index)
        event = next((e for e in reversed(events)
                      if e.street != Street.PREFLOP and e.action in _RAISES), None)
        if event is None:
            return
        # A revealed weak bet is evidence of bluffing on this line. Hidden
        # cards and unobserved opportunities never enter the sample.
        board = result.board[:BOARD_CARDS_VISIBLE[event.street]]
        percentile = _shown_strength(hero, villain, board, self.showdown_rng)
        self._record_rate(event, ("weak_bet", phase(event.street), int(event.button)),
                          percentile < 0.33, result.hand_index)


def _shown_strength(hero: tuple[int, ...], villain: tuple[int, ...], board: tuple[int, ...],
                    rng: np.random.Generator) -> float:
    triples = combination_index(np.asarray(board)[_TRIPLES[len(board)]])
    opponents = draw_worlds(unseen_cards(hero, board), 0, 32, rng)
    tables = rank_tables()
    score = tables.best(combination_index(np.asarray(villain)[_PAIRS]), triples)
    reference = tables.best(combination_index(opponents[:, _PAIRS]), triples)
    return float(((reference < score) + 0.5 * (reference == score)).mean())
