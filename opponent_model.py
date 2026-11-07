"""Half-local opponent estimates from public actions and revealed showdowns."""

from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class Event:
    street: Street
    action: Action
    due: int
    pot: int
    raises: int
    can_raise: bool
    button: bool

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

    def at(self, hand: int) -> tuple[float, float]:
        decay = _DECAY ** max(0, hand - self.hand)
        return self.hits * decay, self.opportunities * decay

    def add(self, hit: bool, hand: int) -> None:
        now = max(hand, self.hand)
        hits, opportunities = self.at(now)
        weight = _DECAY ** (now - hand)
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
                self.events.append(Event(
                    self.street, record.action, due, self.pot, self.raises,
                    self.raises < cap and self.stacks[1] > due and self.stacks[0] > 0,
                    not obs.is_button,
                ))
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
                         can_raise, not obs.is_button)
        if (action == Action.CHECK_CALL and obs.street == Street.RIVER and not obs.is_button
                and not obs.to_call and result.went_to_showdown and result.pot == obs.pot):
            return Event(obs.street, Action.CHECK_CALL, 0, obs.pot, 0,
                         bool(obs.my_stack and obs.opp_stack), True)
        return None


class OpponentModel:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.trace: HandTrace | None = None
        self.rates: dict[tuple[str, int, int], Frequency] = {}
        self.finished: set[int] = set()
        self.showdown_rng = np.random.default_rng(852781)

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
            if event.due and event.raises:
                folded = event.action == Action.FOLD
                self._add("fold", stage, event.lane, folded, hand)
                if not folded and event.can_raise:
                    self._add("reraise", stage, event.lane, event.action in _RAISES, hand)
            elif event.can_raise:
                self._add(*event.bet_context(), event.action in _RAISES, hand)
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
        event = next((e for e in reversed(events)
                      if e.street != Street.PREFLOP and e.action in _RAISES), None)
        if event is None:
            return
        # A revealed weak bet is evidence of bluffing on this line. Hidden
        # cards and unobserved opportunities never enter the sample.
        board = result.board[:BOARD_CARDS_VISIBLE[event.street]]
        triples = combination_index(np.asarray(board)[_TRIPLES[len(board)]])
        opponents = draw_worlds(unseen_cards(hero, board), 0, 32, self.showdown_rng)
        score = rank_tables().best(combination_index(np.asarray(villain)[_PAIRS]), triples)
        reference = rank_tables().best(combination_index(opponents[:, _PAIRS]), triples)
        percentile = float(((reference < score) + 0.5 * (reference == score)).mean())
        self._add("weak_bet", phase(event.street), int(event.button), percentile < 0.33,
                  result.hand_index)
