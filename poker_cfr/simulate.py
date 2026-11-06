"""Monte Carlo matches between two policies, with and without duplicate dealing.

The exact EV is known for these small games, which makes them a good place to
see the measurement problem big games have: poker results are noisy. Duplicate
dealing (replaying the same cards with seats and hands swapped) cancels most
of the card luck, so the same number of hands gives a tighter interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .games import new_game, LEDUC_VARIANTS


def deck(game):
    if game == "kuhn":
        return [0, 1, 2]
    cfg = LEDUC_VARIANTS[game]
    return [r for r in range(cfg.ranks) for _ in range(cfg.suits)]


def chance_events(game):
    return 2 if game == "kuhn" else 3


def play_hand(game: str, seat0, seat1, cards, rng) -> float:
    """Play one hand with fixed chance outcomes. Returns payoff to seat 0."""
    state = new_game(game)
    draws = iter(cards)
    while not state.is_terminal():
        if state.is_chance():
            state = state.apply(next(draws))
        else:
            actor = seat0 if state.player() == 0 else seat1
            state = state.apply(actor.act(state, rng))
    return state.returns()


@dataclass
class MatchResult:
    hands: int
    mean: float
    stderr: float

    @property
    def ci95(self):
        return (self.mean - 1.96 * self.stderr, self.mean + 1.96 * self.stderr)

    def __str__(self):
        lo, hi = self.ci95
        return f"{self.mean:+.4f} chips/hand  (95% CI {lo:+.4f} to {hi:+.4f}, {self.hands} hands)"


def _deal(game, rng):
    cards = np.array(deck(game))
    rng.shuffle(cards)
    return [int(c) for c in cards[:chance_events(game)]]


def plain_match(game, hero, villain, hands, seed=0) -> MatchResult:
    """Fresh random cards every hand, hero's seat alternates."""
    rng = np.random.default_rng(seed)
    results = np.empty(hands)
    for k in range(hands):
        cards = _deal(game, rng)
        if k % 2 == 0:
            results[k] = play_hand(game, hero, villain, cards, rng)
        else:
            results[k] = -play_hand(game, villain, hero, cards, rng)
    return MatchResult(hands, results.mean(), results.std(ddof=1) / np.sqrt(hands))


def duplicate_match(game, hero, villain, hands, seed=0) -> MatchResult:
    """Every deal is played four times: hero in both seats, holding both private cards."""
    rng = np.random.default_rng(seed)
    deals = hands // 4
    results = np.empty(deals)
    for k in range(deals):
        cards = _deal(game, rng)
        swapped = [cards[1], cards[0]] + cards[2:]
        total = 0.0
        for c in (cards, swapped):
            total += play_hand(game, hero, villain, c, rng)
            total -= play_hand(game, villain, hero, c, rng)
        results[k] = total / 4.0
    return MatchResult(deals * 4, results.mean(), results.std(ddof=1) / np.sqrt(deals))
