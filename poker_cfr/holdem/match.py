"""Head to head matches with duplicate dealing.

Every deal is played four times: the bot in each seat, holding each set of
hole cards. Card luck mostly cancels, which matters a lot in no-limit where
a single hand can swing 400 big blinds.

Results are in big blinds per 100 hands (bb/100), which is the same number as
chips per hand at these blinds. mbb/hand is bb/100 times 10.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import play_hand


@dataclass
class MatchResult:
    hands: int
    bb100: float
    stderr: float

    @property
    def ci95(self):
        return self.bb100 - 1.96 * self.stderr, self.bb100 + 1.96 * self.stderr

    def __str__(self):
        lo, hi = self.ci95
        return f"{self.bb100:+8.1f} bb/100  (95% CI {lo:+.1f} to {hi:+.1f}, {self.hands:,} hands)"


def duplicate_match(hero, villain, deals: int, seed: int = 0, progress=None) -> MatchResult:
    rng = np.random.default_rng(seed)
    per_deal = np.empty(deals)
    for d in range(deals):
        cards = [int(c) for c in rng.permutation(52)[:9]]
        swapped = cards[2:4] + cards[0:2] + cards[4:]
        total = 0.0
        for c in (cards, swapped):
            total += play_hand(hero, villain, c).payoff()
            total -= play_hand(villain, hero, c).payoff()
        per_deal[d] = total / 4.0
        if progress and (d + 1) % progress == 0:
            print(f"    {d + 1:,} deals: {per_deal[:d + 1].mean():+.1f} bb/100", flush=True)
    return MatchResult(deals * 4, float(per_deal.mean()),
                       float(per_deal.std(ddof=1) / np.sqrt(deals)))


def _sync(seed, agents):
    """Same random numbers for every agent and for numba's own generator."""
    from .trainer import _seed
    _seed(seed)
    for a in agents:
        if hasattr(a, "rng"):
            a.rng = np.random.default_rng(seed * 7 + 1)


def paired_comparison(new, base, villain, deals: int, seed: int = 0, progress=None) -> MatchResult:
    """How much better `new` does than `base` against the same opponent, in bb/100.

    Every hand is played twice with identical cards and identical random numbers:
    once with `base` in the hero seat, once with `new`. When `new` only differs from
    `base` in some spots (say, it searches on the river and otherwise plays the
    blueprint), both versions play the hand identically until that spot, so all the
    noise from the rest of the hand cancels. Each deal is used in both seats with
    both sets of hole cards, as in duplicate_match.
    """
    rng = np.random.default_rng(seed)
    diffs = np.empty(deals)
    for d in range(deals):
        cards = [int(c) for c in rng.permutation(52)[:9]]
        swapped = cards[2:4] + cards[0:2] + cards[4:]
        total = 0.0
        k = 0
        for c in (cards, swapped):
            for hero_seat in (0, 1):
                hand_seed = seed * 1_000_003 + d * 4 + k
                k += 1
                results = []
                for hero in (base, new):
                    _sync(hand_seed, (hero, villain))
                    if hero_seat == 0:
                        results.append(play_hand(hero, villain, c).payoff())
                    else:
                        results.append(-play_hand(villain, hero, c).payoff())
                total += results[1] - results[0]
        diffs[d] = total / 4.0
        if progress and (d + 1) % progress == 0:
            se = diffs[:d + 1].std(ddof=1) / np.sqrt(d + 1)
            print(f"    {d + 1:,} deals: {diffs[:d + 1].mean():+.1f} bb/100 (+/- {1.96 * se:.1f})", flush=True)
    return MatchResult(deals * 4, float(diffs.mean()), float(diffs.std(ddof=1) / np.sqrt(deals)))
