"""Checks the numba evaluator against a slow, obviously correct reference."""

import itertools
import random

import numpy as np
import pytest

from poker_cfr.holdem.cards import parse_many, preflop_class, preflop_class_name
from poker_cfr.holdem.evaluator import eval_cards, equity_sampled, river_strength


def ref5(cards):
    """Classic 5 card ranking as a comparable tuple."""
    ranks = sorted((c // 4 for c in cards), reverse=True)
    suits = [c % 4 for c in cards]
    flush = len(set(suits)) == 1
    uniq = sorted(set(ranks), reverse=True)
    straight_top = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_top = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:
            straight_top = 3
    counts = sorted(((ranks.count(r), r) for r in uniq), reverse=True)
    shape = [c for c, _ in counts]
    order = [r for _, r in counts]
    if straight_top is not None and flush:
        return (8, straight_top)
    if shape == [4, 1]:
        return (7, *order)
    if shape == [3, 2]:
        return (6, *order)
    if flush:
        return (5, *ranks)
    if straight_top is not None:
        return (4, straight_top)
    if shape == [3, 1, 1]:
        return (3, *order)
    if shape == [2, 2, 1]:
        return (2, *order)
    if shape == [2, 1, 1, 1]:
        return (1, *order)
    return (0, *ranks)


def ref7(cards):
    return max(ref5(c) for c in itertools.combinations(cards, 5))


def fast(cards):
    return int(eval_cards(np.array(cards, dtype=np.int64), len(cards)))


def test_matches_reference_on_random_pairs():
    rng = random.Random(1)
    for _ in range(3000):
        deck = rng.sample(range(52), 9)
        board = deck[4:]
        a, b = deck[:2] + board, deck[2:4] + board
        ra, rb = ref7(a), ref7(b)
        fa, fb = fast(a), fast(b)
        assert (ra > rb) == (fa > fb) and (ra == rb) == (fa == fb), (a, b)


@pytest.mark.parametrize("better,worse", [
    ("AsKsQsJsTs2d3c", "9h9d9c9s2c3d4h"),   # royal flush beats quads
    ("5h4h3h2hAh9c9d", "AcAdAsKsKd2c3c"),   # steel wheel beats a full house
    ("Ac2d3h4s5c9dKd", "KcKdQhJs9c8d2h"),   # wheel beats a pair
    ("AcAdAsKcKd2c3c", "AhKh2h3h9hQdQc"),   # full house beats a flush
    ("QcQdQh2s2c9d8h", "JcJdJhAsAc9d8h"),   # higher trips in a full house wins
    ("AhAs8c8dKc3d2c", "AcAd8h8s4c4d2h"),   # third pair doesn't count, K kicker wins
])
def test_known_orderings(better, worse):
    assert fast(parse_many(better)) > fast(parse_many(worse))


def test_preflop_classes_cover_169():
    seen = {preflop_class(a, b) for a in range(52) for b in range(52) if a != b}
    assert seen == set(range(169))
    assert preflop_class_name(preflop_class(*parse_many("AsKs"))) == "AKs"
    assert preflop_class_name(preflop_class(*parse_many("7d2c"))) == "72o"
    assert preflop_class_name(preflop_class(*parse_many("QhQs"))) == "QQ"


def test_equity_sanity():
    np.random.seed(0)
    aa = equity_sampled(*parse_many("AsAh"), np.zeros(5, np.int64), 0, 20000)
    assert aa == pytest.approx(0.852, abs=0.01)     # AA vs a random hand
    nuts = river_strength(*parse_many("AsKs"), np.array(parse_many("QsJsTs2d3c"), np.int64))
    assert nuts == 1.0
