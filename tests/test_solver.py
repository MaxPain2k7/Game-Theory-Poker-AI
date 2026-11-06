"""Checks against numbers that are known independently of this code.

Kuhn's game value is -1/18 for the first player. Leduc's is about -0.0856.
The uniform random NashConv values match OpenSpiel's own tests
(11/12 for Kuhn, 4.747222... for Leduc).
"""

import numpy as np
import pytest

from poker_cfr import GameTree, CFRSolver, exploitability, expected_value, heads_up_ev
from poker_cfr import policy
from poker_cfr.games import KuhnState, LeducState


@pytest.fixture(scope="module")
def kuhn():
    return GameTree("kuhn")


@pytest.fixture(scope="module")
def leduc():
    return GameTree("leduc")


def uniform(tree):
    return tree.legal / tree.legal.sum(axis=1, keepdims=True)


def test_tree_sizes(kuhn, leduc):
    assert kuhn.num_infosets == 12
    # 288 once suits are merged, the usual 936 counts suits separately
    assert leduc.num_infosets == 288


def test_uniform_nashconv(kuhn, leduc):
    assert 2 * exploitability(kuhn, uniform(kuhn)) == pytest.approx(11 / 12)
    assert 2 * exploitability(leduc, uniform(leduc)) == pytest.approx(4.747222222222222)


@pytest.mark.parametrize("algo", ["cfr", "cfr+", "dcfr"])
def test_kuhn_converges(kuhn, algo):
    s = CFRSolver(kuhn, algo)
    s.iterate(3000)
    avg = s.average_strategy()
    assert exploitability(kuhn, avg) < 1e-3
    assert expected_value(kuhn, avg, avg) == pytest.approx(-1 / 18, abs=1e-3)


@pytest.mark.parametrize("algo", ["cfr+", "dcfr"])
def test_leduc_converges(leduc, algo):
    s = CFRSolver(leduc, algo)
    s.iterate(2000)
    avg = s.average_strategy()
    assert exploitability(leduc, avg) < 1e-3
    assert expected_value(leduc, avg, avg) == pytest.approx(-0.0856, abs=1e-3)


def test_exploitability_drops(leduc):
    s = CFRSolver(leduc, "cfr")
    s.iterate(50)
    early = exploitability(leduc, s.average_strategy())
    s.iterate(950)
    assert exploitability(leduc, s.average_strategy()) < early / 4


def test_equilibrium_beats_baselines(leduc):
    s = CFRSolver(leduc, "cfr+")
    s.iterate(1000)
    avg = s.average_strategy()
    for name in policy.BASELINES:
        assert heads_up_ev(leduc, avg, policy.baseline(leduc, name)) > 0


def test_kuhn_payoffs():
    s = KuhnState((2, 0), "bb")
    assert s.is_terminal() and s.returns() == 2
    assert KuhnState((0, 1), "pbp").returns() == -1


def test_leduc_rules():
    s = LeducState()
    for a in (2, 1):          # deal K to seat 0, Q to seat 1
        s = s.apply(a)
    s = s.apply(2)            # seat 0 bets 2
    s = s.apply(2)            # seat 1 raises
    assert s.legal_actions() == [0, 1]   # raise cap reached
    s = s.apply(1)            # call, round over
    assert s.is_chance() and s.contrib == (5, 5)
    s = s.apply(1)            # board Q pairs seat 1
    s = s.apply(1).apply(1)   # check check
    assert s.is_terminal() and s.returns() == -5


def test_policy_roundtrip(tmp_path, kuhn):
    s = CFRSolver(kuhn, "cfr+")
    s.iterate(200)
    avg = s.average_strategy()
    policy.save(tmp_path / "k.json", kuhn, avg, algo="cfr+")
    _, loaded, meta = policy.load(tmp_path / "k.json")
    assert meta["algo"] == "cfr+"
    assert np.allclose(loaded, avg, atol=1e-5)
