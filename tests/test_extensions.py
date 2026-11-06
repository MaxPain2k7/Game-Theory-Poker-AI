"""Tests for the sampling solver, bigger Leduc variants and opponent exploitation."""

import numpy as np
import pytest

from poker_cfr import GameTree, CFRSolver, exploitability, expected_value, heads_up_ev
from poker_cfr import policy
from poker_cfr.exploit import best_response_to, mix, restricted_nash_response, learn_model
from poker_cfr.mccfr import ExternalSamplingMCCFR


@pytest.fixture(scope="module")
def leduc():
    return GameTree("leduc")


@pytest.fixture(scope="module")
def nash(leduc):
    s = CFRSolver(leduc, "cfr+")
    s.iterate(1500)
    return s.average_strategy()


def test_children_index(leduc):
    kids = np.arange(1, leduc.num_nodes)
    first = leduc.first_child[leduc.parent[kids]]
    assert (kids >= first).all()
    assert (kids < first + leduc.num_children[leduc.parent[kids]]).all()


def test_mccfr_converges_on_kuhn():
    tree = GameTree("kuhn")
    m = ExternalSamplingMCCFR(tree, seed=0)
    m.iterate(30000)
    avg = m.average_strategy()
    assert exploitability(tree, avg) < 0.01
    assert expected_value(tree, avg, avg) == pytest.approx(-1 / 18, abs=0.01)


def test_mccfr_improves_on_leduc(leduc):
    m = ExternalSamplingMCCFR(leduc, seed=0)
    m.iterate(500)
    early = exploitability(leduc, m.average_strategy())
    m.iterate(9500)
    assert exploitability(leduc, m.average_strategy()) < early / 2


def test_bigger_variants_solve():
    tree = GameTree("leduc5")
    assert tree.num_infosets == 780
    s = CFRSolver(tree, "cfr+")
    s.iterate(300)
    assert exploitability(tree, s.average_strategy()) < 0.01


def test_heuristics_are_valid_strategies(leduc):
    for name in policy.HEURISTICS:
        h = policy.heuristic(leduc, name)
        assert np.allclose(h.sum(axis=1), 1)
        assert (h >= 0).all() and (h[~leduc.legal] == 0).all()


def test_mix_endpoints(leduc, nash):
    br = best_response_to(leduc, policy.heuristic(leduc, "lag"))
    assert exploitability(leduc, mix(leduc, [nash, br], [1, 0])) == pytest.approx(exploitability(leduc, nash), abs=1e-9)
    model = policy.heuristic(leduc, "lag")
    halfway = heads_up_ev(leduc, mix(leduc, [nash, br], [0.5, 0.5]), model)
    linear = 0.5 * heads_up_ev(leduc, nash, model) + 0.5 * heads_up_ev(leduc, br, model)
    assert halfway == pytest.approx(linear, abs=1e-9)  # mixing before the hand keeps EV linear


def test_rnr_endpoints(leduc, nash):
    model = policy.heuristic(leduc, "lag")
    r0 = restricted_nash_response(leduc, model, 0.0, 800)
    assert exploitability(leduc, r0) < 2e-3
    r1 = restricted_nash_response(leduc, model, 1.0, 800)
    best = heads_up_ev(leduc, best_response_to(leduc, model), model)
    assert heads_up_ev(leduc, r1, model) == pytest.approx(best, abs=5e-3)


def test_rnr_beats_mixing(leduc, nash):
    model = policy.heuristic(leduc, "nit")
    r = restricted_nash_response(leduc, model, 0.3, 800)
    m = mix(leduc, [nash, best_response_to(leduc, model)], [0.7, 0.3])
    # at least as much edge would cost the mixture far more risk
    assert exploitability(leduc, r) < exploitability(leduc, m) / 10
    assert heads_up_ev(leduc, r, model) > heads_up_ev(leduc, nash, model)


def test_learned_model_falls_back_to_prior(leduc, nash):
    counts = np.zeros_like(nash)
    assert np.allclose(learn_model(leduc, counts, nash), nash)
