"""Tests for showdown-only opponent modelling and Data-Biased Response."""

import numpy as np
import pytest

from poker_cfr import GameTree, CFRSolver, exploitability, heads_up_ev
from poker_cfr import policy
from poker_cfr.exploit import best_response_to, data_biased_response
from poker_cfr.modeling import record_hands, fit_model, count_actions, _opp_decisions


@pytest.fixture(scope="module")
def leduc():
    return GameTree("leduc")


@pytest.fixture(scope="module")
def nash(leduc):
    s = CFRSolver(leduc, "cfr+")
    s.iterate(1500)
    return s.average_strategy()


@pytest.fixture(scope="module")
def nit_hands(leduc, nash):
    true = policy.heuristic(leduc, "nit")
    recs = record_hands(leduc, policy.TablePolicy(leduc, nash), policy.TablePolicy(leduc, true),
                        2000, np.random.default_rng(0))
    return true, recs


def test_replay_matches_recorded_hand(leduc, nit_hands):
    _, recs = nit_hands
    for rec in recs[:200]:
        seq = [a for _, a in _opp_decisions("leduc", rec, rec.opp_card)]
        it = iter(rec.actions)
        assert all(a in it for a in seq)          # opponent moves are a subsequence of the hand
        if rec.opp_seat == 0:
            assert len(seq) >= 1                  # seat 0 always acts first
        keys = [k for k, _ in _opp_decisions("leduc", rec, rec.opp_card)]
        assert all(k[0] == "JQK"[rec.opp_card] for k in keys)


def test_em_spreads_each_hidden_hand_to_one_unit(leduc, nit_hands, nash):
    _, recs = nit_hands
    folded = [r for r in recs if not r.showdown][:50]
    counts = count_actions(leduc, folded, "em", nash)
    oracle = count_actions(leduc, folded, "oracle")
    # same number of opponent decisions, just spread over cards
    assert counts.sum() == pytest.approx(oracle.sum())


def test_selection_bias_and_em_fix(leduc, nit_hands, nash):
    true, recs = nit_hands
    i = leduc.infoset_keys.index("J:r/")
    naive, _ = fit_model(leduc, recs, "showdown", nash)
    em, _ = fit_model(leduc, recs, "em", nash)
    assert true[i, 0] > 0.8
    assert naive[i, 0] < 0.2             # showdown-only data barely ever sees a fold
    assert abs(em[i, 0] - true[i, 0]) < 0.1


def test_dbr_zero_confidence_is_nash_like(leduc, nash):
    model = policy.heuristic(leduc, "lag")
    counts = np.zeros_like(model)
    d = data_biased_response(leduc, model, counts, iters=800)
    assert exploitability(leduc, d) < 3e-3


def test_dbr_full_confidence_exploits(leduc, nash):
    model = policy.heuristic(leduc, "lag")
    counts = np.full_like(model, 1e9)
    d = data_biased_response(leduc, model, counts, k=1.0, p_max=1.0, iters=800)
    best = heads_up_ev(leduc, best_response_to(leduc, model), model)
    assert heads_up_ev(leduc, d, model) > 0.9 * best
