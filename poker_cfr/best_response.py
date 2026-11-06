"""Exact best response, exploitability and head to head expected value.

A best responder picks one action per infoset (it can't see the opponent's
card), so at each of its infosets it adds up the value of every action over
all the histories it can't tell apart, weighted by how likely the opponent
and chance are to have produced each one, then takes the best. Because every
history of an infoset sits at the same depth, one bottom up sweep does it.
"""

from __future__ import annotations

import numpy as np

from .passes import edge_probs, reach, values
from .tree import GameTree


def combine(tree: GameTree, sigma0: np.ndarray, sigma1: np.ndarray) -> np.ndarray:
    """One strategy table where player 0 rows come from sigma0 and player 1 rows from sigma1."""
    is_p0 = (tree.infoset_player == 0)[:, None]
    return np.where(is_p0, sigma0, sigma1)


def best_response_value(tree: GameTree, sigma: np.ndarray, br_player: int):
    """Value to br_player of best responding to sigma, plus the best response itself."""
    A = tree.num_actions
    sigma_flat = sigma.ravel()
    probs = edge_probs(tree, sigma_flat)
    pi = reach(tree, probs)
    weight = pi[1 - br_player] * pi[2]

    sign = 1.0 if br_player == 0 else -1.0
    v = tree.u0 * sign
    best = np.zeros(tree.num_infosets, dtype=np.int64)

    for d in range(tree.max_depth, 0, -1):
        lo, hi = tree.starts[d], tree.starts[d + 1]
        plo, phi = tree.starts[d - 1], tree.starts[d]
        kids = np.arange(lo, hi)
        par = tree.parent[lo:hi]
        br_edge = (tree.edge_owner[lo:hi] == br_player)

        # nodes not controlled by the best responder: plain expectation
        other = ~br_edge
        v[plo:phi] += np.bincount(par[other] - plo,
                                  weights=probs[lo:hi][other] * v[lo:hi][other],
                                  minlength=phi - plo)
        if not br_edge.any():
            continue

        # nodes controlled by the best responder: choose per infoset
        bk, bp = kids[br_edge], par[br_edge]
        eidx = tree.edge_index[bk]
        q = np.bincount(eidx, weights=weight[bp] * v[bk],
                        minlength=tree.num_infosets * A).reshape(-1, A)
        infosets = np.unique(tree.infoset[bp])
        q_rows = np.where(tree.legal[infosets], q[infosets], -np.inf)
        best[infosets] = q_rows.argmax(axis=1)
        chosen = tree.edge_slot[bk] == best[tree.infoset[bp]]
        v[bp[chosen]] = v[bk[chosen]]

    br = np.zeros_like(sigma)
    mine = tree.infoset_player == br_player
    br[np.nonzero(mine)[0], best[mine]] = 1.0
    return float(v[0]), br


def exploitability(tree: GameTree, sigma: np.ndarray) -> float:
    """Average of what each seat could win by best responding. Zero at a Nash equilibrium."""
    v0, _ = best_response_value(tree, sigma, 0)
    v1, _ = best_response_value(tree, sigma, 1)
    return (v0 + v1) / 2.0


def expected_value(tree: GameTree, sigma0: np.ndarray, sigma1: np.ndarray) -> float:
    """Exact expected payoff to player 0 when seat 0 plays sigma0 and seat 1 plays sigma1."""
    sigma = combine(tree, sigma0, sigma1)
    probs = edge_probs(tree, sigma.ravel())
    return float(values(tree, probs)[0])


def heads_up_ev(tree: GameTree, hero: np.ndarray, villain: np.ndarray) -> float:
    """Hero's exact EV per hand, averaged over both seats."""
    as_p0 = expected_value(tree, hero, villain)
    as_p1 = -expected_value(tree, villain, hero)
    return (as_p0 + as_p1) / 2.0
