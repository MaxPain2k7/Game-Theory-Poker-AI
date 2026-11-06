"""Vectorized forward and backward passes over a GameTree."""

from __future__ import annotations

import numpy as np

from .tree import GameTree


def edge_probs(tree: GameTree, sigma_flat: np.ndarray) -> np.ndarray:
    """Probability of reaching each node from its parent under sigma."""
    probs = tree.chance_prob.copy()
    dec = tree.is_decision_edge
    probs[dec] = sigma_flat[tree.edge_index[dec]]
    return probs


def reach(tree: GameTree, probs: np.ndarray) -> np.ndarray:
    """Reach probabilities split by contributor: rows are p0, p1, chance."""
    pi = np.ones((3, tree.num_nodes))
    for d in range(1, tree.max_depth + 1):
        lo, hi = tree.starts[d], tree.starts[d + 1]
        par = tree.parent[lo:hi]
        pi[:, lo:hi] = pi[:, par]
        cols = np.arange(lo, hi)
        pi[tree.edge_owner[lo:hi], cols] *= probs[lo:hi]
    return pi


def values(tree: GameTree, probs: np.ndarray) -> np.ndarray:
    """Expected payoff to player 0 at every node when everyone plays probs."""
    v = tree.u0.copy()
    for d in range(tree.max_depth, 0, -1):
        lo, hi = tree.starts[d], tree.starts[d + 1]
        plo, phi = tree.starts[d - 1], tree.starts[d]
        v[plo:phi] += np.bincount(tree.parent[lo:hi] - plo,
                                  weights=probs[lo:hi] * v[lo:hi],
                                  minlength=phi - plo)
    return v


def regret_matching(regrets: np.ndarray, legal: np.ndarray) -> np.ndarray:
    pos = np.where(legal, np.maximum(regrets, 0.0), 0.0)
    total = pos.sum(axis=1, keepdims=True)
    uniform = legal / legal.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, pos / total, uniform)


def normalize(weights: np.ndarray, legal: np.ndarray) -> np.ndarray:
    w = np.where(legal, weights, 0.0)
    total = w.sum(axis=1, keepdims=True)
    uniform = legal / legal.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, w / total, uniform)
