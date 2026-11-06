"""External sampling MCCFR for the hold'em abstraction, compiled with numba.

Each iteration draws a deal from the precomputed pool (so card buckets and the
showdown winner are already known), then walks the betting tree once per
player: every action of the player being updated is tried, the opponent's
action is sampled from its current strategy.

Iterations run in parallel threads that share the regret tables without
locks ("Hogwild"). Occasional lost updates are harmless noise next to the
Monte Carlo noise that's already there.

Two tricks from Libratus/Pluribus speed things up:
  * Linear discounting early on: every batch multiplies accumulated regrets
    and strategy sums by t/(t+1), so the bad play of the first iterations
    fades out quickly.
  * Negative regret pruning: once training is under way, actions whose regret
    is very negative are skipped 95% of the time when the traverser explores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit, prange

from .betting import BettingTree, MAX_ACTIONS

PRUNE_THRESHOLD = -300_000.0


@njit(cache=True)
def _strategy(regret, row, n, out):
    total = 0.0
    for a in range(n):
        r = regret[row, a]
        out[a] = r if r > 0 else 0.0
        total += out[a]
    if total > 0:
        for a in range(n):
            out[a] /= total
    else:
        for a in range(n):
            out[a] = 1.0 / n


# not cached: numba's on-disk cache is unreliable for recursive functions
@njit
def _walk(node, trav, deal, prune, is_terminal, player, street, n_actions, child, term_kind,
          folder, contrib, row_base, buckets, winner, regret, strat_sum):
    if is_terminal[node]:
        if term_kind[node] == 0:
            u0 = -contrib[node, 0] if folder[node] == 0 else contrib[node, 1]
        else:
            c = min(contrib[node, 0], contrib[node, 1])
            u0 = winner[deal] * c
        return float(u0) if trav == 0 else float(-u0)

    p = player[node]
    n = n_actions[node]
    row = row_base[node] + buckets[deal, p, street[node]]
    sigma = np.empty(MAX_ACTIONS)
    _strategy(regret, row, n, sigma)

    if p == trav:
        vals = np.zeros(MAX_ACTIONS)
        explored = np.zeros(MAX_ACTIONS, np.bool_)
        v = 0.0
        any_explored = False
        for a in range(n):
            # always walk the last action if everything else was skipped, so a node
            # never returns a made-up value of zero
            last_chance = a == n - 1 and not any_explored
            if prune and not last_chance and regret[row, a] < PRUNE_THRESHOLD and np.random.random() < 0.95:
                continue
            explored[a] = True
            any_explored = True
            vals[a] = _walk(child[node, a], trav, deal, prune, is_terminal, player, street, n_actions,
                            child, term_kind, folder, contrib, row_base, buckets, winner, regret, strat_sum)
            v += sigma[a] * vals[a]
        for a in range(n):
            if explored[a]:
                regret[row, a] += vals[a] - v
        return v

    for a in range(n):
        strat_sum[row, a] += sigma[a]
    x = np.random.random()
    acc = 0.0
    pick = n - 1
    for a in range(n):
        acc += sigma[a]
        if x < acc:
            pick = a
            break
    return _walk(child[node, pick], trav, deal, prune, is_terminal, player, street, n_actions,
                 child, term_kind, folder, contrib, row_base, buckets, winner, regret, strat_sum)


@njit(parallel=True)
def _run_batch(iters, prune, is_terminal, player, street, n_actions, child, term_kind, folder,
               contrib, row_base, buckets, winner, regret, strat_sum):
    n_deals = winner.shape[0]
    total = 0.0
    for i in prange(iters):
        deal = np.random.randint(n_deals)
        for trav in range(2):
            total += _walk(0, trav, deal, prune, is_terminal, player, street, n_actions, child,
                           term_kind, folder, contrib, row_base, buckets, winner, regret, strat_sum)
    return total


@njit(parallel=True, cache=True)
def _scale(arr, factor):
    for i in prange(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] *= factor


class HoldemTrainer:
    def __init__(self, tree: BettingTree, buckets: np.ndarray, winner: np.ndarray, seed: int = 0):
        self.tree = tree
        self.buckets = buckets.astype(np.int64)
        self.winner = winner.astype(np.int64)
        self.regret = np.zeros((tree.num_rows, MAX_ACTIONS), np.float32)
        self.strat_sum = np.zeros((tree.num_rows, MAX_ACTIONS), np.float32)
        self.iterations = 0
        self.batches = 0
        _seed(seed)

    def _args(self):
        t = self.tree
        return (t.is_terminal, t.player, t.street, t.n_actions, t.child, t.term_kind, t.folder,
                t.contrib, t.row_base, self.buckets, self.winner, self.regret, self.strat_sum)

    def run_batch(self, iters: int, prune: bool = False, discount: bool = False):
        _run_batch(iters, prune, *self._args())
        self.iterations += iters
        self.batches += 1
        if discount:
            f = self.batches / (self.batches + 1.0)
            _scale(self.regret, f)
            _scale(self.strat_sum, f)

    def policy(self) -> np.ndarray:
        """Average strategy where it has been visited, current strategy otherwise. (rows, MAX_ACTIONS)"""
        n = self.tree.n_actions
        mask = np.zeros_like(self.regret, dtype=bool)
        dec = np.nonzero(~self.tree.is_terminal)[0]
        counts = self.tree.buckets_per_street[self.tree.street[dec]]
        n_per_row = np.repeat(n[dec], counts)
        mask[np.arange(MAX_ACTIONS)[None, :] < n_per_row[:, None]] = True
        avg = np.where(mask, self.strat_sum, 0.0).astype(np.float64)
        pos = np.where(mask, np.maximum(self.regret, 0.0), 0.0).astype(np.float64)
        uni = mask / n_per_row[:, None]
        cur = np.where(pos.sum(1, keepdims=True) > 0, pos / np.maximum(pos.sum(1, keepdims=True), 1e-30), uni)
        tot = avg.sum(1, keepdims=True)
        return np.where(tot > 0, avg / np.maximum(tot, 1e-30), cur).astype(np.float32)

    def save(self, path, cuts: np.ndarray, **meta):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, regret=self.regret, strat_sum=self.strat_sum, cuts=cuts,
                            iterations=self.iterations, batches=self.batches,
                            buckets=self.tree.buckets, **meta)

    def load(self, path):
        data = np.load(path)
        self.regret[:] = data["regret"]
        self.strat_sum[:] = data["strat_sum"]
        self.iterations = int(data["iterations"])
        self.batches = int(data["batches"])


@njit(cache=True)
def _seed(s):
    np.random.seed(s)
