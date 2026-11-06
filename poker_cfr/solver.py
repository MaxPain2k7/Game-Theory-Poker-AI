"""Counterfactual regret minimization: vanilla CFR, CFR+ and Discounted CFR.

All three share the same loop. Each iteration does one pass per player
(alternating updates): build the current strategy from regrets, push reach
probabilities down the tree, pull values back up, then credit every action
with how much better it would have done than the strategy actually played,
weighted by how likely the opponent and chance were to get us there.

  cfr     plain regret matching, uniform averaging (Zinkevich et al. 2007)
  cfr+    regrets floored at zero, linear averaging (Tammelin 2014)
  dcfr    positive regrets discounted by t^1.5/(t^1.5+1), negative ones
          halved, average weighted by (t/(t+1))^2 (Brown and Sandholm 2019)
"""

from __future__ import annotations

import numpy as np

from .passes import edge_probs, reach, regret_matching, normalize, values
from .tree import GameTree

ALGOS = ("cfr", "cfr+", "dcfr")


class CFRSolver:
    def __init__(self, tree: GameTree, algo: str = "cfr+",
                 alpha: float = 1.5, beta: float = 0.0, gamma: float = 2.0,
                 frozen: tuple[np.ndarray, np.ndarray] | None = None):
        """frozen=(mask, table) pins the infosets where mask is True to fixed
        rows of table. Used for opponent models in Restricted Nash Response."""
        if algo not in ALGOS:
            raise ValueError(f"algo must be one of {ALGOS}")
        self.tree = tree
        self.algo = algo
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        shape = (tree.num_infosets, tree.num_actions)
        self.regrets = np.zeros(shape)
        self.strategy_sum = np.zeros(shape)
        self.iteration = 0
        self.frozen = frozen

        # cached per player edge data
        self._edges = []
        for p in (0, 1):
            kids = tree.player_edges[p]
            self._edges.append((kids, tree.parent[kids], tree.edge_index[kids]))

    def _pin(self, sigma: np.ndarray) -> np.ndarray:
        if self.frozen is not None:
            mask, table = self.frozen
            sigma[mask] = table[mask]
        return sigma

    def current_strategy(self) -> np.ndarray:
        return self._pin(regret_matching(self.regrets, self.tree.legal))

    def average_strategy(self) -> np.ndarray:
        return self._pin(normalize(self.strategy_sum, self.tree.legal))

    def _update_player(self, p: int, t: int):
        tree = self.tree
        size = tree.num_infosets * tree.num_actions
        sigma = self.current_strategy()
        sigma_flat = sigma.ravel()
        probs = edge_probs(tree, sigma_flat)
        pi = reach(tree, probs)
        v = values(tree, probs)
        if p == 1:
            v = -v

        kids, parents, eidx = self._edges[p]
        cf_reach = pi[1 - p, parents] * pi[2, parents]
        inst_regret = cf_reach * (v[kids] - v[parents])
        delta = np.bincount(eidx, weights=inst_regret, minlength=size)
        self.regrets += delta.reshape(self.regrets.shape)

        # player p's own reach is the same for every history in an infoset,
        # so summing it per history only rescales a row, which normalizing removes
        own = pi[p, parents] * sigma_flat[eidx]
        contrib = np.bincount(eidx, weights=own, minlength=size).reshape(self.regrets.shape)

        if self.algo == "cfr":
            self.strategy_sum += contrib
        elif self.algo == "cfr+":
            self.regrets = np.maximum(self.regrets, 0.0)
            self.strategy_sum += t * contrib
        else:  # dcfr
            self.strategy_sum += contrib

    def _discount(self, t: int):
        if self.algo != "dcfr":
            return
        pos = t ** self.alpha / (t ** self.alpha + 1)
        neg = t ** self.beta / (t ** self.beta + 1)
        self.regrets = np.where(self.regrets > 0, self.regrets * pos, self.regrets * neg)
        self.strategy_sum *= (t / (t + 1)) ** self.gamma

    def iterate(self, n: int = 1):
        for _ in range(n):
            self.iteration += 1
            t = self.iteration
            for p in (0, 1):
                self._update_player(p, t)
            self._discount(t)
