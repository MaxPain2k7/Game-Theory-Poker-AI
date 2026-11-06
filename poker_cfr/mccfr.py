"""External sampling Monte Carlo CFR (Lanctot et al. 2009).

Full CFR touches every node of the tree on every iteration. That stops being
possible once the tree is big (heads up limit hold'em has around 10^14
decision points). External sampling walks one sampled path for chance and the
opponent, while still trying every action of the player being updated. Each
iteration is far cheaper, the regret estimates are noisy but unbiased, and
the average strategy still converges to Nash.

It's written in plain Python because it follows one path at a time, the way
it has to in games too large to vectorize over the whole tree. Progress is
measured in nodes touched so it can be compared fairly with full traversal CFR.
"""

from __future__ import annotations

import random

import numpy as np

from .tree import GameTree, TERMINAL, CHANCE


class ExternalSamplingMCCFR:
    def __init__(self, tree: GameTree, seed: int = 0, linear: bool = False):
        """linear=True weights iteration t by t (Linear CFR). Off by default, it made
        little difference on these games."""
        self.tree = tree
        self.linear = linear
        self.rng = random.Random(seed)
        self.iteration = 0
        self.nodes_touched = 0

        # plain Python lists are much faster than numpy for single element access
        self.node_type = tree.node_type.tolist()
        self.infoset = tree.infoset.tolist()
        self.player = tree.player.tolist()
        self.first_child = tree.first_child.tolist()
        self.num_children = tree.num_children.tolist()
        self.u0 = tree.u0.tolist()
        self.prob = tree.chance_prob.tolist()
        self.legal = [np.nonzero(row)[0].tolist() for row in tree.legal]
        self.regrets = [[0.0] * len(l) for l in self.legal]
        self.strategy_sum = [[0.0] * len(l) for l in self.legal]

    @staticmethod
    def _match(regrets):
        pos = [r if r > 0 else 0.0 for r in regrets]
        total = sum(pos)
        if total > 0:
            return [p / total for p in pos]
        return [1.0 / len(regrets)] * len(regrets)

    def _sample(self, probs):
        x = self.rng.random()
        acc = 0.0
        for k, p in enumerate(probs):
            acc += p
            if x < acc:
                return k
        return len(probs) - 1

    def _walk(self, node, traverser, weight):
        """Returns the sampled value of node for the traverser."""
        self.nodes_touched += 1
        kind = self.node_type[node]
        if kind == TERMINAL:
            u = self.u0[node]
            return u if traverser == 0 else -u

        first, n = self.first_child[node], self.num_children[node]
        if kind == CHANCE:
            k = self._sample(self.prob[first:first + n])
            return self._walk(first + k, traverser, weight)

        I = self.infoset[node]
        sigma = self._match(self.regrets[I])
        if self.player[node] == traverser:
            vals = [self._walk(first + k, traverser, weight) for k in range(n)]
            v = sum(s * x for s, x in zip(sigma, vals))
            reg = self.regrets[I]
            for k in range(n):
                reg[k] += weight * (vals[k] - v)
            return v

        # opponent node: credit its average strategy here, then follow one action
        ssum = self.strategy_sum[I]
        for k in range(n):
            ssum[k] += weight * sigma[k]
        return self._walk(first + self._sample(sigma), traverser, weight)

    def iterate(self, n: int = 1):
        for _ in range(n):
            self.iteration += 1
            w = float(self.iteration) if self.linear else 1.0
            for p in (0, 1):
                self._walk(0, p, w)

    def average_strategy(self) -> np.ndarray:
        tree = self.tree
        sigma = np.zeros((tree.num_infosets, tree.num_actions))
        for I, legal in enumerate(self.legal):
            row = self.strategy_sum[I]
            total = sum(row)
            for k, a in enumerate(legal):
                sigma[I, a] = row[k] / total if total > 0 else 1.0 / len(legal)
        return sigma
