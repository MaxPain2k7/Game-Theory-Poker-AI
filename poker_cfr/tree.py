"""Flattens a game into numpy arrays so CFR can run as vectorized passes.

Nodes are stored in breadth first order, so every depth level is one
contiguous slice. A forward pass (reach probabilities) walks the levels top
down and a backward pass (values) walks them bottom up, with one numpy call
per level instead of one Python call per node.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .games import new_game

TERMINAL, CHANCE, DECISION = 0, 1, 2
OWNER_CHANCE = 2
MODEL_PREFIX = "model|"


class GameTree:
    def __init__(self, game_name: str, rnr: tuple[int, float] | None = None,
                 dbr: tuple[int, dict] | None = None):
        """rnr=(model_seat, p) builds the Restricted Nash Response game: a coin
        flip at the root decides whether model_seat is locked to a fixed
        opponent model (probability p) or free to best respond. Its infosets
        in the locked half get a "model|" prefix so they can be frozen.

        dbr=(model_seat, confidence) builds the Data-Biased Response game: the
        coin flip happens separately at every decision of model_seat, landing
        on "use the model" with that spot's confidence (by infoset key, 0 if
        missing). So the model is trusted where there was data and ignored
        where there wasn't."""
        if rnr is not None and dbr is not None:
            raise ValueError("pick rnr or dbr, not both")
        root = new_game(game_name)
        self.game_name = game_name
        self.rnr = rnr
        self.dbr = dbr
        self.num_actions = root.NUM_ACTIONS
        self.action_names = root.ACTION_NAMES

        node_type, player, infoset, parent = [], [], [], []
        edge_owner, edge_slot, edge_prob, depth, u0 = [], [], [], [], []
        infoset_ids: dict[str, int] = {}
        infoset_player, infoset_depth, infoset_legal = [], [], []

        if rnr is None:
            queue = deque([(root, -1, -1, -1, 1.0, 0, False, None)])
        else:
            # extra chance node at the root: model branch with probability p
            _, p_model = rnr
            for col, val in ((node_type, CHANCE), (player, -1), (infoset, -1), (parent, -1),
                             (edge_owner, -1), (edge_slot, -1), (edge_prob, 1.0), (depth, 0), (u0, 0.0)):
                col.append(val)
            queue = deque([(root, 0, OWNER_CHANCE, -1, p_model, 1, True, None),
                           (root, 0, OWNER_CHANCE, -1, 1.0 - p_model, 1, False, None)])
        while queue:
            state, par, owner, slot, prob, d, locked, gate = queue.popleft()
            idx = len(node_type)
            parent.append(par)
            edge_owner.append(owner)
            edge_slot.append(slot)
            edge_prob.append(prob)
            depth.append(d)

            if state.is_terminal():
                node_type.append(TERMINAL)
                player.append(-1)
                infoset.append(-1)
                u0.append(state.returns())
                continue
            u0.append(0.0)

            if state.is_chance():
                node_type.append(CHANCE)
                player.append(-1)
                infoset.append(-1)
                for outcome, p in state.chance_outcomes():
                    queue.append((state.apply(outcome), idx, OWNER_CHANCE, -1, p, d + 1, locked, None))
                continue

            p = state.player()
            key = state.infoset_key()
            if dbr is not None and p == dbr[0] and gate is None:
                # a coin flip in front of this decision: model or free choice
                conf = float(dbr[1].get(key, 0.0))
                node_type.append(CHANCE)
                player.append(-1)
                infoset.append(-1)
                queue.append((state, idx, OWNER_CHANCE, -1, conf, d + 1, False, "model"))
                queue.append((state, idx, OWNER_CHANCE, -1, 1.0 - conf, d + 1, False, "free"))
                continue
            if (locked and p == rnr[0]) or gate == "model":
                key = MODEL_PREFIX + key
            legal = state.legal_actions()
            if key not in infoset_ids:
                infoset_ids[key] = len(infoset_player)
                infoset_player.append(p)
                infoset_depth.append(d)
                mask = [False] * self.num_actions
                for a in legal:
                    mask[a] = True
                infoset_legal.append(mask)
            i = infoset_ids[key]
            # every history in one infoset must sit at the same depth,
            # the level-by-level best response relies on it
            assert infoset_depth[i] == d, f"infoset {key} spans depths"
            node_type.append(DECISION)
            player.append(p)
            infoset.append(i)
            for a in legal:
                queue.append((state.apply(a), idx, p, a, 1.0, d + 1, locked, None))

        self.node_type = np.array(node_type, dtype=np.int8)
        self.player = np.array(player, dtype=np.int8)
        self.infoset = np.array(infoset, dtype=np.int64)
        self.parent = np.array(parent, dtype=np.int64)
        self.edge_owner = np.array(edge_owner, dtype=np.int64)
        self.edge_slot = np.array(edge_slot, dtype=np.int64)
        self.chance_prob = np.array(edge_prob, dtype=np.float64)
        self.depth = np.array(depth, dtype=np.int64)
        self.u0 = np.array(u0, dtype=np.float64)

        self.infoset_keys = list(infoset_ids.keys())
        self.infoset_player = np.array(infoset_player, dtype=np.int8)
        self.legal = np.array(infoset_legal, dtype=bool)
        self.num_infosets = len(self.infoset_keys)
        self.num_nodes = len(self.node_type)

        # level slices: nodes of depth d live in [starts[d], starts[d+1])
        max_d = int(self.depth.max())
        self.starts = np.searchsorted(self.depth, np.arange(max_d + 2))
        self.max_depth = max_d

        # flat (infoset, slot) index for every decision edge, -1 elsewhere
        is_dec_edge = self.edge_owner[1:] < 2
        is_dec_edge = np.concatenate([[False], is_dec_edge & (self.edge_owner[1:] >= 0)])
        self.is_decision_edge = is_dec_edge
        self.edge_index = np.full(self.num_nodes, -1, dtype=np.int64)
        kids = np.nonzero(is_dec_edge)[0]
        self.edge_index[kids] = self.infoset[self.parent[kids]] * self.num_actions + self.edge_slot[kids]

        # per player lists of decision edges (child node ids), used for updates
        self.player_edges = [kids[self.edge_owner[kids] == p] for p in (0, 1)]

        # BFS puts each node's children next to each other, so a start index
        # and a count are enough to find them (used by the sampling solver)
        self.num_children = np.bincount(self.parent[1:], minlength=self.num_nodes)
        self.first_child = np.full(self.num_nodes, -1, dtype=np.int64)
        has_kids = self.num_children > 0
        first = np.full(self.num_nodes, np.iinfo(np.int64).max)
        np.minimum.at(first, self.parent[1:], np.arange(1, self.num_nodes))
        self.first_child[has_kids] = first[has_kids]

    def summary(self):
        n_term = int((self.node_type == TERMINAL).sum())
        return (f"{self.game_name}: {self.num_nodes} nodes, {n_term} terminals, "
                f"{self.num_infosets} infosets, depth {self.max_depth}")
