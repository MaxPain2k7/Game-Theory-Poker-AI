"""Real-time search on the river.

When the bot reaches a river decision it stops looking up its pre-trained
(blueprint) answer and solves the rest of the hand from the exact spot:

  1. Ranges. Both players' 1,081 possible hands (every two card combo that
     doesn't clash with the board) get a weight: how likely that hand was to
     take every action seen so far, under the blueprint strategy. A player who
     raised the turn holds weak hands far less often than one who checked.
  2. Subgame. A betting tree from the current spot with the real pot, the real
     stacks and several bet sizes from here on. No buckets and no translation:
     every hand is its own row and every bet made so far counts at its exact size.
  3. Solve. Vector CFR+ over all hands at once for a few hundred iterations.
     Showdown values use the sorted-strength trick with card removal, so a
     terminal costs O(hands) instead of O(hands^2).

This is "unsafe" re-solving (as in Pluribus): the ranges assume the opponent
played like the blueprint. Libratus used a safe variant with an extra gadget
that guarantees the re-solved strategy is no more exploitable than the
blueprint. Here the head to head results are what decide if it helped.
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit, prange

from .agents import CFRAgent, _check_or_call
from .betting import Spot, legal_abstract_actions, apply_action, FOLD, CALL
from .cards import preflop_class
from .evaluator import eval7, equity_sampled, river_strength

# every two card combo, index 0..1325
COMBOS = np.array([(a, b) for a in range(52) for b in range(a + 1, 52)], np.int64)
COMBO_INDEX = {(int(a), int(b)): i for i, (a, b) in enumerate(COMBOS)}

RIVER_SIZES = {3: [[0.33, 0.75, 1.25, 2.0], [0.75, 1.5], [1.0]]}
RIVER_MAX_RAISES = {3: 4}


def combo_index(c1, c2):
    a, b = min(c1, c2), max(c1, c2)
    return COMBO_INDEX[(a, b)]


def valid_mask(board):
    """Combos that don't share a card with the board."""
    bad = np.zeros(52, bool)
    bad[list(board)] = True
    return ~(bad[COMBOS[:, 0]] | bad[COMBOS[:, 1]])


# ---- bucketing every hand at once

@njit(parallel=True, cache=True)
def _equity_all(combos, mask, board, n_board, samples):
    out = np.zeros(combos.shape[0], np.float32)
    for i in prange(combos.shape[0]):
        if mask[i]:
            out[i] = equity_sampled(combos[i, 0], combos[i, 1], board, n_board, samples)
    return out


@njit(parallel=True, cache=True)
def _river_all(combos, mask, board):
    out = np.zeros(combos.shape[0], np.float32)
    for i in prange(combos.shape[0]):
        if mask[i]:
            out[i] = river_strength(combos[i, 0], combos[i, 1], board)
    return out


@njit(parallel=True, cache=True)
def _rank_all(combos, mask, board):
    out = np.zeros(combos.shape[0], np.int64)
    for i in prange(combos.shape[0]):
        if mask[i]:
            out[i] = eval7(combos[i, 0], combos[i, 1], board)
    return out


def buckets_for_all(street, board, cuts, samples=150, kind="equity", centroids=None, river_cuts=None):
    """Blueprint bucket of every combo on this street's board (-1 where the combo clashes)."""
    mask = valid_mask(board)
    out = np.full(len(COMBOS), -1, np.int64)
    if street == 0:
        out[mask] = [preflop_class(a, b) for a, b in COMBOS[mask]]
        return out
    b = np.zeros(5, np.int64)
    b[:len(board)] = board
    if street == 3:
        s = _river_all(COMBOS, mask, b)
        cut = river_cuts if kind == "dist" else cuts[2]
        out[mask] = np.searchsorted(cut, s[mask])
        return out
    if kind == "dist":
        from .abstraction import dist_bucket_of
        for i in np.nonzero(mask)[0]:
            out[i] = dist_bucket_of(COMBOS[i], list(board), centroids, river_cuts)
        return out
    eq = _equity_all(COMBOS, mask, b, len(board), samples)
    out[mask] = np.searchsorted(cuts[street - 1], eq[mask])
    return out


# ---- terminal values with card removal

@njit(cache=True)
def showdown_values(rank, cards, opp_reach, order, stake):
    """Value of each hand at showdown against the opponent's range, times stake.
    rank: hand strength per combo, order: combos sorted by rank (valid ones only)."""
    n = rank.shape[0]
    out = np.zeros(n)
    card_sum = np.zeros(52)
    total = 0.0
    m = order.shape[0]
    # wins: opponent hands strictly weaker
    i = 0
    while i < m:
        j = i
        while j < m and rank[order[j]] == rank[order[i]]:
            j += 1
        for k in range(i, j):
            h = order[k]
            out[h] += total - card_sum[cards[h, 0]] - card_sum[cards[h, 1]]
        for k in range(i, j):
            h = order[k]
            r = opp_reach[h]
            total += r
            card_sum[cards[h, 0]] += r
            card_sum[cards[h, 1]] += r
        i = j
    # losses: opponent hands strictly stronger
    card_sum[:] = 0.0
    total = 0.0
    i = m - 1
    while i >= 0:
        j = i
        while j >= 0 and rank[order[j]] == rank[order[i]]:
            j -= 1
        for k in range(i, j, -1):
            h = order[k]
            out[h] -= total - card_sum[cards[h, 0]] - card_sum[cards[h, 1]]
        for k in range(i, j, -1):
            h = order[k]
            r = opp_reach[h]
            total += r
            card_sum[cards[h, 0]] += r
            card_sum[cards[h, 1]] += r
        i = j
    for h in range(n):
        out[h] *= stake
    return out


@njit(cache=True)
def compatible_reach(cards, opp_reach, valid):
    """Total opponent reach that doesn't share a card with each hand."""
    n = cards.shape[0]
    card_sum = np.zeros(52)
    total = 0.0
    for h in range(n):
        if valid[h]:
            r = opp_reach[h]
            total += r
            card_sum[cards[h, 0]] += r
            card_sum[cards[h, 1]] += r
    out = np.zeros(n)
    for h in range(n):
        if valid[h]:
            out[h] = total - card_sum[cards[h, 0]] - card_sum[cards[h, 1]] + opp_reach[h]
    return out


# ---- the subgame

class RiverSubgame:
    """Betting tree from a river spot with exact chip amounts, solved with vector CFR+."""

    def __init__(self, spot: Spot, board, sizes=RIVER_SIZES, max_raises=RIVER_MAX_RAISES):
        self.root_player = spot.to_act
        self.board = list(board)
        self.valid = valid_mask(board)
        b = np.array(self.board, np.int64)
        self.rank = _rank_all(COMBOS, self.valid, b)
        idx = np.nonzero(self.valid)[0]
        self.order = idx[np.argsort(self.rank[idx], kind="stable")]

        self.nodes = []
        self._build(spot, None, sizes, max_raises)

    def _build(self, spot, term, sizes, max_raises):
        i = len(self.nodes)
        node = {"contrib": spot.contrib}
        self.nodes.append(node)
        if term is not None:
            node["terminal"] = term[0]
            node["folder"] = term[1] if term[0] == "fold" else -1
            return i
        acts = legal_abstract_actions(spot, sizes, max_raises)
        node["player"] = spot.to_act
        node["actions"] = acts
        node["children"] = []
        for a in acts:
            nxt, t = apply_action(spot, a[0], a[2])
            node["children"].append(self._build(nxt, t, sizes, max_raises))
        return i

    def _terminal(self, node, trav, opp_reach):
        c = node["contrib"]
        if node["terminal"] == "fold":
            comp = compatible_reach(COMBOS, opp_reach, self.valid)
            if node["folder"] == trav:
                return -c[trav] * comp
            return c[1 - trav] * comp
        return showdown_values(self.rank, COMBOS, opp_reach, self.order, float(min(c)))

    def solve(self, ranges, iters=300):
        """ranges: (2, 1326) reach weights for seat 0 and seat 1 at this spot."""
        n = len(COMBOS)
        valid = self.valid.astype(float)
        r = [ranges[0] * valid, ranges[1] * valid]
        self.regret = {}
        self.avg = {}
        for i, node in enumerate(self.nodes):
            if "actions" in node:
                k = len(node["actions"])
                self.regret[i] = np.zeros((k, n))
                self.avg[i] = np.zeros((k, n))
        for t in range(1, iters + 1):
            for trav in (0, 1):
                self._walk(0, trav, r[trav], r[1 - trav], t)
        return self

    def _sigma(self, i):
        pos = np.maximum(self.regret[i], 0.0)
        tot = pos.sum(0)
        k = pos.shape[0]
        return np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), 1.0 / k)

    def _walk(self, i, trav, my_reach, opp_reach, t):
        node = self.nodes[i]
        if "terminal" in node:
            return self._terminal(node, trav, opp_reach)
        sigma = self._sigma(i)
        kids = node["children"]
        if node["player"] == trav:
            vals = np.array([self._walk(c, trav, my_reach * sigma[a], opp_reach, t) for a, c in enumerate(kids)])
            v = (sigma * vals).sum(0)
            self.regret[i] = np.maximum(self.regret[i] + vals - v, 0.0)       # CFR+
            self.avg[i] += t * my_reach * sigma
            return v
        v = 0.0
        for a, c in enumerate(kids):
            v = v + self._walk(c, trav, my_reach, opp_reach * sigma[a], t)
        return v

    def root_strategy(self, combo):
        """Average strategy at the root for one hand: list of (action, probability)."""
        avg = self.avg[0][:, combo]
        tot = avg.sum()
        p = avg / tot if tot > 0 else np.full(len(avg), 1.0 / len(avg))
        return list(zip(self.nodes[0]["actions"], p))

    def root_strategy_all(self):
        avg = self.avg[0]
        tot = avg.sum(0)
        k = avg.shape[0]
        return np.where(tot > 0, avg / np.where(tot > 0, tot, 1.0), 1.0 / k)

    def exploitability(self, ranges):
        """Average over seats of what a best responder gains against the solved strategy,
        in chips per hand, normalised by the ranges' total weight. Used to check convergence."""
        valid = self.valid.astype(float)
        r = [ranges[0] * valid, ranges[1] * valid]
        gains = []
        for br in (0, 1):
            v = self._br(0, br, r[1 - br])
            gains.append((v * r[br]).sum())
        mass = (r[0].sum() * r[1].sum())
        val0 = self._ev(0, r[0], r[1])
        return ((gains[0] - val0) + (gains[1] + val0)) / 2 / max(mass, 1e-12)

    def _avg_sigma(self, i):
        avg = self.avg[i]
        tot = avg.sum(0)
        k = avg.shape[0]
        return np.where(tot > 0, avg / np.where(tot > 0, tot, 1.0), 1.0 / k)

    def _br(self, i, br, opp_reach):
        node = self.nodes[i]
        if "terminal" in node:
            return self._terminal(node, br, opp_reach)
        kids = node["children"]
        if node["player"] == br:
            return np.max([self._br(c, br, opp_reach) for c in kids], axis=0)
        sigma = self._avg_sigma(i)
        return sum(self._br(c, br, opp_reach * sigma[a]) for a, c in enumerate(kids))

    def _ev(self, i, r0, r1):
        """Seat 0's total expected value (summed over both ranges) under the average strategy."""
        node = self.nodes[i]
        if "terminal" in node:
            return (self._terminal(node, 0, r1) * r0).sum()
        sigma = self._avg_sigma(i)
        if node["player"] == 0:
            return sum(self._ev(c, r0 * sigma[a], r1) for a, c in enumerate(node["children"]))
        return sum(self._ev(c, r0, r1 * sigma[a]) for a, c in enumerate(node["children"]))


# ---- the searching bot

class SearchAgent(CFRAgent):
    """Blueprint play preflop, flop and turn, real-time search on the river."""
    name = "CFR bot + river search"

    def __init__(self, *args, iters: int = 200, range_samples: int = 150, **kw):
        super().__init__(*args, **kw)
        self.iters = iters
        self.range_samples = range_samples
        self.searches = 0
        self.search_seconds = 0.0

    def new_hand(self):
        super().new_hand()
        self.override = {}          # action position -> per-combo probability of the move the bot made
        self.all_buckets = {}

    # ---- ranges
    def _path(self, state):
        """[(actor, node, abstract action, street, position)] for every action so far, or None."""
        from .engine import HandState
        replay = HandState(state.hole[0], state.hole[1], state.full_board)
        node, out, t = 0, [], self.tree
        for i, (actor, token, street) in enumerate(state.actions):
            if t.is_terminal[node] or t.player[node] != actor:
                return None
            if i not in self.memo:
                a = self._translate(node, replay, token)
                if a is None:
                    return None
                self.memo[i] = a
            out.append((actor, node, self.memo[i], street, i))
            node = int(t.child[node, self.memo[i]])
            replay.apply(token)
        return out

    def _buckets(self, street, board):
        key = (street, tuple(board))
        if key not in self.all_buckets:
            self.all_buckets[key] = buckets_for_all(street, board, self.cuts, self.range_samples, self.kind,
                                                    self.centroids, self.river_cuts)
        return self.all_buckets[key]

    def ranges(self, state, path):
        """(2, 1326) weights: how likely each combo was to make each player's moves so far."""
        board_now = list(state.board)
        w = np.tile(valid_mask(board_now).astype(float), (2, 1))
        for actor, node, a, street, pos in path:
            if pos in self.override:
                w[actor] *= self.override[pos]
                continue
            board = list(state.full_board[:(0, 3, 4, 5)[street]])
            b = self._buckets(street, board)
            ok = b >= 0
            probs = np.zeros(len(COMBOS))
            probs[ok] = self.policy[self.tree.row_base[node] + b[ok], a].astype(float)
            w[actor] *= probs
        return w

    # ---- acting
    def act(self, state, seat):
        if state.street != 3:
            return super().act(state, seat)
        return self._search_act(state, seat, self._solve_river)

    def _solve_river(self, spot, board, ranges):
        game = RiverSubgame(spot, board).solve(ranges, self.iters)
        return game.root_strategy_all(), game.nodes[0]["actions"]

    def _search_act(self, state, seat, solver):
        """Solve from the current spot with `solver` and play the result. Falls back to the
        blueprint when the hand can't be mapped onto the tree or a range comes out empty."""
        start = time.time()
        path = self._path(state)
        if path is None:
            return CFRAgent.act(self, state, seat)
        r = self.ranges(state, path)
        me = combo_index(*state.hole[seat])
        if r[seat, me] <= 0 or r[1 - seat].sum() <= 0:
            # the blueprint says this line was impossible for one of us: no sensible range to solve with
            return CFRAgent.act(self, state, seat)
        spot = Spot(street=state.street, contrib=tuple(state.contrib), street_in=tuple(state.street_in),
                    to_act=state.to_act, last_raise=state.last_raise, raises=state.raises, acted=state.acted)
        strat, acts = solver(spot, list(state.board), r)
        p = strat[:, me]
        k = int(self.rng.choice(len(acts), p=p / p.sum()))
        self.override[len(state.actions)] = strat[k]
        self.decisions += 1
        self.searches += 1
        self.search_seconds += time.time() - start
        kind, _, raise_to = acts[k]
        if kind == FOLD:
            token = "f" if state.to_call() > 0 else "k"
        elif kind == CALL:
            token = _check_or_call(state)
        else:
            token = f"b{raise_to}"
        if not state.legal(token):
            token = _check_or_call(state)
        return token
