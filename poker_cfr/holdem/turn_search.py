"""Real-time search on the turn.

On a turn decision the bot solves the rest of the hand: the turn betting from
the current spot, every river card that can come and the river betting after
it. Like the river search it works on ranges over all 1,326 combos with exact
chip amounts, but the tree is far larger (one river subtree per river card),
so the solver is compiled with numba.

  * The betting tree doesn't depend on the cards, so it is built once. River
    decisions get their own regrets per river card: a hand plays a river that
    made a flush differently from one that paired the board.
  * Each iteration samples K of the 48 unseen cards as river cards (K = 48 is
    exact). A combo holding the sampled card is dropped for that card. Pairs of
    hands that don't clash keep each card with probability 44/48, so river
    values are scaled by 48/44 to stay unbiased next to the turn fold payoffs.
  * The river part uses a coarse bet menu to keep the tree small. That only
    shapes the turn decision: when the river actually comes, the river search
    solves it again with the full menu.

Regrets use CFR+ and the turn strategy is a linearly weighted average.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .betting import legal_abstract_actions, apply_action
from .search import (COMBOS, SearchAgent, valid_mask, _rank_all, showdown_values,
                     compatible_reach)

TURN_SIZES = {2: [[0.5, 1.0], [1.0]], 3: [[0.75]]}
TURN_MAX_RAISES = {2: 3, 3: 2}
MAXA = 4
DECISION, FOLD_END, SHOWDOWN_END = 0, 1, 2


@njit
def _walk(n, trav, my, opp, slot, S, K, scale, w,
          kind, player, street, nch, child, folder, c0, c1, row,
          combos, valid_t, valid_r, rank_r, order_r, n_order,
          reg_t, avg_t, reg_r):
    H = my.shape[0]
    if kind[n] == FOLD_END:
        if slot < 0:
            comp = compatible_reach(combos, opp, valid_t)
        else:
            comp = compatible_reach(combos, opp, valid_r[slot])
        if folder[n] == trav:
            stake = -float(c0[n] if trav == 0 else c1[n])
        else:
            stake = float(c1[n] if trav == 0 else c0[n])
        return comp * stake

    if kind[n] == SHOWDOWN_END:
        stake = float(min(c0[n], c1[n]))
        if slot >= 0:
            return showdown_values(rank_r[slot], combos, opp, order_r[slot, :n_order[slot]], stake)
        # all-in on the turn: deal the river and show down
        acc = np.zeros(H)
        for i in range(K):
            s = S[i]
            v = showdown_values(rank_r[s], combos, opp * valid_r[s], order_r[s, :n_order[s]], stake)
            for h in range(H):
                if valid_r[s, h]:
                    acc[h] += v[h]
        return acc * (scale / K)

    m = nch[n]
    r = row[n]
    sigma = np.empty((m, H))
    for h in range(H):
        tot = 0.0
        for a in range(m):
            x = reg_t[r, a, h] if slot < 0 else reg_r[r, slot, a, h]
            if x < 0:
                x = 0.0
            sigma[a, h] = x
            tot += x
        if tot > 0:
            for a in range(m):
                sigma[a, h] /= tot
        else:
            for a in range(m):
                sigma[a, h] = 1.0 / m

    vals = np.zeros((m, H))
    mover = player[n] == trav
    for a in range(m):
        c = child[n, a]
        my_c = my * sigma[a] if mover else my
        opp_c = opp if mover else opp * sigma[a]
        if slot < 0 and street[c] == 3:
            # the turn betting is over: average over the sampled river cards
            for i in range(K):
                s = S[i]
                v = _walk(c, trav, my_c * valid_r[s], opp_c * valid_r[s], s, S, K, scale, w,
                          kind, player, street, nch, child, folder, c0, c1, row,
                          combos, valid_t, valid_r, rank_r, order_r, n_order, reg_t, avg_t, reg_r)
                for h in range(H):
                    if valid_r[s, h]:
                        vals[a, h] += v[h]
            for h in range(H):
                vals[a, h] *= scale / K
        else:
            vals[a] = _walk(c, trav, my_c, opp_c, slot, S, K, scale, w,
                            kind, player, street, nch, child, folder, c0, c1, row,
                            combos, valid_t, valid_r, rank_r, order_r, n_order, reg_t, avg_t, reg_r)

    v = np.zeros(H)
    if not mover:
        for a in range(m):
            for h in range(H):
                v[h] += vals[a, h]
        return v
    for h in range(H):
        for a in range(m):
            v[h] += sigma[a, h] * vals[a, h]
    for a in range(m):
        for h in range(H):
            if slot < 0:
                x = reg_t[r, a, h] + vals[a, h] - v[h]
                reg_t[r, a, h] = x if x > 0 else 0.0
                avg_t[r, a, h] += w * my[h] * sigma[a, h]
            else:
                x = reg_r[r, slot, a, h] + vals[a, h] - v[h]
                reg_r[r, slot, a, h] = x if x > 0 else 0.0
    return v


@njit
def _solve(iters, K, r0, r1, seed, wpow, kind, player, street, nch, child, folder, c0, c1, row,
           combos, valid_t, valid_r, rank_r, order_r, n_order, reg_t, avg_t, reg_r):
    np.random.seed(seed)
    n_cards = valid_r.shape[0]
    scale = n_cards / (n_cards - 4.0)
    # walk through a shuffled deck K cards at a time, reshuffling after each full pass,
    # so every river card gets the same number of visits
    order = np.random.permutation(n_cards)
    pos = 0
    S = np.empty(K, np.int64)
    for t in range(1, iters + 1):
        for i in range(K):
            if pos == n_cards:
                order = np.random.permutation(n_cards)
                pos = 0
            S[i] = order[pos]
            pos += 1
        for trav in range(2):
            my = r0 if trav == 0 else r1
            opp = r1 if trav == 0 else r0
            _walk(0, trav, my, opp, -1, S, K, scale, float(t) ** wpow,
                  kind, player, street, nch, child, folder, c0, c1, row,
                  combos, valid_t, valid_r, rank_r, order_r, n_order, reg_t, avg_t, reg_r)


class TurnSubgame:
    """Turn plus river betting from a turn spot, solved over ranges with sampled river cards."""

    def __init__(self, spot, board, sizes=TURN_SIZES, max_raises=TURN_MAX_RAISES):
        assert len(board) == 4 and spot.street == 2
        self.board = list(board)
        kind, player, street, nch, child, folder, c0, c1, row, acts = [], [], [], [], [], [], [], [], [], []
        n_turn = n_river = 0
        stack = [(spot, None, -1, -1)]
        while stack:
            sp, term, parent, slot = stack.pop()
            i = len(kind)
            if parent >= 0:
                child[parent][slot] = i
            street.append(sp.street)
            c0.append(sp.contrib[0])
            c1.append(sp.contrib[1])
            child.append([-1] * MAXA)
            if term is not None:
                kind.append(FOLD_END if term[0] == "fold" else SHOWDOWN_END)
                folder.append(term[1] if term[0] == "fold" else -1)
                player.append(-1); nch.append(0); row.append(-1); acts.append(None)  # noqa: E702
                continue
            a_list = legal_abstract_actions(sp, sizes, max_raises)
            kind.append(DECISION)
            folder.append(-1)
            player.append(sp.to_act)
            nch.append(len(a_list))
            acts.append(a_list)
            if sp.street == 2:
                row.append(n_turn)
                n_turn += 1
            else:
                row.append(n_river)
                n_river += 1
            for k in range(len(a_list) - 1, -1, -1):
                a = a_list[k]
                nxt, t = apply_action(sp, a[0], a[2])
                stack.append((nxt, t, i, k))

        as_int = lambda x: np.array(x, np.int64)  # noqa: E731
        self.kind, self.player, self.street = as_int(kind), as_int(player), as_int(street)
        self.nch, self.child, self.folder = as_int(nch), as_int(child), as_int(folder)
        self.c0, self.c1, self.row = as_int(c0), as_int(c1), as_int(row)
        self.actions = acts
        self.n_turn, self.n_river = n_turn, n_river

        # per river card: which combos survive, hand ranks and the showdown order
        self.valid_t = valid_mask(board)
        self.deck = np.array([c for c in range(52) if c not in board], np.int64)
        n = len(COMBOS)
        self.valid_r = np.zeros((len(self.deck), n), np.bool_)
        self.rank_r = np.zeros((len(self.deck), n), np.int64)
        self.order_r = np.zeros((len(self.deck), n), np.int64)
        self.n_order = np.zeros(len(self.deck), np.int64)
        for s, card in enumerate(self.deck):
            v = self.valid_t & (COMBOS[:, 0] != card) & (COMBOS[:, 1] != card)
            rank = _rank_all(COMBOS, v, np.array(self.board + [int(card)], np.int64))
            idx = np.nonzero(v)[0]
            order = idx[np.argsort(rank[idx], kind="stable")]
            self.valid_r[s], self.rank_r[s] = v, rank
            self.order_r[s, :len(order)] = order
            self.n_order[s] = len(order)

    def solve(self, ranges, iters=150, cards=8, seed=0, wpow=1.0):
        """ranges: (2, 1326) reach weights at this spot. cards: river cards sampled per iteration."""
        n = len(COMBOS)
        r0 = (ranges[0] * self.valid_t).astype(np.float64)
        r1 = (ranges[1] * self.valid_t).astype(np.float64)
        self.reg_t = np.zeros((max(self.n_turn, 1), MAXA, n))
        self.avg_t = np.zeros((max(self.n_turn, 1), MAXA, n))
        self.reg_r = np.zeros((max(self.n_river, 1), len(self.deck), MAXA, n), np.float32)
        _solve(iters, min(cards, len(self.deck)), r0, r1, seed, wpow, self.kind, self.player, self.street,
               self.nch, self.child, self.folder, self.c0, self.c1, self.row, COMBOS, self.valid_t,
               self.valid_r, self.rank_r, self.order_r, self.n_order, self.reg_t, self.avg_t, self.reg_r)
        return self

    def root_strategy_all(self):
        k = self.nch[0]
        avg = self.avg_t[self.row[0], :k]
        tot = avg.sum(0)
        return np.where(tot > 0, avg / np.where(tot > 0, tot, 1.0), 1.0 / k)

    def root_actions(self):
        return self.actions[0]

    def root_values(self, ranges, trav, iters_done=0):
        """Expected value of each of trav's combos at the root under the current regrets,
        with every river card enumerated. Used by the tests."""
        r0 = (ranges[0] * self.valid_t).astype(np.float64)
        r1 = (ranges[1] * self.valid_t).astype(np.float64)
        my, opp = (r0, r1) if trav == 0 else (r1, r0)
        K = len(self.deck)
        reg_t, avg_t, reg_r = self.reg_t.copy(), self.avg_t.copy(), self.reg_r.copy()
        return _walk(0, trav, my, opp, -1, np.arange(K), K, K / (K - 4.0), 1.0,
                     self.kind, self.player, self.street, self.nch, self.child, self.folder,
                     self.c0, self.c1, self.row, COMBOS, self.valid_t, self.valid_r, self.rank_r,
                     self.order_r, self.n_order, reg_t, avg_t, reg_r)


class TurnSearchAgent(SearchAgent):
    """Blueprint preflop and flop, real-time search on the turn and the river."""
    name = "CFR bot + turn and river search"

    def __init__(self, *args, turn_iters: int = 200, turn_cards: int = 8, turn_wpow: float = 2.0, **kw):
        super().__init__(*args, **kw)
        self.turn_iters = turn_iters
        self.turn_cards = turn_cards
        self.turn_wpow = turn_wpow
        self.turn_searches = 0
        self.turn_seconds = 0.0

    def act(self, state, seat):
        if state.street != 2:
            return super().act(state, seat)
        before = self.search_seconds
        token = self._search_act(state, seat, self._solve_turn)
        self.turn_seconds += self.search_seconds - before
        return token

    def _solve_turn(self, spot, board, ranges):
        self.turn_searches += 1
        seed = int(self.rng.integers(2 ** 31))
        game = TurnSubgame(spot, board).solve(ranges, self.turn_iters, self.turn_cards, seed, self.turn_wpow)
        return game.root_strategy_all(), game.root_actions()
