"""Local Best Response (Lisý and Bowling 2017): a lower bound on how exploitable the bot is.

A real best response in no-limit hold'em is far out of reach, but an exploiter
that can see the bot's strategy can still do a lot with a simple rule. LBR
keeps the bot's range: every two card hand it could hold, weighted by how
likely it was to make each move it made. At each of its own decisions it
picks the action with the best local value, assuming that after it acts both
players just check or call to showdown:

    check/call:  wp * (pot + c) - c
    raise by a:  fp * pot + (1 - fp) * (wp' * (pot + c + 2a) - (c + a))

where c is the amount to call, wp is LBR's showdown equity against the range
with the rest of the board dealt out, fp is the chance the bot folds to the
raise (read straight from its strategy) and wp' is the equity against only the
hands that don't fold. Folding is worth 0. Following the paper's strongest
setup, LBR only checks or calls before the turn (betting early gives away
information it could exploit later) and bets pot or all-in.

Whatever LBR wins per hand is a lower bound on the bot's exploitability: a
smarter exploiter could only win more.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .betting import FOLD
from .evaluator import eval7
from .search import COMBOS, valid_mask, buckets_for_all


@njit(parallel=True, cache=True)
def showdown_equity(h0, h1, board, n_board, combos, weights):
    """Equity of (h0, h1) against a range (ties count half) over every runout of the board."""
    used = np.zeros(52, np.bool_)
    used[h0] = True
    used[h1] = True
    for i in range(n_board):
        used[board[i]] = True
    deck = np.empty(52, np.int64)
    m = 0
    for c in range(52):
        if not used[c]:
            deck[m] = c
            m += 1
    need = 5 - n_board
    if need == 0:
        n_run = 1
    elif need == 1:
        n_run = m
    else:
        n_run = m * (m - 1) // 2
    r1 = np.full(n_run, -1, np.int64)
    r2 = np.full(n_run, -1, np.int64)
    if need == 1:
        for i in range(m):
            r1[i] = deck[i]
    elif need == 2:
        k = 0
        for i in range(m):
            for j in range(i + 1, m):
                r1[k] = deck[i]
                r2[k] = deck[j]
                k += 1
    num = np.zeros(n_run)
    den = np.zeros(n_run)
    for r in prange(n_run):
        full = np.empty(5, np.int64)
        for i in range(n_board):
            full[i] = board[i]
        if need >= 1:
            full[n_board] = r1[r]
        if need == 2:
            full[n_board + 1] = r2[r]
        mine = eval7(h0, h1, full)
        for j in range(combos.shape[0]):
            w = weights[j]
            if w <= 0:
                continue
            a, b = combos[j, 0], combos[j, 1]
            if a == r1[r] or a == r2[r] or b == r1[r] or b == r2[r]:
                continue
            theirs = eval7(a, b, full)
            den[r] += w
            if mine > theirs:
                num[r] += w
            elif mine == theirs:
                num[r] += 0.5 * w
    total = den.sum()
    return num.sum() / total if total > 0 else 0.5


class LBRAgent:
    """Exploits a CFRAgent (or SearchAgent) using full knowledge of its blueprint strategy."""
    name = "local best response"

    def __init__(self, bot, raise_fracs=(1.0,), allin=True, act_streets=(2, 3)):
        self.bot = bot
        self.raise_fracs = raise_fracs
        self.allin = allin
        self.act_streets = act_streets
        self.new_hand()

    def new_hand(self):
        self.range = None
        self.processed = 0
        self.node = 0
        self.on_tree = True
        self.bucket_cache = {}

    def _buckets(self, street, board):
        key = (street, tuple(board))
        if key not in self.bucket_cache:
            b = self.bot
            self.bucket_cache[key] = buckets_for_all(street, board, b.cuts, b.samples, b.kind,
                                                     b.centroids, b.river_cuts)
        return self.bucket_cache[key]

    def _bot_probs(self, node, street, board, action):
        """Probability of `action` at bot node `node` for every combo (0 for clashing combos)."""
        b = self._buckets(street, board)
        ok = b >= 0
        out = np.zeros(len(COMBOS))
        out[ok] = self.bot.policy[self.bot.tree.row_base[node] + b[ok], action].astype(np.float64)
        return out

    def _update(self, state, seat):
        """Follow the hand on the bot's tree and narrow its range with every bot move."""
        from .engine import HandState
        tree = self.bot.tree
        replay = HandState(state.hole[0], state.hole[1], state.full_board)
        for i, (actor, token, street) in enumerate(state.actions):
            if i < self.processed:
                replay.apply(token)
                continue
            if not self.on_tree or tree.is_terminal[self.node] or tree.player[self.node] != actor:
                self.on_tree = False
                return
            a = self.bot.memo.get(i)
            if a is None and actor == seat and token in ("k", "c"):
                a = self.bot._translation_options(self.node, replay, token)[0][0]
            if a is None:
                self.on_tree = False
                return
            if actor != seat:
                board = list(state.full_board[:(0, 3, 4, 5)[street]])
                self.range *= self._bot_probs(self.node, street, board, a)
            self.node = int(tree.child[self.node, a])
            replay.apply(token)
            self.processed = i + 1

    def _fold_vector(self, state, seat, token):
        """Chance the bot folds to `token`, for every combo it could hold."""
        tree = self.bot.tree
        b = self._buckets(state.street, list(state.board))
        out = np.zeros(len(COMBOS))
        for a, p in self.bot._translation_options(self.node, state, token):
            child = int(tree.child[self.node, a])
            if tree.is_terminal[child] or tree.player[child] != 1 - seat:
                continue
            kinds = list(tree.kind[child, :tree.n_actions[child]])
            if FOLD not in kinds:
                continue
            ok = b >= 0
            out[ok] += p * self.bot.policy[tree.row_base[child] + b[ok], kinds.index(FOLD)].astype(np.float64)
        return out

    def act(self, state, seat):
        hole = state.hole[seat]
        if self.range is None:
            self.range = np.ones(len(COMBOS))
            for c in hole:
                self.range[(COMBOS[:, 0] == c) | (COMBOS[:, 1] == c)] = 0.0
        self._update(state, seat)
        to_call = state.to_call()
        passive = "c" if to_call > 0 else "k"
        if state.street not in self.act_streets or not self.on_tree or self.bot.tree.player[self.node] != seat:
            return passive
        w = self.range * valid_mask(list(state.board))
        if w.sum() <= 0:
            return passive
        w = w / w.sum()
        board = np.zeros(5, np.int64)
        board[:len(state.board)] = state.board
        wp = showdown_equity(hole[0], hole[1], board, len(state.board), COMBOS, w)

        pot, c = state.pot, to_call
        options = {passive: wp * (pot + c) - c}
        if c > 0:
            options["f"] = 0.0
        bounds = state.raise_bounds()
        if bounds is not None:
            opp_in = state.street_in[1 - seat]
            targets = [opp_in + f * (pot + c) for f in self.raise_fracs]
            if self.allin:
                targets.append(bounds[1])
            for to in targets:
                to = int(min(max(round(to), bounds[0]), bounds[1]))
                token = f"b{to}"
                if token in options:
                    continue
                a = to - opp_in
                pf = self._fold_vector(state, seat, token)
                fp = float((w * pf).sum())
                called = w * (1 - pf)
                if called.sum() <= 0:
                    options[token] = pot
                    continue
                wp2 = showdown_equity(hole[0], hole[1], board, len(state.board), COMBOS, called / called.sum())
                options[token] = fp * pot + (1 - fp) * (wp2 * (pot + c + 2 * a) - (c + a))
        return max(options, key=options.get)
