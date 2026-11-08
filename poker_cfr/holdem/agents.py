"""Players for the real game: the CFR bot plus simple opponents to measure it against.

Every agent has new_hand() and act(state, seat) -> action string.
"""

from __future__ import annotations

import numpy as np

from . import abstraction
from .betting import BettingTree, FOLD, CALL, RAISE, ALLIN, STACK
from .cards import preflop_class
from .evaluator import equity_sampled, river_strength

# an opponent bet that commits at least this share of their stack is read as all-in
ALLIN_SHARE = 0.6


def hand_equity(hole, board, samples=600):
    b = np.zeros(5, np.int64)
    b[:len(board)] = board
    if len(board) == 5:
        return float(river_strength(hole[0], hole[1], b))
    return float(equity_sampled(hole[0], hole[1], b, len(board), samples))


def _check_or_call(state):
    return "c" if state.to_call() > 0 else "k"


# ---- baselines

class CallingStation:
    """Never bets, never folds."""
    name = "calling station"

    def new_hand(self):
        pass

    def act(self, state, seat):
        return _check_or_call(state)


class Maniac:
    """Bets or raises the pot whenever it can, otherwise calls."""
    name = "maniac"

    def new_hand(self):
        pass

    def act(self, state, seat):
        b = state.raise_bounds()
        if b is None:
            return _check_or_call(state)
        to = state.street_in[1 - seat] + state.pot + state.to_call()
        return f"b{int(min(max(to, b[0]), b[1]))}"


class RandomBot:
    """Picks fold, check/call or a random legal bet size uniformly."""
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def new_hand(self):
        pass

    def act(self, state, seat):
        opts = [_check_or_call(state)]
        if state.to_call() > 0:
            opts.append("f")
        b = state.raise_bounds()
        if b is not None:
            opts.append(f"b{int(self.rng.integers(b[0], b[1] + 1))}")
        return opts[int(self.rng.integers(len(opts)))]


class EquityBot:
    """Plays its hand strength straight: raises strong hands for value, calls when
    the pot odds are good enough, folds the rest and never bluffs."""
    name = "equity bot"

    def __init__(self, samples=400):
        self.samples = samples

    def new_hand(self):
        pass

    def act(self, state, seat):
        eq = hand_equity(state.hole[seat], state.board, self.samples)
        to_call = state.to_call()
        pot_odds = to_call / (state.pot + to_call) if to_call else 0.0
        b = state.raise_bounds()
        value = 0.62 if state.street == 0 else 0.7
        if eq > value and b is not None and state.raises < 3:
            size = 0.75 * (state.pot + to_call)
            to = int(min(max(state.street_in[1 - seat] + size, b[0]), b[1]))
            return f"b{to}"
        if to_call == 0:
            return "k"
        return "c" if eq > pot_odds + 0.05 else "f"


# ---- the CFR bot

class CFRAgent:
    """Plays a trained abstract strategy in the real game.

    Before each decision it replays the hand's real actions on the abstract tree.
    Real bets that aren't one of the abstract sizes are mapped onto the two nearest
    sizes with the pseudo-harmonic mapping (Ganzfried and Sandholm 2013), picking
    between them at random in proportion. The mapping is remembered for the rest of
    the hand so the bot's picture of the hand stays consistent.

    If the real hand ever leaves the abstract tree (rare, e.g. a raise after the
    abstraction's raise cap) the bot falls back to a simple pot odds rule.
    """
    name = "CFR bot"

    def __init__(self, tree: BettingTree, policy: np.ndarray, cuts: np.ndarray,
                 samples: int = 600, seed: int = 0, kind: str = "equity",
                 centroids: np.ndarray | None = None, river_cuts: np.ndarray | None = None):
        """kind "equity": buckets from equity percentiles (cuts).
        kind "dist": flop and turn buckets from equity histogram clusters (centroids),
        river from strength percentiles (river_cuts)."""
        self.tree = tree
        self.policy = policy
        self.cuts = cuts
        self.kind = kind
        self.centroids = centroids
        self.river_cuts = river_cuts
        self.samples = samples
        self.rng = np.random.default_rng(seed)
        self.fallbacks = 0
        self.decisions = 0
        self.new_hand()

    @classmethod
    def from_file(cls, path, **kw):
        data = np.load(path)
        tree = BettingTree(buckets=int(data["buckets"]))
        if "policy" in data.files:
            policy = data["policy"]
        else:
            from .trainer import HoldemTrainer
            t = HoldemTrainer.__new__(HoldemTrainer)
            t.tree, t.regret, t.strat_sum = tree, data["regret"], data["strat_sum"]
            policy = HoldemTrainer.policy(t)
        kind = str(data["kind"]) if "kind" in data.files else "equity"
        if kind == "dist":
            return cls(tree, policy, None, kind="dist", centroids=data["centroids"],
                       river_cuts=data["river_cuts"], **kw)
        return cls(tree, policy, data["cuts"], **kw)

    def new_hand(self):
        self.memo = {}          # action position -> abstract action index
        self.bucket_cache = {}

    # ---- mapping real actions onto the tree
    @staticmethod
    def _real_fraction(state_before, token):
        opp = 1 - state_before.to_act
        n = int(token[1:])
        return (n - state_before.street_in[opp]) / (state_before.pot + state_before.to_call())

    def _translation_options(self, node, state_before, token):
        """[(abstract action, probability)] a real action maps to at this node, [] if none fits."""
        t = self.tree
        n_act = t.n_actions[node]
        kinds = t.kind[node, :n_act]
        if token == "f":
            hits = np.nonzero(kinds == FOLD)[0]
            return [(int(hits[0]), 1.0)] if len(hits) else []
        if token in ("k", "c"):
            hits = np.nonzero(kinds == CALL)[0]
            return [(int(hits[0]), 1.0)] if len(hits) else []
        n = int(token[1:])
        allin_idx = np.nonzero(kinds == ALLIN)[0]
        me = state_before.to_act
        stack_left = STACK - state_before.contrib[me]
        committed = (n - state_before.street_in[me]) / max(stack_left, 1)
        # mapping a big but not all-in bet to all-in would end the abstract hand
        # while the real one carries on, so only near-shoves count as shoves
        if committed >= ALLIN_SHARE and len(allin_idx):
            return [(int(allin_idx[0]), 1.0)]
        cands = [(float(t.frac[node, a]), a) for a in range(n_act) if kinds[a] == RAISE]
        if not cands:
            return [(int(allin_idx[0]), 1.0)] if len(allin_idx) else []
        cands.sort()
        x = self._real_fraction(state_before, token)
        if x <= cands[0][0]:
            return [(cands[0][1], 1.0)]
        if x >= cands[-1][0]:
            return [(cands[-1][1], 1.0)]
        for (fa, a), (fb, bb) in zip(cands, cands[1:]):
            if fa <= x <= fb:
                p_a = ((fb - x) * (1 + fa)) / ((fb - fa) * (1 + x))
                return [(a, p_a), (bb, 1.0 - p_a)]
        return [(cands[-1][1], 1.0)]

    def _translate(self, node, state_before, token):
        opts = self._translation_options(node, state_before, token)
        if not opts:
            return None
        if len(opts) == 1:
            return opts[0][0]
        (a, p_a), (b, _) = opts
        return a if self.rng.random() < p_a else b

    def _node_for(self, state):
        """Walk the abstract tree along the real history. Returns a node or None if off tree."""
        from .engine import HandState
        replay = HandState(state.hole[0], state.hole[1], state.full_board)
        node = 0
        t = self.tree
        for i, (actor, token, _) in enumerate(state.actions):
            if t.is_terminal[node] or t.player[node] != actor:
                return None
            if i not in self.memo:
                a = self._translate(node, replay, token)
                if a is None:
                    return None
                self.memo[i] = a
            node = int(t.child[node, self.memo[i]])
            replay.apply(token)
        if t.is_terminal[node] or t.player[node] != state.to_act or t.street[node] != state.street:
            return None
        return node

    def _bucket(self, state, seat):
        key = (state.street, state.board)
        if key not in self.bucket_cache:
            hole = state.hole[seat]
            if state.street == 0:
                self.bucket_cache[key] = preflop_class(hole[0], hole[1])
            elif self.kind == "dist":
                self.bucket_cache[key] = abstraction.dist_bucket_of(hole, state.board, self.centroids,
                                                                    self.river_cuts)
            else:
                eq = hand_equity(hole, state.board, self.samples)
                self.bucket_cache[key] = abstraction.bucket_of(state.street, eq, self.cuts)
        return self.bucket_cache[key]

    def strategy_here(self, state, seat):
        """(node, probabilities over the node's actions) or (None, None) off tree."""
        node = self._node_for(state)
        if node is None:
            return None, None
        row = self.tree.row_base[node] + self._bucket(state, seat)
        p = self.policy[row, :self.tree.n_actions[node]].astype(np.float64)
        return node, p / p.sum()

    def act(self, state, seat):
        self.decisions += 1
        node, probs = self.strategy_here(state, seat)
        if node is None:
            self.fallbacks += 1
            return self._fallback(state, seat)
        a = int(self.rng.choice(len(probs), p=probs))
        self.memo[len(state.actions)] = a
        return self._to_real(node, a, state)

    def _to_real(self, node, a, state):
        t = self.tree
        kind = t.kind[node, a]
        if kind == FOLD:
            return "f" if state.to_call() > 0 else "k"
        if kind == CALL:
            return _check_or_call(state)
        b = state.raise_bounds()
        if b is None:
            return _check_or_call(state)
        if kind == ALLIN:
            return f"b{b[1]}"
        to = state.street_in[1 - state.to_act] + t.frac[node, a] * (state.pot + state.to_call())
        return f"b{int(min(max(round(to), b[0]), b[1]))}"

    def _fallback(self, state, seat):
        eq = hand_equity(state.hole[seat], state.board, 300)
        to_call = state.to_call()
        if to_call == 0:
            return "k"
        return "c" if eq > to_call / (state.pot + to_call) else "f"
