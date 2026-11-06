"""The abstract betting tree for heads up no-limit hold'em.

Slumbot format: 20,000 chip stacks, blinds 50/100. Seat 0 is the small blind
(the button): it acts first preflop and last on every later street.

The bot can't consider every bet size, so each node offers a few sizes as a
fraction of the pot (the pot after calling, the usual "pot-sized raise"
convention) plus all-in. Sizes depend on the street and on how many raises
have happened so far on it. Facing any real bet the bot maps it onto these
sizes with action translation (see agents.py).

The tree is flattened into numpy arrays so the numba trainer can walk it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STACK = 20_000
SB, BB = 50, 100

FOLD, CALL, RAISE, ALLIN = 0, 1, 2, 3
MAX_ACTIONS = 5

# raise sizes as a fraction of the pot after calling, by raise number on the street
DEFAULT_SIZES = {
    0: [[0.75, 1.5], [0.75, 1.25], [1.0]],   # preflop: open to 2.5bb or 4bb, two 3-bet sizes, one 4-bet size
    1: [[0.33, 0.75, 1.25], [0.75]],         # flop: small, medium and overbet, one raise size
    2: [[0.33, 0.75, 1.25], [0.75]],         # turn
    3: [[0.33, 0.75, 1.25], [0.75]],         # river
}
# all-in stays available as long as the raise count is below this
MAX_RAISES = {0: 4, 1: 3, 2: 3, 3: 3}


@dataclass
class Spot:
    """Betting state, independent of cards."""
    street: int = 0
    contrib: tuple = (SB, BB)        # chips each player has put in this hand
    street_in: tuple = (SB, BB)      # chips each player has put in this street
    to_act: int = 0
    last_raise: int = BB             # size of the last raise on this street (for min-raise)
    raises: int = 0                  # bets plus raises on this street, the big blind doesn't count
    acted: int = 0                   # actions taken on this street
    history: str = ""                # abstract actions, streets separated by "/"

    @property
    def pot(self):
        return self.contrib[0] + self.contrib[1]

    def to_call(self):
        return self.street_in[1 - self.to_act] - self.street_in[self.to_act]


def legal_abstract_actions(s: Spot, sizes=DEFAULT_SIZES, max_raises=MAX_RAISES):
    """List of (kind, fraction, raise_to_street_total) the abstraction allows here."""
    me = s.to_act
    to_call = s.to_call()
    acts = []
    if to_call > 0:
        acts.append((FOLD, 0.0, 0))
    acts.append((CALL, 0.0, 0))
    remaining = STACK - s.contrib[me]
    opp_allin = s.contrib[1 - me] >= STACK
    can_raise = remaining > to_call and not opp_allin and s.raises < max_raises[s.street]
    if not can_raise:
        return acts
    level_sizes = sizes[s.street]
    allin_to = s.street_in[me] + remaining
    if s.raises < len(level_sizes):
        pot_after_call = s.pot + to_call
        min_by = max(s.last_raise, BB)
        for frac in level_sizes[s.raises]:
            by = int(round(frac * pot_after_call))
            if by < min_by:
                continue
            raise_to = s.street_in[1 - me] + by
            if raise_to >= allin_to:
                continue
            acts.append((RAISE, frac, raise_to))
    acts.append((ALLIN, 0.0, allin_to))
    return acts


def apply_action(s: Spot, kind: int, raise_to: int = 0, label: str | None = None):
    """Returns (next_spot, terminal) where terminal is None, ("fold", folder) or ("showdown",)."""
    me, opp = s.to_act, 1 - s.to_act
    street_in = list(s.street_in)
    contrib = list(s.contrib)
    hist = s.history + (label or "")
    if kind == FOLD:
        return Spot(s.street, tuple(contrib), tuple(street_in), me, s.last_raise, s.raises, s.acted + 1, hist), ("fold", me)

    if kind in (RAISE, ALLIN):
        add = raise_to - street_in[me]
        by = raise_to - street_in[opp]
        street_in[me] = raise_to
        contrib[me] += add
        nxt = Spot(s.street, tuple(contrib), tuple(street_in), opp, max(by, s.last_raise),
                   s.raises + 1, s.acted + 1, hist)
        return nxt, None

    # call or check
    to_call = s.to_call()
    add = min(to_call, STACK - contrib[me])
    street_in[me] += add
    contrib[me] += add
    closes = True
    if s.street == 0 and s.raises == 0 and s.acted == 0:
        closes = False      # small blind completes, big blind still has the option
    elif to_call == 0 and s.acted == 0:
        closes = False      # first check of a postflop street
    if not closes:
        return Spot(s.street, tuple(contrib), tuple(street_in), opp, s.last_raise, s.raises,
                    s.acted + 1, hist), None
    if contrib[0] >= STACK or contrib[1] >= STACK or s.street == 3:
        return Spot(s.street, tuple(contrib), tuple(street_in), opp, s.last_raise, s.raises,
                    s.acted + 1, hist), ("showdown",)
    # next street: big blind (seat 1) acts first
    return Spot(s.street + 1, tuple(contrib), (0, 0), 1, BB, 0, 0, hist + "/"), None


def action_label(kind, frac):
    if kind == FOLD:
        return "f"
    if kind == CALL:
        return "c"
    if kind == ALLIN:
        return "A"
    return f"r{frac:g}"


class BettingTree:
    """Flattened abstract betting tree.

    Decision nodes have a player, street and up to MAX_ACTIONS actions, each with
    a child index, a kind (fold, call, raise, all-in), the pot fraction for
    raises and the street total it raises to. Terminal nodes are a fold (with
    who folded) or a showdown. Both store each player's total contribution.
    row_base[node] + bucket is the row for that node in the regret tables.
    """

    def __init__(self, sizes=DEFAULT_SIZES, max_raises=MAX_RAISES, buckets: int = 50):
        self.sizes = sizes
        self.max_raises = max_raises
        self.buckets = buckets
        self.buckets_per_street = np.array([169, buckets, buckets, buckets], np.int64)

        is_terminal, player, street, n_act = [], [], [], []
        child, kind, frac, raise_to = [], [], [], []
        term_kind, folder, c0, c1, histories = [], [], [], [], []

        stack = [(Spot(), None, -1, -1)]
        while stack:
            spot, term, parent, slot = stack.pop()
            idx = len(is_terminal)
            if parent >= 0:
                child[parent][slot] = idx
            histories.append(spot.history)
            c0.append(spot.contrib[0])
            c1.append(spot.contrib[1])
            street.append(spot.street)
            child.append([-1] * MAX_ACTIONS)
            if term is not None:
                is_terminal.append(True)
                player.append(-1)
                n_act.append(0)
                kind.append([-1] * MAX_ACTIONS)
                frac.append([0.0] * MAX_ACTIONS)
                raise_to.append([0] * MAX_ACTIONS)
                term_kind.append(0 if term[0] == "fold" else 1)
                folder.append(term[1] if term[0] == "fold" else -1)
                continue
            acts = legal_abstract_actions(spot, sizes, max_raises)
            is_terminal.append(False)
            player.append(spot.to_act)
            n_act.append(len(acts))
            pad = MAX_ACTIONS - len(acts)
            kind.append([a[0] for a in acts] + [-1] * pad)
            frac.append([a[1] for a in acts] + [0.0] * pad)
            raise_to.append([a[2] for a in acts] + [0] * pad)
            term_kind.append(-1)
            folder.append(-1)
            for k in range(len(acts) - 1, -1, -1):
                a = acts[k]
                nxt, t = apply_action(spot, a[0], a[2], action_label(a[0], a[1]))
                stack.append((nxt, t, idx, k))

        self.is_terminal = np.array(is_terminal, np.bool_)
        self.player = np.array(player, np.int64)
        self.street = np.array(street, np.int64)
        self.n_actions = np.array(n_act, np.int64)
        self.child = np.array(child, np.int64)
        self.kind = np.array(kind, np.int64)
        self.frac = np.array(frac, np.float64)
        self.raise_to = np.array(raise_to, np.int64)
        self.term_kind = np.array(term_kind, np.int64)
        self.folder = np.array(folder, np.int64)
        self.contrib = np.stack([np.array(c0, np.int64), np.array(c1, np.int64)], axis=1)
        self.histories = histories
        self.num_nodes = len(is_terminal)

        # one block of rows per decision node, sized by that street's bucket count
        self.row_base = np.full(self.num_nodes, -1, np.int64)
        dec = np.nonzero(~self.is_terminal)[0]
        sizes_per_node = self.buckets_per_street[self.street[dec]]
        self.row_base[dec] = np.concatenate([[0], np.cumsum(sizes_per_node)[:-1]])
        self.num_rows = int(sizes_per_node.sum())

    def summary(self):
        dec = (~self.is_terminal).sum()
        per_street = [int(((~self.is_terminal) & (self.street == s)).sum()) for s in range(4)]
        return (f"{self.num_nodes:,} nodes, {dec:,} decision nodes (by street {per_street}), "
                f"{self.num_rows:,} infoset rows with {self.buckets} postflop buckets")

    def find(self, history: str) -> int:
        """Node index for an abstract history string like 'r1c/c' (for debugging and tests)."""
        return self.histories.index(history)
