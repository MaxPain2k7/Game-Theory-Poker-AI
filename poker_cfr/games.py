"""Game rules for Kuhn poker and Leduc hold'em.

Both games are two player, zero sum and small enough to solve exactly.
Every state exposes the same small interface so the tree builder, the
solver and the play/simulation code never need to know which game it is.

Action slots are fixed per game so strategies line up as (infoset, slot):
  Kuhn:  0 = pass (check or fold), 1 = bet (bet or call)
  Leduc: 0 = fold, 1 = call (or check), 2 = raise (or bet)
"""

from __future__ import annotations

from typing import NamedTuple

RANKS = "JQK"
ALL_RANKS = "23456789TJQKA"


def rank_chars(n):
    """Names for n ranks, topping out at K so 3 ranks read J Q K like the classic game."""
    if n == 13:
        return ALL_RANKS
    return ALL_RANKS[12 - n:12]


class KuhnState:
    """Kuhn poker. Three cards (J, Q, K), one card each, ante 1, one bet of 1."""

    NUM_ACTIONS = 2
    ACTION_NAMES = ("p", "b")
    ACTION_LABELS = ("pass", "bet")
    NAME = "kuhn"

    __slots__ = ("hands", "history")

    def __init__(self, hands=(), history=""):
        self.hands = hands
        self.history = history

    # chance
    def is_chance(self):
        return len(self.hands) < 2

    def chance_outcomes(self):
        left = [r for r in range(3) if r not in self.hands]
        return [(r, 1.0 / len(left)) for r in left]

    # decisions
    def is_terminal(self):
        return self.history in ("pp", "bp", "bb", "pbp", "pbb")

    def player(self):
        return len(self.history) % 2

    def legal_actions(self):
        return [0, 1]

    def apply(self, action):
        if self.is_chance():
            return KuhnState(self.hands + (action,), self.history)
        return KuhnState(self.hands, self.history + self.ACTION_NAMES[action])

    def returns(self):
        """Payoff to player 0 at a terminal state."""
        h = self.history
        if h == "bp":
            return 1.0
        if h == "pbp":
            return -1.0
        stake = 1.0 if h == "pp" else 2.0
        return stake if self.hands[0] > self.hands[1] else -stake

    def infoset_key(self):
        return f"{RANKS[self.hands[self.player()]]}:{self.history}"

    def rank_char(self, card):
        return RANKS[card]

    def describe_for(self, seat):
        return f"your card {RANKS[self.hands[seat]]} | actions so far: {self.history or '-'}"


class LeducConfig(NamedTuple):
    name: str
    ranks: int
    suits: int
    ranks_str: str


LEDUC_VARIANTS = {
    "leduc": LeducConfig("leduc", 3, 2, rank_chars(3)),
    "leduc5": LeducConfig("leduc5", 5, 2, rank_chars(5)),
    "leduc13": LeducConfig("leduc13", 13, 2, rank_chars(13)),
}


class LeducState:
    """Leduc hold'em.

    The classic game uses six cards (two each of J, Q, K). Each player antes 1
    and gets one private card. Round one bets are 2, then one public card is
    dealt and round two bets are 4. At most two raises per round. At showdown
    a pair with the board wins, otherwise the higher card wins, equal ranks
    split. The leduc5 and leduc13 variants keep the rules but use 5 or 13
    ranks, which grows the tree so the sampling solvers have something to do.
    """

    NUM_ACTIONS = 3
    ACTION_NAMES = ("f", "c", "r")
    ACTION_LABELS = ("fold", "call", "raise")
    BET_SIZES = (2, 4)
    MAX_RAISES = 2

    __slots__ = ("cfg", "hands", "public", "rnd", "hist", "contrib", "raises",
                 "to_act", "facing", "folded", "over", "need_public")

    def __init__(self, variant="leduc"):
        self.cfg = LEDUC_VARIANTS[variant]
        self.hands = ()
        self.public = None
        self.rnd = 0
        self.hist = ("", "")
        self.contrib = (1, 1)
        self.raises = 0
        self.to_act = 0
        self.facing = False
        self.folded = None
        self.over = False
        self.need_public = False

    def _copy(self):
        s = LeducState.__new__(LeducState)
        for name in self.__slots__:
            setattr(s, name, getattr(self, name))
        return s

    @property
    def NAME(self):
        return self.cfg.name

    # chance
    def is_chance(self):
        return len(self.hands) < 2 or self.need_public

    def chance_outcomes(self):
        counts = [self.cfg.suits] * self.cfg.ranks
        for c in self.hands:
            counts[c] -= 1
        if self.public is not None:
            counts[self.public] -= 1
        total = sum(counts)
        return [(r, counts[r] / total) for r in range(self.cfg.ranks) if counts[r] > 0]

    # decisions
    def is_terminal(self):
        return self.over

    def player(self):
        return self.to_act

    def legal_actions(self):
        if self.facing:
            return [0, 1, 2] if self.raises < self.MAX_RAISES else [0, 1]
        return [1, 2]

    def apply(self, action):
        s = self._copy()
        if self.is_chance():
            if len(s.hands) < 2:
                s.hands = s.hands + (action,)
            else:
                s.public = action
                s.need_public = False
            return s

        p = s.to_act
        hist = list(s.hist)
        acted_before = len(hist[s.rnd]) > 0
        hist[s.rnd] += self.ACTION_NAMES[action]
        s.hist = tuple(hist)
        contrib = list(s.contrib)

        if action == 0:  # fold
            s.folded = p
            s.over = True
            return s

        if action == 2:  # raise or bet
            contrib[p] = contrib[1 - p] + self.BET_SIZES[s.rnd]
            s.contrib = tuple(contrib)
            s.raises += 1
            s.facing = True
            s.to_act = 1 - p
            return s

        # call or check
        round_over = s.facing or acted_before
        contrib[p] = contrib[1 - p]
        s.contrib = tuple(contrib)
        s.facing = False
        if not round_over:
            s.to_act = 1 - p
            return s
        if s.rnd == 0:
            s.rnd = 1
            s.raises = 0
            s.to_act = 0
            s.need_public = True
        else:
            s.over = True
        return s

    def returns(self):
        """Payoff to player 0 at a terminal state."""
        if self.folded == 0:
            return -float(self.contrib[0])
        if self.folded == 1:
            return float(self.contrib[1])
        a, b = self._strength(self.hands[0]), self._strength(self.hands[1])
        if a == b:
            return 0.0
        return float(self.contrib[0]) if a > b else -float(self.contrib[0])

    def _strength(self, card):
        return 100 + card if card == self.public else card

    def rank_char(self, card):
        return self.cfg.ranks_str[card]

    def infoset_key(self):
        me = self.rank_char(self.hands[self.to_act])
        board = self.rank_char(self.public) if self.public is not None else ""
        return f"{me}{board}:{self.hist[0]}/{self.hist[1]}"

    def describe_for(self, seat):
        board = self.rank_char(self.public) if self.public is not None else "?"
        pot = sum(self.contrib)
        return (f"your card {self.rank_char(self.hands[seat])} | board {board} | pot {pot} | "
                f"round 1: {self.hist[0] or '-'}  round 2: {self.hist[1] or '-'}")


GAMES = ("kuhn",) + tuple(LEDUC_VARIANTS)


def new_game(name):
    if name == "kuhn":
        return KuhnState()
    if name in LEDUC_VARIANTS:
        return LeducState(name)
    raise ValueError(f"unknown game {name}, pick from {GAMES}")
