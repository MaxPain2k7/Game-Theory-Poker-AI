"""Real heads up no-limit hold'em with any bet size, using Slumbot's conventions.

Actions are strings: "k" check, "c" call, "f" fold, "b<N>" bet or raise so this
player's total for the street is N. Streets in the history are separated by
"/". Seat 0 is the small blind, acts first preflop and last after that.
"""

from __future__ import annotations

import numpy as np

from .betting import STACK, SB, BB
from .evaluator import eval7


class HandState:
    def __init__(self, hole0, hole1, board):
        self.hole = (tuple(hole0), tuple(hole1))
        self.full_board = tuple(board)       # all five cards, revealed street by street
        self.street = 0
        self.contrib = [SB, BB]
        self.street_in = [SB, BB]
        self.to_act = 0
        self.last_raise = BB
        self.acted = 0
        self.raises = 0
        self.history = ""
        self.folded = None
        self.over = False
        self.actions = []                    # (player, token, street) in order

    # ---- info
    @property
    def board(self):
        return self.full_board[:(0, 3, 4, 5)[self.street]]

    @property
    def pot(self):
        return self.contrib[0] + self.contrib[1]

    def to_call(self):
        return self.street_in[1 - self.to_act] - self.street_in[self.to_act]

    def can_raise(self):
        me = self.to_act
        return (self.contrib[1 - me] < STACK and STACK - self.contrib[me] > self.to_call())

    def raise_bounds(self):
        """(min, max) legal street total for a bet or raise, or None if raising isn't allowed."""
        if not self.can_raise():
            return None
        me, opp = self.to_act, 1 - self.to_act
        max_to = self.street_in[me] + STACK - self.contrib[me]
        min_to = min(self.street_in[opp] + max(self.last_raise, BB), max_to)
        return min_to, max_to

    def legal(self, token: str) -> bool:
        if self.over:
            return False
        if token == "k":
            return self.to_call() == 0
        if token in ("c", "f"):
            return self.to_call() > 0
        if token.startswith("b"):
            b = self.raise_bounds()
            n = int(token[1:])
            return b is not None and b[0] <= n <= b[1]
        return False

    # ---- transitions
    def apply(self, token: str):
        if not self.legal(token):
            raise ValueError(f"illegal action {token!r} at {self.history!r}")
        me, opp = self.to_act, 1 - self.to_act
        self.actions.append((me, token, self.street))
        self.history += token
        if token == "f":
            self.folded = me
            self.over = True
            return self
        if token.startswith("b"):
            n = int(token[1:])
            by = n - self.street_in[opp]
            self.contrib[me] += n - self.street_in[me]
            self.street_in[me] = n
            self.last_raise = max(by, self.last_raise)
            self.raises += 1
            self.acted += 1
            self.to_act = opp
            return self
        # check or call
        to_call = self.to_call()
        self.contrib[me] += to_call
        self.street_in[me] += to_call
        closes = True
        if self.street == 0 and self.raises == 0 and self.acted == 0:
            closes = False
        elif to_call == 0 and self.acted == 0:
            closes = False
        self.acted += 1
        if not closes:
            self.to_act = opp
            return self
        if self.contrib[0] >= STACK or self.contrib[1] >= STACK or self.street == 3:
            self.over = True
            if self.street < 3:
                self.street = 3          # run the board out
            return self
        self.street += 1
        self.street_in = [0, 0]
        self.to_act = 1
        self.last_raise = BB
        self.acted = 0
        self.raises = 0
        self.history += "/"
        return self

    def payoff(self) -> int:
        """Chips won by seat 0 (negative if it lost)."""
        if self.folded == 0:
            return -self.contrib[0]
        if self.folded == 1:
            return self.contrib[1]
        b = np.array(self.full_board, np.int64)
        s0 = eval7(self.hole[0][0], self.hole[0][1], b)
        s1 = eval7(self.hole[1][0], self.hole[1][1], b)
        c = min(self.contrib)
        return c if s0 > s1 else (-c if s1 > s0 else 0)


def play_hand(agent0, agent1, cards, on_action=None) -> HandState:
    """cards: 9 ints, seat 0 hole, seat 1 hole, then the board."""
    state = HandState(cards[0:2], cards[2:4], cards[4:9])
    agents = (agent0, agent1)
    for a in agents:
        a.new_hand()
    while not state.over:
        seat = state.to_act
        token = agents[seat].act(state, seat)
        if on_action:
            on_action(state, seat, token)
        state.apply(token)
    return state
