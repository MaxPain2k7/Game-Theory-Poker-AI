"""Play heads up no-limit hold'em against the bot in your terminal.

    python holdem_play.py

Type k (check), c (call), f (fold) or a number for a bet/raise to that street
total, e.g. 300. Type a for all-in, q to quit. Seats swap every hand.
"""

import argparse

import numpy as np

from poker_cfr.holdem.agents import CFRAgent
from poker_cfr.holdem.cards import names
from poker_cfr.holdem.engine import HandState

STREETS = ("preflop", "flop", "turn", "river")


def ask(state):
    b = state.raise_bounds()
    opts = ["k" if state.to_call() == 0 else f"c (call {state.to_call()})"]
    if state.to_call() > 0:
        opts.append("f")
    if b:
        opts.append(f"bet/raise to {b[0]}-{b[1]} (a = all-in)")
    while True:
        x = input(f"  {' | '.join(opts)} > ").strip().lower()
        if x == "q":
            raise SystemExit
        if x == "a" and b:
            x = f"b{b[1]}"
        elif x.isdigit():
            x = f"b{x}"
        if state.legal(x):
            return x
        print("  not legal here")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="models/holdem_bot.npz")
    args = ap.parse_args()
    bot = CFRAgent.from_file(args.bot)
    rng = np.random.default_rng()
    total, hands = 0, 0
    while True:
        hands += 1
        you = hands % 2
        cards = [int(c) for c in rng.permutation(52)[:9]]
        state = HandState(cards[0:2], cards[2:4], cards[4:9])
        bot.new_hand()
        print(f"\nhand {hands}: you are the {'small blind (button)' if you == 0 else 'big blind'}, "
              f"your cards {names(state.hole[you])}")
        shown = 0
        while not state.over:
            if state.street > shown:
                shown = state.street
                print(f"  {STREETS[shown]}: {names(state.board)}  pot {state.pot}")
            if state.to_act == you:
                state.apply(ask(state))
            else:
                tok = bot.act(state, 1 - you)
                word = {"k": "checks", "c": "calls", "f": "folds"}.get(tok, f"bets/raises to {tok[1:]}")
                print(f"  bot {word}")
                state.apply(tok)
        result = state.payoff() * (1 if you == 0 else -1)
        total += result
        if state.folded is None:
            print(f"  board {names(state.full_board)} | bot had {names(state.hole[1 - you])}")
        print(f"  you {'won' if result > 0 else 'lost' if result < 0 else 'split'} {abs(result)} chips. "
              f"total {total:+} over {hands} hands ({total / hands:+.0f} bb/100)")


if __name__ == "__main__":
    main()
