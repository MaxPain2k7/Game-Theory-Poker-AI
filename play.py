"""Play against a trained bot in the terminal.

    python play.py strategies/leduc_cfrplus.json

You switch seats every hand. Type the letter of an action (f, c, r for
Leduc, p, b for Kuhn) or q to quit.
"""

import argparse

import numpy as np

from poker_cfr import policy
from poker_cfr.games import new_game


def labels(state):
    """Action words that match the spot: check/bet when nothing is pending, fold/call/raise otherwise."""
    if state.NAME.startswith("leduc"):
        return state.ACTION_LABELS if state.facing else ("fold", "check", "bet")
    facing = state.history.endswith("b")
    return ("fold", "call") if facing else ("check", "bet")


def prompt(state):
    legal = state.legal_actions()
    names = state.ACTION_NAMES
    words = labels(state)
    options = "  ".join(f"[{names[a]}] {words[a]}" for a in legal)
    while True:
        choice = input(f"  {options} > ").strip().lower()
        if choice == "q":
            raise SystemExit
        if choice in names and names.index(choice) in legal:
            return names.index(choice)
        print("  not a legal move")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strategy")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    tree, sigma, _ = policy.load(args.strategy)
    bot = policy.TablePolicy(tree, sigma)
    rng = np.random.default_rng(args.seed)
    total, hand = 0.0, 0

    while True:
        hand += 1
        you = (hand + 1) % 2
        state = new_game(tree.game_name)
        print(f"\nhand {hand} - you are seat {you + 1} ({'first' if you == 0 else 'second'} to act)")
        while not state.is_terminal():
            if state.is_chance():
                dealing_board = getattr(state, "need_public", False)
                outs = state.chance_outcomes()
                card = int(rng.choice([o for o, _ in outs], p=[p for _, p in outs]))
                state = state.apply(card)
                if dealing_board:
                    print(f"  board card: {state.rank_char(card)}")
                continue
            if state.player() == you:
                print("  " + state.describe_for(you))
                state = state.apply(prompt(state))
            else:
                a = bot.act(state, rng)
                print(f"  bot: {labels(state)[a]}")
                state = state.apply(a)

        result = state.returns() * (1 if you == 0 else -1)
        total += result
        print(f"  bot had {state.rank_char(state.hands[1 - you])}. you {'won' if result > 0 else 'lost' if result < 0 else 'split'} "
              f"{abs(result):g}. running total {total:+g} over {hand} hands")


if __name__ == "__main__":
    main()
