"""Evaluate a saved strategy.

Prints exact exploitability, exact EV against simple baseline bots, then
a sampled match against one baseline played two ways (plain dealing and
duplicate dealing) so you can see how much tighter duplicate gets.

    python evaluate.py strategies/leduc_cfrplus.json --hands 20000
"""

import argparse

from poker_cfr import exploitability, expected_value, heads_up_ev
from poker_cfr import policy
from poker_cfr.simulate import plain_match, duplicate_match


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strategy")
    ap.add_argument("--hands", type=int, default=20000)
    ap.add_argument("--opponent", choices=policy.BASELINES + policy.HEURISTICS, default="calling-station")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tree, sigma, meta = policy.load(args.strategy)
    print(f"{tree.game_name} strategy, {meta.get('algo')} after {meta.get('iterations')} iterations")
    print(f"exploitability : {exploitability(tree, sigma):.8f} chips/hand")
    print(f"game value     : {expected_value(tree, sigma, sigma):+.6f} chips/hand for seat 0\n")

    print("exact EV vs baseline bots (averaged over both seats)")
    names = policy.BASELINES + (policy.HEURISTICS if tree.game_name.startswith("leduc") else ())
    for name in names:
        ev = heads_up_ev(tree, sigma, policy.opponent(tree, name))
        print(f"  {name:<16} {ev:+.4f} chips/hand")

    opp = policy.opponent(tree, args.opponent)
    exact = heads_up_ev(tree, sigma, opp)
    hero, villain = policy.TablePolicy(tree, sigma), policy.TablePolicy(tree, opp)
    print(f"\nsampled match vs {args.opponent} (exact answer {exact:+.4f})")
    plain = plain_match(tree.game_name, hero, villain, args.hands, args.seed)
    dup = duplicate_match(tree.game_name, hero, villain, args.hands, args.seed)
    print(f"  plain      {plain}")
    print(f"  duplicate  {dup}")
    ratio = plain.stderr / dup.stderr
    if ratio >= 1.3:
        print(f"  duplicate cut the standard error by {ratio:.1f}x, so about {ratio ** 2:.1f}x fewer hands for the same precision")
    else:
        print("  duplicate barely helped here: most of the noise comes from the mixed strategies, not the cards")


if __name__ == "__main__":
    main()
