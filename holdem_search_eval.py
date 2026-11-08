"""Measure what real-time search adds.

    python holdem_search_eval.py --opponent blueprint --deals 3000
    python holdem_search_eval.py --mode turn --opponent blueprint --deals 750 --seed 1

Paired comparison: every hand is played twice with the same cards and the same
random numbers. --mode river compares river search with the plain blueprint bot,
--mode turn compares turn plus river search with river search only. The two bots
play identically until the street where they differ, so the difference isolates
that street's search. Writes results/holdem_search_<mode>_<opponent>[_seed].md.
"""

import argparse
import time
from pathlib import Path

from poker_cfr.holdem.agents import CFRAgent, EquityBot, CallingStation, Maniac
from poker_cfr.holdem.match import paired_comparison
from poker_cfr.holdem.search import SearchAgent
from poker_cfr.holdem.turn_search import TurnSearchAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="models/holdem_bot.npz")
    ap.add_argument("--mode", choices=["river", "turn"], default="river")
    ap.add_argument("--opponent", choices=["blueprint", "equity", "station", "maniac"], default="blueprint")
    ap.add_argument("--deals", type=int, default=3000)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--turn-iters", type=int, default=200)
    ap.add_argument("--turn-cards", type=int, default=4)
    args = ap.parse_args()

    if args.mode == "river":
        base = CFRAgent.from_file(args.bot)
        new = SearchAgent.from_file(args.bot, iters=args.iters)
    else:
        base = SearchAgent.from_file(args.bot, iters=args.iters)
        new = TurnSearchAgent.from_file(args.bot, iters=args.iters, turn_iters=args.turn_iters,
                                        turn_cards=args.turn_cards)
    villain = {"blueprint": lambda: CFRAgent.from_file(args.bot), "equity": EquityBot,
               "station": CallingStation, "maniac": Maniac}[args.opponent]()
    start = time.time()
    res = paired_comparison(new, base, villain, args.deals, seed=args.seed, progress=250)
    lo, hi = res.ci95
    label = "river search minus blueprint" if args.mode == "river" else "turn search minus river search"
    line = (f"{label} vs {args.opponent}: {res.bb100:+.1f} bb/100 (95% CI {lo:+.1f} to {hi:+.1f}), "
            f"{res.hands:,} paired hands, {new.searches:,} searches, "
            f"{(time.time() - start) / 60:.0f} min, se {res.stderr:.2f}")
    print(line, flush=True)
    Path("results").mkdir(exist_ok=True)
    suffix = "" if args.seed == 7 else f"_{args.seed}"
    name = args.opponent if args.mode == "river" else f"turn_{args.opponent}"
    Path(f"results/holdem_search_{name}{suffix}.md").write_text(line + "\n")


if __name__ == "__main__":
    main()
