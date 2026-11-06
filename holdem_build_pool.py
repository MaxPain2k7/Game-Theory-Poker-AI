"""Precompute the deal pool used to train the hold'em bot.

    python holdem_build_pool.py --deals 2000000

Each deal stores hole cards, board, showdown winner and both players' equity
per street. About 0.4 ms per deal on two cores, so 2M deals is ~12 minutes.
Writes data/holdem_pool.npz.
"""

import argparse
import time

from poker_cfr.holdem import abstraction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=2_000_000)
    ap.add_argument("--samples", type=int, default=600, help="Monte Carlo samples for flop and turn equity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/holdem_pool.npz")
    args = ap.parse_args()

    start = time.time()
    pool = abstraction.build_pool(args.deals, args.samples, args.seed,
                                  log=lambda m: print(m, f"({time.time() - start:.0f}s)", flush=True))
    abstraction.save(args.out, pool)
    print(f"wrote {args.out} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
