"""Equity histograms for every deal in the pool, for distribution-aware buckets.

    python holdem_build_hist.py

Reads data/holdem_pool.npz, writes data/holdem_hist.npz. About 0.35 ms per
deal on two cores, so 2M deals take ~12 minutes.
"""

import time

import numpy as np

from poker_cfr.holdem import abstraction


def main():
    start = time.time()
    pool = abstraction.load("data/holdem_pool.npz")
    hists = abstraction.build_histograms(pool["cards"],
                                         log=lambda m: print(m, f"({time.time() - start:.0f}s)", flush=True))
    np.savez_compressed("data/holdem_hist.npz", hists=hists)
    print(f"wrote data/holdem_hist.npz in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
