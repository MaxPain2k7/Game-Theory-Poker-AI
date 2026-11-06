"""Card abstraction: a precomputed pool of deals with each player's hand strength per street.

Computing equity is by far the most expensive part of a hold'em iteration, so
it is done once up front. Each deal in the pool stores both players' hole
cards, the board, the showdown winner and each player's equity on the flop,
turn and river. Training then samples deals from the pool, which turns an
iteration into a pure tree walk.

Two bucketings are supported. Equity buckets use equity percentiles, so each
bucket holds about the same share of hands: flop and turn use equity against a
random hand with the rest of the board dealt out (which counts draws), the
river uses exact strength against all opponent hands. Histogram buckets
(further down) cluster the whole distribution of river strength instead.
Preflop always uses the 169 lossless starting-hand classes. Both are imperfect
recall: a bucket on the turn doesn't remember which flop bucket it came from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit, prange

from .cards import preflop_class
from .evaluator import eval7, equity_sampled, river_strength, equity_histogram


@njit(parallel=True, cache=True)
def _fill_pool(seed, n, samples):
    cards = np.empty((n, 9), np.int64)
    eq = np.empty((n, 2, 3), np.float32)
    winner = np.empty(n, np.int8)
    np.random.seed(seed)
    for i in prange(n):
        deck = np.arange(52)
        for j in range(9):
            k = j + np.random.randint(52 - j)
            deck[j], deck[k] = deck[k], deck[j]
        for j in range(9):
            cards[i, j] = deck[j]
        board = deck[4:9].copy()
        for p in range(2):
            a, b = deck[2 * p], deck[2 * p + 1]
            eq[i, p, 0] = equity_sampled(a, b, board, 3, samples)
            eq[i, p, 1] = equity_sampled(a, b, board, 4, samples)
            eq[i, p, 2] = river_strength(a, b, board)
        s0 = eval7(deck[0], deck[1], board)
        s1 = eval7(deck[2], deck[3], board)
        winner[i] = 1 if s0 > s1 else (-1 if s1 > s0 else 0)
    return cards, eq, winner


def build_pool(n: int, samples: int = 600, seed: int = 0, chunk: int = 100_000, log=print):
    parts = []
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        parts.append(_fill_pool(seed * 1_000_003 + start, m, samples))
        if log:
            log(f"  pool {start + m:,}/{n:,}")
    cards = np.concatenate([p[0] for p in parts])
    eq = np.concatenate([p[1] for p in parts])
    winner = np.concatenate([p[2] for p in parts])
    pre = np.array([[preflop_class(c[0], c[1]), preflop_class(c[2], c[3])] for c in cards], np.int16)
    return {"cards": cards.astype(np.int8), "equity": eq, "winner": winner, "preflop": pre}


def thresholds(equity: np.ndarray, buckets: int) -> np.ndarray:
    """Per street cut points so each bucket gets an equal share of hands. Shape (3, buckets - 1)."""
    qs = np.linspace(0, 1, buckets + 1)[1:-1]
    return np.stack([np.quantile(equity[:, :, s].ravel(), qs) for s in range(3)]).astype(np.float32)


def bucketize(pool: dict, cuts: np.ndarray) -> np.ndarray:
    """(n, 2, 4) bucket per deal, player and street (preflop class, then flop/turn/river bucket)."""
    n = len(pool["winner"])
    out = np.empty((n, 2, 4), np.int16)
    out[:, :, 0] = pool["preflop"]
    for s in range(3):
        out[:, :, s + 1] = np.searchsorted(cuts[s], pool["equity"][:, :, s])
    return out


def save(path, pool: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pool)


def load(path) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def bucket_of(street: int, equity: float, cuts: np.ndarray) -> int:
    return int(np.searchsorted(cuts[street - 1], equity))


# ---- histogram buckets
#
# Johanson et al. (2013). Histograms are compared through their cumulative sums:
# for 1D histograms the earth mover's distance is the L1 distance between CDFs,
# L2 on CDFs is a close and much cheaper stand-in.

HIST_BINS = 20
HIST_RUNOUTS = 40
HIST_OPPS = 40


@njit(parallel=True, cache=True)
def _fill_hists(cards, runouts, opps, bins):
    n = cards.shape[0]
    out = np.zeros((n, 2, 2, bins), np.uint8)
    for i in prange(n):
        board = np.empty(5, np.int64)
        for j in range(5):
            board[j] = cards[i, 4 + j]
        for p in range(2):
            a, b = cards[i, 2 * p], cards[i, 2 * p + 1]
            for s in range(2):
                h = equity_histogram(a, b, board, 3 + s, runouts, opps, bins)
                for k in range(bins):
                    out[i, p, s, k] = h[k]
    return out


def build_histograms(cards, runouts=HIST_RUNOUTS, opps=HIST_OPPS, bins=HIST_BINS, chunk=100_000, log=print):
    """(n, 2 players, 2 streets [flop, turn], bins) histogram counts for every pool deal."""
    cards = cards.astype(np.int64)
    parts = []
    for start in range(0, len(cards), chunk):
        parts.append(_fill_hists(cards[start:start + chunk], runouts, opps, bins))
        if log:
            log(f"  histograms {min(start + chunk, len(cards)):,}/{len(cards):,}")
    return np.concatenate(parts)


def to_cdf(hists):
    h = hists.astype(np.float32)
    h /= np.maximum(h.sum(axis=-1, keepdims=True), 1)
    return np.cumsum(h, axis=-1)


def _nearest(x, centroids, chunk=200_000):
    out = np.empty(len(x), np.int64)
    c2 = (centroids ** 2).sum(1)
    for s in range(0, len(x), chunk):
        xs = x[s:s + chunk]
        d = c2[None, :] - 2 * xs @ centroids.T
        out[s:s + chunk] = d.argmin(1)
    return out


def kmeans(x, k, iters=40, sample=300_000, seed=0):
    """Plain k-means with k-means++ seeding, fitted on a random sample of rows."""
    rng = np.random.default_rng(seed)
    if len(x) > sample:
        x = x[rng.choice(len(x), sample, replace=False)]
    centroids = [x[rng.integers(len(x))]]
    d2 = ((x - centroids[0]) ** 2).sum(1)
    for _ in range(1, k):
        centroids.append(x[rng.choice(len(x), p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((x - centroids[-1]) ** 2).sum(1))
    c = np.array(centroids, np.float32)
    for _ in range(iters):
        lab = _nearest(x, c)
        new = np.array([x[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)], np.float32)
        shift = np.abs(new - c).max()
        c = new
        if shift < 1e-4:
            break
    # order clusters by mean strength so bucket numbers still read weak to strong
    mean_strength = (1 - c).sum(1)
    return c[np.argsort(mean_strength)]


def distribution_buckets(pool, hists, k=50, log=print):
    """(buckets, centroids, river_cuts). Flop and turn from histogram clusters, river
    from strength percentiles, preflop from the 169 classes. centroids is (2, k, bins)."""
    n = len(pool["winner"])
    out = np.empty((n, 2, 4), np.int16)
    out[:, :, 0] = pool["preflop"]
    cents = []
    for s in range(2):
        cdf = to_cdf(hists[:, :, s, :]).reshape(-1, hists.shape[-1])
        c = kmeans(cdf, k)
        cents.append(c)
        out[:, :, s + 1] = _nearest(cdf, c).reshape(n, 2)
        if log:
            log(f"  clustered {'flop' if s == 0 else 'turn'}")
    river_cuts = np.quantile(pool["equity"][:, :, 2].ravel(), np.linspace(0, 1, k + 1)[1:-1]).astype(np.float32)
    out[:, :, 3] = np.searchsorted(river_cuts, pool["equity"][:, :, 2])
    return out, np.stack(cents), river_cuts


def dist_bucket_of(hole, board, centroids, river_cuts):
    """Bucket for a live hand under the distribution-aware abstraction."""
    from .evaluator import river_strength
    b = np.zeros(5, np.int64)
    b[:len(board)] = board
    if len(board) == 5:
        return int(np.searchsorted(river_cuts, river_strength(hole[0], hole[1], b)))
    s = len(board) - 3
    h = equity_histogram(hole[0], hole[1], b, len(board), HIST_RUNOUTS, HIST_OPPS, HIST_BINS)
    return int(_nearest(to_cdf(h[None, :]), centroids[s])[0])
