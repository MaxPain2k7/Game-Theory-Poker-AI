"""Seven card hand evaluator and equity calculations, compiled with numba.

eval7 returns an int where a bigger number is a better hand:
category * 16^5 + up to five ranks in base 16 (most important first).
Categories: 0 high card, 1 pair, 2 two pair, 3 trips, 4 straight,
5 flush, 6 full house, 7 quads, 8 straight flush.
"""

from __future__ import annotations

import numpy as np
from numba import njit

C = 16 ** 5


@njit(cache=True)
def _straight_top(mask):
    """Top rank of the best straight in a 13 bit rank mask, or -1."""
    for top in range(12, 3, -1):
        need = 0x1F << (top - 4)
        if mask & need == need:
            return top
    # wheel: A 2 3 4 5
    if mask & 0x100F == 0x100F:
        return 3
    return -1


@njit(cache=True)
def _top_bits(mask, n):
    """Encode the n highest set ranks of mask in base 16, highest first."""
    out = 0
    taken = 0
    for r in range(12, -1, -1):
        if taken == n:
            break
        if mask >> r & 1:
            out = out * 16 + r
            taken += 1
    return out


@njit(cache=True, inline="always")
def _add(c, counts, s0, s1, s2, s3, rank_mask):
    r = c >> 2
    counts += 1 << (3 * r)
    bit = 1 << r
    su = c & 3
    if su == 0:
        s0 |= bit
    elif su == 1:
        s1 |= bit
    elif su == 2:
        s2 |= bit
    else:
        s3 |= bit
    return counts, s0, s1, s2, s3, rank_mask | bit


@njit(cache=True)
def _popcount13(m):
    n = 0
    while m:
        m &= m - 1
        n += 1
    return n


@njit(cache=True)
def _score(counts, s0, s1, s2, s3, rank_mask):
    """Core ranking from packed rank counts (3 bits per rank) and per-suit rank masks.
    No arrays are allocated, which keeps it fast inside hot loops."""
    for m in (s0, s1, s2, s3):
        if _popcount13(m) >= 5:
            sf = _straight_top(m)
            if sf >= 0:
                return 8 * C + sf
            return 5 * C + _top_bits(m, 5)

    quad = -1
    trips1 = -1
    trips2 = -1
    pair1 = -1
    pair2 = -1
    for r in range(12, -1, -1):
        k = (counts >> (3 * r)) & 7
        if k == 4:
            quad = r
        elif k == 3:
            if trips1 < 0:
                trips1 = r
            elif trips2 < 0:
                trips2 = r
        elif k == 2:
            if pair1 < 0:
                pair1 = r
            elif pair2 < 0:
                pair2 = r

    if quad >= 0:
        return 7 * C + quad * 16 + _top_bits(rank_mask & ~(1 << quad), 1)
    if trips1 >= 0 and (trips2 >= 0 or pair1 >= 0):
        p = trips2 if trips2 > pair1 else pair1
        return 6 * C + trips1 * 16 + p
    st = _straight_top(rank_mask)
    if st >= 0:
        return 4 * C + st
    if trips1 >= 0:
        return 3 * C + trips1 * 256 + _top_bits(rank_mask & ~(1 << trips1), 2)
    if pair2 >= 0:
        rest = rank_mask & ~(1 << pair1) & ~(1 << pair2)
        return 2 * C + pair1 * 256 + pair2 * 16 + _top_bits(rest, 1)
    if pair1 >= 0:
        return 1 * C + pair1 * 4096 + _top_bits(rank_mask & ~(1 << pair1), 3)
    return _top_bits(rank_mask, 5)


@njit(cache=True)
def eval_cards(cards, n):
    """Evaluate the first n (5 to 7) cards of an int array."""
    counts = 0
    s0 = s1 = s2 = s3 = 0
    rank_mask = 0
    for i in range(n):
        counts, s0, s1, s2, s3, rank_mask = _add(cards[i], counts, s0, s1, s2, s3, rank_mask)
    return _score(counts, s0, s1, s2, s3, rank_mask)


@njit(cache=True)
def eval7(a, b, board):
    """Two hole cards plus a five card board."""
    counts = 0
    s0 = s1 = s2 = s3 = 0
    rank_mask = 0
    counts, s0, s1, s2, s3, rank_mask = _add(a, counts, s0, s1, s2, s3, rank_mask)
    counts, s0, s1, s2, s3, rank_mask = _add(b, counts, s0, s1, s2, s3, rank_mask)
    for i in range(5):
        counts, s0, s1, s2, s3, rank_mask = _add(board[i], counts, s0, s1, s2, s3, rank_mask)
    return _score(counts, s0, s1, s2, s3, rank_mask)


@njit(cache=True)
def _remaining(used, n_used):
    taken = np.zeros(52, np.bool_)
    for i in range(n_used):
        taken[used[i]] = True
    out = np.empty(52 - n_used, np.int64)
    k = 0
    for c in range(52):
        if not taken[c]:
            out[k] = c
            k += 1
    return out


@njit(cache=True)
def equity_sampled(h0, h1, board, n_board, samples):
    """Chance that hole cards (h0, h1) beat a random hand by the river (ties count half),
    estimated by dealing `samples` random opponent hands and runouts."""
    used = np.empty(2 + n_board, np.int64)
    used[0] = h0
    used[1] = h1
    for i in range(n_board):
        used[2 + i] = board[i]
    deck = _remaining(used, 2 + n_board)
    m = deck.shape[0]
    need = 2 + (5 - n_board)
    full = np.empty(5, np.int64)
    for i in range(n_board):
        full[i] = board[i]
    score = 0.0
    for _ in range(samples):
        # partial Fisher-Yates for the cards we need
        for j in range(need):
            k = j + np.random.randint(m - j)
            deck[j], deck[k] = deck[k], deck[j]
        for i in range(n_board, 5):
            full[i] = deck[2 + i - n_board]
        mine = eval7(h0, h1, full)
        theirs = eval7(deck[0], deck[1], full)
        if mine > theirs:
            score += 1.0
        elif mine == theirs:
            score += 0.5
    return score / samples


@njit(cache=True)
def river_strength(h0, h1, board):
    """Exact share of opponent hands beaten on a complete board (ties half)."""
    used = np.empty(7, np.int64)
    used[0] = h0
    used[1] = h1
    for i in range(5):
        used[2 + i] = board[i]
    deck = _remaining(used, 7)
    mine = eval7(h0, h1, board)
    score = 0.0
    n = 0
    m = deck.shape[0]
    for i in range(m):
        for j in range(i + 1, m):
            theirs = eval7(deck[i], deck[j], board)
            if mine > theirs:
                score += 1.0
            elif mine == theirs:
                score += 0.5
            n += 1
    return score / n


@njit(cache=True)
def equity_histogram(h0, h1, board, n_board, runouts, opps, bins):
    """Distribution of river hand strength for hole cards (h0, h1) on a 3 or 4 card board.

    For each of `runouts` random completions of the board, the river strength is
    estimated against `opps` random opponent hands. The result goes into one
    of `bins` equal width bins on [0, 1]. Returns integer counts.

    Two hands with the same average equity can have very different shapes: a
    flush draw is mostly near 0 or near 1, a medium pair sits in the middle.
    """
    used = np.empty(2 + n_board, np.int64)
    used[0] = h0
    used[1] = h1
    for i in range(n_board):
        used[2 + i] = board[i]
    deck = _remaining(used, 2 + n_board)
    m = deck.shape[0]
    full = np.empty(5, np.int64)
    for i in range(n_board):
        full[i] = board[i]
    hist = np.zeros(bins, np.int64)
    need = 5 - n_board
    for _ in range(runouts):
        for j in range(need):
            k = j + np.random.randint(m - j)
            deck[j], deck[k] = deck[k], deck[j]
        for i in range(n_board, 5):
            full[i] = deck[i - n_board]
        mine = eval7(h0, h1, full)
        score = 0.0
        rest = m - need
        for _o in range(opps):
            a = need + np.random.randint(rest)
            b = need + np.random.randint(rest - 1)
            if b >= a:
                b += 1
            theirs = eval7(deck[a], deck[b], full)
            if mine > theirs:
                score += 1.0
            elif mine == theirs:
                score += 0.5
        s = score / opps
        idx = int(s * bins)
        if idx >= bins:
            idx = bins - 1
        hist[idx] += 1
    return hist
