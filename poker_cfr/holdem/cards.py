"""Cards are ints 0..51: rank = card // 4 (0 is a deuce, 12 an ace), suit = card % 4."""

from __future__ import annotations

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"


def parse(text: str) -> int:
    """'Ah' -> 50"""
    return RANK_CHARS.index(text[0].upper()) * 4 + SUIT_CHARS.index(text[1].lower())


def parse_many(text) -> list[int]:
    if isinstance(text, (list, tuple)):
        return [parse(t) for t in text]
    text = text.replace(" ", "")
    return [parse(text[i:i + 2]) for i in range(0, len(text), 2)]


def name(card: int) -> str:
    return RANK_CHARS[card // 4] + SUIT_CHARS[card % 4]


def names(cards) -> str:
    return " ".join(name(c) for c in cards)


def preflop_class(c1: int, c2: int) -> int:
    """One of 169 strategically different starting hands.
    0-12 pairs, 13-90 suited, 91-168 offsuit."""
    r1, r2 = c1 // 4, c2 // 4
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo:
        return hi
    tri = hi * (hi - 1) // 2 + lo
    return 13 + tri if c1 % 4 == c2 % 4 else 91 + tri


def preflop_class_name(idx: int) -> str:
    if idx < 13:
        return RANK_CHARS[idx] * 2
    suited = idx < 91
    tri = idx - (13 if suited else 91)
    hi = 1
    while (hi + 1) * hi // 2 <= tri:
        hi += 1
    lo = tri - hi * (hi - 1) // 2
    return RANK_CHARS[hi] + RANK_CHARS[lo] + ("s" if suited else "o")
