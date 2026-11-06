"""Saving and loading strategies, plus a few simple baseline opponents."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .tree import GameTree


def to_dict(tree: GameTree, sigma: np.ndarray) -> dict:
    out = {}
    for i, key in enumerate(tree.infoset_keys):
        out[key] = {tree.action_names[a]: round(float(sigma[i, a]), 6)
                    for a in range(tree.num_actions) if tree.legal[i, a]}
    return out


def from_dict(tree: GameTree, table: dict) -> np.ndarray:
    sigma = np.zeros((tree.num_infosets, tree.num_actions))
    for i, key in enumerate(tree.infoset_keys):
        row = table[key]
        for a, name in enumerate(tree.action_names):
            sigma[i, a] = row.get(name, 0.0)
    sigma /= sigma.sum(axis=1, keepdims=True)
    return sigma


def save(path, tree: GameTree, sigma: np.ndarray, **meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"game": tree.game_name, **meta, "strategy": to_dict(tree, sigma)}
    path.write_text(json.dumps(payload, indent=1))


def load(path):
    payload = json.loads(Path(path).read_text())
    tree = GameTree(payload["game"])
    return tree, from_dict(tree, payload["strategy"]), payload


def _kuhn_facing_bet(key: str) -> bool:
    return key.split(":", 1)[1].endswith("b")


BASELINES = ("random", "calling-station", "maniac")
HEURISTICS = ("nit", "lag")


def _strength(tree: GameTree, key: str) -> float:
    """Rough hand strength in [0, 1] from an infoset key like "QK:rc/c"."""
    from .games import LEDUC_VARIANTS
    ranks = LEDUC_VARIANTS[tree.game_name].ranks_str
    cards = key.split(":", 1)[0]
    mine = ranks.index(cards[0]) / (len(ranks) - 1)
    if len(cards) == 1:
        return mine
    return 1.0 if cards[0] == cards[1] else 0.75 * mine


def heuristic(tree: GameTree, name: str) -> np.ndarray:
    """Two human-like Leduc opponents that play by rules of thumb.

    nit: tight and passive. Bets only strong hands, folds a lot to pressure.
    lag: loose and aggressive. Bluffs often, raises a lot, rarely folds.
    Both are randomized, so you can't read them perfectly from a few hands.
    """
    if not tree.game_name.startswith("leduc"):
        raise ValueError("heuristic opponents are written for Leduc variants")
    sigma = np.zeros((tree.num_infosets, tree.num_actions))
    for i, key in enumerate(tree.infoset_keys):
        s = _strength(tree, key)
        facing, can_raise = tree.legal[i, 0], tree.legal[i, 2]
        fold = call = raise_ = 0.0
        if name == "nit":
            if not facing:
                raise_ = 0.8 if s >= 0.85 else 0.05
            elif s >= 0.85:
                raise_ = 0.5
            elif s >= 0.5:
                call, fold = 0.8, 0.2
            else:
                call, fold = 0.15, 0.85
        elif name == "lag":
            if not facing:
                raise_ = 0.9 if s >= 0.6 else 0.55
            elif s >= 0.6:
                raise_ = 0.7
            elif s >= 0.3:
                call, raise_, fold = 0.6, 0.2, 0.2
            else:
                fold, call, raise_ = 0.5, 0.3, 0.2
        else:
            raise ValueError(f"unknown heuristic {name}, pick from {HEURISTICS}")
        if not can_raise:
            raise_ = 0.0
        call = max(0.0, 1.0 - fold - raise_) if facing else 1.0 - raise_
        sigma[i] = [fold, call, raise_]
    return sigma


def opponent(tree: GameTree, name: str) -> np.ndarray:
    return heuristic(tree, name) if name in HEURISTICS else baseline(tree, name)


def baseline(tree: GameTree, name: str) -> np.ndarray:
    """random: uniform over legal actions.
    calling-station: never bets, never folds, just checks and calls.
    maniac: bets and raises whenever it can, calls otherwise."""
    A = tree.num_actions
    sigma = np.zeros((tree.num_infosets, A))
    for i, key in enumerate(tree.infoset_keys):
        legal = np.nonzero(tree.legal[i])[0]
        if name == "random":
            sigma[i, legal] = 1.0 / len(legal)
        elif name == "calling-station":
            if tree.game_name == "kuhn":
                # slot 1 means call when facing a bet, slot 0 means check otherwise
                sigma[i, 1 if _kuhn_facing_bet(key) else 0] = 1.0
            else:
                sigma[i, 1] = 1.0
        elif name == "maniac":
            sigma[i, legal.max()] = 1.0
        else:
            raise ValueError(f"unknown baseline {name}, pick from {BASELINES}")
    return sigma


class TablePolicy:
    """Looks up a strategy row by infoset key so it can drive live game states."""

    def __init__(self, tree: GameTree, sigma: np.ndarray):
        self.rows = {key: sigma[i] for i, key in enumerate(tree.infoset_keys)}

    def probs(self, state) -> np.ndarray:
        return self.rows[state.infoset_key()]

    def act(self, state, rng: np.random.Generator) -> int:
        p = self.probs(state)
        return int(rng.choice(len(p), p=p))
