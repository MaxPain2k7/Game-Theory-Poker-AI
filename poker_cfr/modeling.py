"""Learning an opponent from realistic hand histories.

In a real game you see every action but only see the opponent's card when
the hand reaches showdown. That creates two problems:

  * Selection bias. If you only learn from hands that went to showdown, you
    never see a hand where the opponent folded, so a naive model thinks they
    almost never fold. Tight players get modelled as calling stations.
  * Missing data. A hand that ended in a fold still tells you a lot (what the
    opponent did at each step) even though you don't know their card.

The fix here is expectation maximization. Treat the hidden card as a latent
variable: given the current model, work out how likely each possible card
was given what the opponent did, add fractional counts for every card in
proportion, refit the model, repeat. Showdown hands go in with their known
card at full weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .games import new_game, LEDUC_VARIANTS
from .simulate import _deal
from .tree import GameTree


@dataclass
class HandRecord:
    opp_seat: int
    hero_card: int
    opp_card: int          # only to be used when showdown is True, or by the oracle
    board: int | None      # None if the hand ended before the board was dealt
    actions: list          # every decision slot in order, both players
    showdown: bool


def record_hands(tree: GameTree, hero_policy, opp_policy, hands: int, rng) -> list[HandRecord]:
    records = []
    for k in range(hands):
        opp_seat = k % 2
        cards = _deal(tree.game_name, rng)
        state = new_game(tree.game_name)
        draws = iter(cards)
        actions = []
        while not state.is_terminal():
            if state.is_chance():
                state = state.apply(next(draws))
                continue
            actor = opp_policy if state.player() == opp_seat else hero_policy
            a = actor.act(state, rng)
            actions.append(a)
            state = state.apply(a)
        records.append(HandRecord(
            opp_seat=opp_seat, hero_card=cards[1 - opp_seat], opp_card=cards[opp_seat],
            board=state.public, actions=actions, showdown=state.folded is None))
    return records


def _opp_decisions(game: str, rec: HandRecord, opp_card: int):
    """Replay a hand with a hypothesized opponent card. Yields (infoset key, action)
    for each opponent decision. The board is only needed if the hand got that far."""
    hands = [0, 0]
    hands[rec.opp_seat] = opp_card
    hands[1 - rec.opp_seat] = rec.hero_card
    state = new_game(game)
    state = state.apply(hands[0]).apply(hands[1])
    out = []
    for a in rec.actions:
        if state.is_chance():
            state = state.apply(rec.board)
        if state.player() == rec.opp_seat:
            out.append((state.infoset_key(), a))
        state = state.apply(a)
    return out


def _card_prior(game: str, rec: HandRecord) -> np.ndarray:
    cfg = LEDUC_VARIANTS[game]
    counts = np.full(cfg.ranks, float(cfg.suits))
    counts[rec.hero_card] -= 1
    if rec.board is not None:
        counts[rec.board] -= 1
    return counts / counts.sum()


def count_actions(tree: GameTree, records, mode: str, model: np.ndarray | None = None) -> np.ndarray:
    """Expected opponent action counts per infoset.

    mode "oracle":   every hand with the true card (what you'd get with full histories)
    mode "showdown": only hands that reached showdown (the naive, biased approach)
    mode "em":       showdown hands at full weight, other hands spread over the
                     possible cards using `model` to compute the posterior
    """
    index = {key: i for i, key in enumerate(tree.infoset_keys)}
    counts = np.zeros((tree.num_infosets, tree.num_actions))
    for rec in records:
        if mode == "oracle" or rec.showdown:
            for key, a in _opp_decisions(tree.game_name, rec, rec.opp_card):
                counts[index[key], a] += 1.0
            continue
        if mode == "showdown":
            continue
        prior = _card_prior(tree.game_name, rec)
        per_card, like = [], np.zeros_like(prior)
        for c in np.nonzero(prior)[0]:
            seq = _opp_decisions(tree.game_name, rec, int(c))
            per_card.append((c, seq))
            like[c] = prior[c] * np.prod([model[index[k], a] for k, a in seq]) if seq else prior[c]
        if like.sum() <= 0:
            continue
        post = like / like.sum()
        for c, seq in per_card:
            for key, a in seq:
                counts[index[key], a] += post[c]
    return counts


def fit_model(tree: GameTree, records, mode: str, prior: np.ndarray,
              strength: float = 2.0, em_iters: int = 12, smoothing: float = 0.05):
    """Returns (model, counts). Unseen spots fall back to `prior` (our own equilibrium)."""
    from .exploit import learn_model

    if mode != "em":
        counts = count_actions(tree, records, mode)
        return learn_model(tree, counts, prior, strength), counts
    # start from uniform play and keep a little uniform mass in the E step:
    # if the model ever says "never folds Q", EM could never assign a fold
    # to a Q again and would stay stuck at that zero
    uniform = tree.legal / tree.legal.sum(axis=1, keepdims=True)
    model = uniform
    for _ in range(em_iters):
        smoothed = (1 - smoothing) * model + smoothing * uniform
        counts = count_actions(tree, records, "em", smoothed)
        model = learn_model(tree, counts, prior, strength)
    return model, counts
