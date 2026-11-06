"""Learning opponents from realistic, showdown-only hand histories.

    python modeling_demo.py

1. Selection bias: fold frequencies estimated from full histories, from
   showdown hands only (naive) and with EM over the hidden cards.
2. What that does to profit: best response to each kind of model, scored
   against the real opponent.
3. Robust responses on a learned model: Restricted Nash Response vs
   Data-Biased Response frontiers, scored against the real opponent.

Writes results/modeling.png and results/modeling.md. Takes a few minutes.
"""

import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from poker_cfr import GameTree, CFRSolver, exploitability, heads_up_ev
from poker_cfr import policy
from poker_cfr.exploit import best_response_to, restricted_nash_response, data_biased_response
from poker_cfr.modeling import record_hands, fit_model

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
MODES = {"oracle": "full histories", "showdown": "showdown hands only", "em": "EM over hidden cards"}
SPOTS = {"J:r/": "J facing a bet, first round", "Q:r/": "Q facing a bet, first round",
         "J:cr/": "J facing a bet after checking", "Q:cr/": "Q facing a bet after checking"}


def main():
    tree = GameTree("leduc")
    solver = CFRSolver(tree, "cfr+")
    solver.iterate(3000)
    nash = solver.average_strategy()
    hero = policy.TablePolicy(tree, nash)
    index = {k: i for i, k in enumerate(tree.infoset_keys)}
    md = []

    # 1 + 2: selection bias and its cost
    t = time.time()
    md += ["## Fold frequency at a few spots (nit, 3,000 hands)", "",
           "| spot | true | " + " | ".join(MODES.values()) + " |", "|---|---|---|---|---|"]
    true_nit = policy.heuristic(tree, "nit")
    recs = record_hands(tree, hero, policy.TablePolicy(tree, true_nit), 3000, np.random.default_rng(11))
    showdown_share = np.mean([r.showdown for r in recs])
    models = {m: fit_model(tree, recs, m, nash)[0] for m in MODES}
    for key, label in SPOTS.items():
        i = index[key]
        md.append(f"| {label} | {true_nit[i, 0]:.2f} | " + " | ".join(f"{models[m][i, 0]:.2f}" for m in MODES) + " |")
    md += ["", f"Only {showdown_share:.0%} of hands reached showdown.", ""]

    sizes = [300, 1000, 3000, 10000]
    md += ["## Best response to the learned model, EV against the real opponent (mean of 3 seeds)", ""]
    for opp in ("nit", "lag"):
        true = policy.heuristic(tree, opp)
        md += [f"### {opp} (Nash gets {heads_up_ev(tree, nash, true):+.3f}, ceiling "
               f"{heads_up_ev(tree, best_response_to(tree, true), true):+.3f})", "",
               "| hands | " + " | ".join(MODES.values()) + " |", "|---|---|---|---|"]
        for n in sizes:
            evs = {m: [] for m in MODES}
            for seed in (1, 2, 3):
                r = record_hands(tree, hero, policy.TablePolicy(tree, true), n, np.random.default_rng(seed))
                for m in MODES:
                    model = fit_model(tree, r, m, nash)[0]
                    evs[m].append(heads_up_ev(tree, best_response_to(tree, model), true))
            md.append(f"| {n:,} | " + " | ".join(f"{np.mean(evs[m]):+.3f}" for m in MODES) + " |")
        md.append("")
    print(f"selection bias tables: {time.time() - t:.0f}s")

    # 3: robust responses on a learned (EM) model
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE)
    grid = [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
    n_hands = 1000
    md += [f"## Robust responses on a model learned from {n_hands:,} showdown-only hands", "",
           "risk = worst case loss per hand, edge = EV per hand against the real opponent", ""]
    for ax, opp in zip(axes, ("nit", "lag")):
        t = time.time()
        true = policy.heuristic(tree, opp)
        recs = record_hands(tree, hero, policy.TablePolicy(tree, true), n_hands, np.random.default_rng(5))
        em_model, em_counts = fit_model(tree, recs, "em", nash)
        naive_model, _ = fit_model(tree, recs, "showdown", nash)
        curves = {
            "RNR, true strategy (ceiling)": [restricted_nash_response(tree, true, p, 1000) for p in grid],
            "RNR, EM model": [restricted_nash_response(tree, em_model, p, 1000) for p in grid],
            "DBR, EM model": [data_biased_response(tree, em_model, em_counts, 20.0, p, 1000) for p in grid],
            "RNR, showdown-only model": [restricted_nash_response(tree, naive_model, p, 1000) for p in grid],
        }
        colors = dict(zip(curves, (INK_MUTED, BLUE, ORANGE, YELLOW)))
        md += [f"### vs {opp}", "", "| method | " + " | ".join(f"p={p}" for p in grid) + " |",
               "|---|" + "---|" * len(grid)]
        ax.set_facecolor(SURFACE)
        for name, strats in curves.items():
            pts = np.array([(exploitability(tree, s), heads_up_ev(tree, s, true)) for s in strats])
            ls = "--" if "ceiling" in name else "-"
            marker = "s" if "showdown" in name else "o"  # yellow sits next to orange, so shape tells them apart too
            ax.plot(pts[:, 0] + 1e-4, pts[:, 1], ls, color=colors[name], lw=2, marker=marker, ms=3.5, label=name)
            md.append(f"| {name} | " + " | ".join(f"{e:+.3f} @ {r:.3f}" for r, e in pts) + " |")
        md.append("")
        ax.set_xscale("log")
        ax.set_title(f"vs {'tight passive (nit)' if opp == 'nit' else 'loose aggressive (lag)'}",
                     loc="left", fontsize=11, color=INK)
        ax.set_xlabel("risk: worst case loss (chips/hand, log)")
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        print(f"frontier vs {opp}: {time.time() - t:.0f}s")
    axes[0].set_ylabel("edge vs the real opponent (chips/hand)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle(f"Responses built from {n_hands:,} showdown-only hands, scored against the real opponent",
                 x=0.06, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig("results/modeling.png", dpi=160, facecolor=SURFACE)
    Path("results/modeling.md").write_text("\n".join(md) + "\n")
    print("wrote results/modeling.png and results/modeling.md")


if __name__ == "__main__":
    main()
