"""Figure for the twenty-year experiment. python -m resumegen.plot_divergence"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# validated categorical slots 1-3, light mode
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#b8b7b0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e2",
    "grid.linewidth": 0.7, "axes.axisbelow": True, "legend.frameon": False,
})


def main():
    df = pd.read_csv(os.path.join(OUT, "divergence_paths.csv"))
    df["is_winner"] = df.who == df.winner

    # mean log gap (winner - loser) by arm and period
    g = (df.pivot_table(index=["arm", "pair_id", "t"], columns="is_winner",
                        values="salary")
           .rename(columns={True: "w", False: "l"}).reset_index())
    g["loggap"] = np.log(g.w / g.l)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.07, right=0.98, wspace=0.24)

    # ---- A: gap over time -------------------------------------------------
    ax = axes[0]
    arms = [("baseline", BLUE, "-", "as modelled"),
            ("no anchoring", ORANGE, "-", "salary history not carried"),
            ("all three off", AQUA, (0, (4, 2)), "no memory of the past")]
    for arm, colour, style, label in arms:
        sub = g[g.arm == arm]
        if not len(sub):
            continue
        m = sub.groupby("t")["loggap"].mean()
        se = sub.groupby("t")["loggap"].sem()
        # convert log points to a percentage difference
        ax.fill_between(m.index, np.expm1(m - se) * 100, np.expm1(m + se) * 100,
                        color=colour, alpha=0.15, lw=0)
        ax.plot(m.index, np.expm1(m) * 100, color=colour, lw=2, ls=style,
                label=label, zorder=3)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("years after the hiring decision")
    ax.set_ylabel("winner's pay advantage (%)")
    ax.set_title("A. One arbitrary decision, twenty years\nmean gap between matched twins, ±1 SE",
                 loc="left", fontsize=10, color=INK, pad=8)
    ax.legend(loc="lower left", fontsize=8)

    # ---- B: where each pair ends up --------------------------------------
    ax = axes[1]
    final_log = g[(g.arm == "baseline") & (g.t == g.t.max())]["loggap"]
    null_log = g[(g.arm == "all three off") & (g.t == g.t.max())]["loggap"]
    # histogram shows each pair's own percentage gap; the summary line is the
    # geometric mean, expm1(mean log), which is what divergence_summary.csv
    # reports. The arithmetic mean of the ratios runs higher (~16%) because a
    # few pairs diverge a long way -- worth not quoting the two interchangeably.
    final = np.expm1(final_log) * 100
    nullf = np.expm1(null_log) * 100
    geo_mean = np.expm1(final_log.mean()) * 100
    bins = np.linspace(-60, 90, 34)
    ax.hist(nullf, bins=bins, color=AQUA, alpha=0.45, label="no memory")
    ax.hist(final, bins=bins, color=BLUE, alpha=0.60, label="as modelled")
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(geo_mean, color=BLUE, lw=2)
    ax.annotate(f"typical winner\n+{geo_mean:.1f}%", xy=(geo_mean + 3, ax.get_ylim()[1] * 0.86),
                fontsize=8, color=INK2, ha="left")
    ax.set_xlabel("winner's pay advantage after 20 years (%)")
    ax.set_ylabel("pairs")
    ax.set_title(f"B. Spread across {len(final)} pairs\nwinner ahead in "
                 f"{(final > 0).mean():.0%} of them, vs {(nullf > 0).mean():.0%} with no memory",
                 loc="left", fontsize=10, color=INK, pad=8)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Two equally qualified candidates, one job, one arbitrary choice",
                 x=0.07, y=0.965, ha="left", fontsize=12.5, color=INK)
    fig.text(0.07, 0.895,
             "Matched pairs from PSID-grounded personas: same seniority, experience, "
             "education and pay. Only the hiring decision differs.",
             ha="left", fontsize=8.5, color=INK2)

    path = os.path.join(OUT, "divergence.png")
    fig.savefig(path, dpi=170)
    print(f"wrote {path}")
    print(f"  baseline: geometric mean gap {geo_mean:+.1f}%  "
          f"(winner ahead in {(final > 0).mean():.0%})")
    print(f"            arithmetic mean of ratios {final.mean():+.1f}% "
          f"-- skewed by a few big winners, do not quote as 'the' effect")
    print(f"  null:     geometric mean gap "
          f"{np.expm1(null_log.mean()) * 100:+.1f}%  "
          f"(winner ahead in {(nullf > 0).mean():.0%})")


if __name__ == "__main__":
    main()
