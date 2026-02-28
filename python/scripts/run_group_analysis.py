#!/usr/bin/env python
"""
Run cohort-level group analysis with boxplots.

Compares LG, gamma, and tau distributions across clinical cohorts
(REM_OSA, NREM_OSA, high_CAI, SS_OSA, Heart_Failure) with
Mann-Whitney significance tests.

Usage:
    python -m scripts.run_group_analysis
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hlg.analysis.statistics import add_statistical_significance
from hlg.config import config

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    versions = ["REM_OSA", "NREM_OSA", "high_CAI", "SS_OSA", "Heart_Failure"]
    # Map version names to the actual folder names in data/interm_Results/non-smooth/
    folder_names = {
        "REM_OSA": "mgh_REM_OSA",
        "NREM_OSA": "mgh_NREM_OSA",
        "high_CAI": "mgh_high_CAI",
        "SS_OSA": "mgh_SS_OSA",
        "Heart_Failure": "redeker_Heart_Failure",
    }
    smooth = "non-smooth"

    fig = plt.figure(figsize=(5.5, 10))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    axes = [ax1, ax2, ax3]

    med_prop = {"color": "r", "linestyle": "dashed", "linewidth": 1.5}
    boxdic = {
        "showmeans": False,
        "meanline": False,
        "showfliers": False,
        "widths": 0.8,
        "medianprops": med_prop,
    }

    lens = []
    ref_data = []
    for v, version in enumerate(versions):
        folder = folder_names[version]
        interm_folder = os.path.join(config.interm_dir, smooth, folder)

        interm_result = pd.read_csv(os.path.join(interm_folder, "all_segments.csv"))
        LG_data = interm_result["LG_data"]
        G_data = interm_result["G_data"]
        D_data = interm_result["D_data"]
        valid_data = interm_result["valid_data"]
        lens.append(len(interm_result))

        for i, (ax, vals) in enumerate(zip(axes, [LG_data, G_data, D_data])):
            no_lower_bound = np.logical_and(G_data == 0.1, D_data == 5)
            no_upper_bound = np.logical_and(G_data == 0.1, D_data == 50)
            no_edges = ~np.logical_or(no_lower_bound, no_upper_bound)
            inds = np.logical_and(no_edges, valid_data == True)
            ax.boxplot(vals[inds], positions=[v + 1], **boxdic)

            if i != 2:
                if v == 0:
                    ref_data.append(vals[inds])
                else:
                    add_statistical_significance(vals[inds], ref_data[i], v + 1, ax, i)

    # ── Strip all spines ──────────────────────────────────────────────
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(bottom=False, left=False)
    for ax in [ax1, ax2]:
        ax.set_xticks([])

    # ── Custom y-axes (matching reference figure exactly) ─────────────
    fz = 11

    # LG panel: ticks at 0.0, 0.3, 0.6, 0.9, 1.2, 1.5
    ax1.set_ylim([-0.5, 1.75])
    lg_ticks = [round(v, 1) for v in np.arange(0, 1.6, 0.3)]
    ax1.set_yticks(lg_ticks)
    ax1.set_yticklabels([str(t) for t in lg_ticks], fontsize=fz)
    ax1.set_ylabel("LG\n", fontsize=fz + 1)

    # γ panel: ticks at 0.1, 0.3, 0.5, 0.7, 0.9, 1.1
    ax2.set_ylim([-0.2, 1.22])
    g_ticks = [round(v, 1) for v in np.arange(0.1, 1.2, 0.2)]
    ax2.set_yticks(g_ticks)
    ax2.set_yticklabels([str(t) for t in g_ticks], fontsize=fz)
    ax2.set_ylabel("$\\gamma$\n", fontsize=fz + 1)

    # τ panel: ticks at 10, 20, 30, 40, 50
    ax3.set_ylim([5, 55])
    t_ticks = list(range(10, 55, 10))
    ax3.set_yticks(t_ticks)
    ax3.set_yticklabels([str(t) for t in t_ticks], fontsize=fz)
    ax3.set_ylabel("$\\tau$\n", fontsize=fz + 1)

    # Draw custom y-axis lines alongside the ticks, extending slightly
    # beyond the outermost ticks for a cleaner look.
    ylims = [(-0.225, 1.725), (-0.05, 1.25), (5, 55)]
    for ax, ticks, (ylo, yhi) in zip(axes, [lg_ticks, g_ticks, t_ticks], ylims):
        x_pos = 0.4
        ax.plot([x_pos, x_pos], [ylo, yhi], "k", lw=0.8, clip_on=False)
        for t in ticks:
            ax.plot([x_pos - 0.05, x_pos], [t, t], "k", lw=0.8, clip_on=False)

    # ── X-axis labels (bottom panel only) ─────────────────────────────
    ax3.set_xticks(range(1, len(versions) + 1))
    ax3.set_xticklabels(
        [v.replace("_", " ") + f"\n(N={n})" for v, n in zip(versions, lens)],
        fontsize=fz - 2,
    )

    # ── Legend (hardcoded coordinates in data space of the LG panel) ──
    ax1.plot([0.6, 0.8], [1.6, 1.6], "r--", lw=1.5)
    ax1.text(0.9, 1.6, "median", ha="left", va="center", fontsize=fz)
    ax1.text(0.715, 1.4, "***", ha="center", va="center", fontsize=fz)
    ax1.text(0.9, 1.4, "p < 0.001", ha="left", va="center", fontsize=fz)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "cohort_boxplots", "LG_gamma_tau_boxplots.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
