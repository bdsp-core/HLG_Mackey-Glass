#!/usr/bin/env python
"""
Altitude-dependent Loop Gain spaghetti plot with per-patient histograms.

Reads per-recording intermediate CSVs from the altitude ("rt") dataset
and produces a combined figure with:

  * **Top grid** (8 rows x 4 columns) -- per-patient, per-altitude LG
    histograms with normalised bin heights and a 95th-percentile marker.
  * **Bottom-left panel** -- spaghetti plot of the 95th-percentile LG
    across altitude levels for each patient, plus mean/median overlays.
  * **Bottom-right panel** -- spaghetti plot of the SS percentage
    (encoded in the CSV filename) across altitude levels.

The altitude study tests whether loop gain increases with elevation,
consistent with hypoxia-driven chemoreflex sensitisation being a major
driver of periodic breathing at high altitude.

Usage:
    python -m scripts.run_altitude_analysis
"""

import glob
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hlg.analysis.altitude import create_histogram_bars, plot_histogram_bins
from hlg.config import config

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    # ── Configuration ─────────────────────────────────────────────────
    version = "Altitude"
    dataset = "rt"
    smooth = "non-smooth"
    base_folder = os.path.join(config.interm_dir, smooth, f"{dataset}_{version}")

    # ── Figure layout ─────────────────────────────────────────────────
    # Top 9 rows: histogram grid.  Bottom 6 rows: two spaghetti panels.
    rows = 15
    columns = 4
    percentile = 0.95
    fig, axes = plt.subplots(rows, columns, figsize=(16, 14))

    # Bottom panels span the lower portion of the grid.
    ax1 = plt.subplot2grid((rows, columns), (9, 0), colspan=2, rowspan=6, fig=fig)
    ax2 = plt.subplot2grid((rows, columns), (9, 2), colspan=2, rowspan=6, fig=fig)
    axes1 = [ax1, ax2]

    colors = list(mcolors.TABLEAU_COLORS.keys())
    heights = ["Sea level", "5,000 ft", "8,000 ft", "13,000 ft"]
    _med_prop = {"color": "k", "linestyle": "solid", "linewidth": 1.5}
    fz = 14

    cnt = -1
    for p in range(1, 12):
        csvs = np.sort(glob.glob(os.path.join(base_folder, f"P40-{p}-*.csv")))
        if len(csvs) == 0:
            continue
        cnt += 1

        # option1: 95th-percentile LG per altitude level.
        # option2: SS percentage (parsed from filename) per altitude.
        option1, option2 = [], []

        for col, csv in enumerate(csvs):
            # The SS percentage is encoded in the filename as the last
            # hyphen-separated token before '.csv'.
            option2.append(float(csv.split("-")[-1].split(".csv")[0]))

            Alt_data = pd.read_csv(csv)
            if len(Alt_data) == 0:
                continue

            LG = Alt_data["LG_data"].values
            G = Alt_data["G_data"].values
            D = Alt_data["D_data"].values
            valid_data = Alt_data["valid_data"].values

            # Exclude segments where the EM optimiser hit parameter
            # boundaries -- same edge filter as the SS-relationship plot.
            no_lower_bound = np.logical_and(G == 0.1, D == 5)
            no_upper_bound = np.logical_and(G == 0.1, D == 50)
            no_edges = ~np.logical_or(no_lower_bound, no_upper_bound)
            valids = np.logical_and(no_edges, valid_data == True)

            # For the 95th percentile, invalid segments are treated as
            # LG == 0 (conservative: pulls the percentile down when
            # many segments are invalid).
            val = np.concatenate([LG[valids], np.zeros(sum(valid_data == False))])
            option1.append(np.nanquantile(val, percentile))

            color = colors[cnt]
            tag = f"#{cnt + 1}"
            height = heights[col]

            # Render one cell of the histogram grid.
            bins, pct = create_histogram_bars(LG, valids, percentile)
            plot_histogram_bins(bins, pct, axes, cnt, col, tag, height, color, fz)

        # ── Spaghetti lines ───────────────────────────────────────────
        tag = f"Patient {cnt + 1}"
        ax1.plot(range(1, 5), option1, c=color, marker="v", ms=6, alpha=0.75, lw=1, label=tag)
        ax2.plot(range(1, 5), option2, c=color, marker="o", ms=6, alpha=0.75, lw=1, label=tag)

        if cnt == 0:
            total_1, total_2 = option1, option2
        else:
            total_1 = np.vstack([total_1, option1])
            total_2 = np.vstack([total_2, option2])

    # ── Aggregate statistics (mean / median across patients) ──────────
    for _a, (ax, total) in enumerate(zip(axes1, [total_1, total_2])):
        ax.plot(range(1, 5), np.nanmean(total, 0), c="k", ls="dashed", lw=1.5, label="Mean")
        ax.plot(range(1, 5), np.nanmedian(total, 0), c="k", ls="dotted", lw=1.5, label="Median")
        ax.spines["top"].set_visible(False)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(heights, fontsize=fz)
        marge = 0.05 * np.nanmax(total)
        ax.set_ylim(np.nanmin(total) - marge, np.nanmax(total) + marge)

    # Left panel: 95th-percentile LG.
    ax1.set_ylabel("LG\n($95^{th}$ percentile)\n", fontsize=fz)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="y", labelsize=fz - 3)
    ax1.set_ylim(-0.1, 1.7)
    ax1.legend(frameon=False, loc=2, ncols=3, fontsize=fz - 2)

    # Right panel: SS percentage with y-axis on the right.
    ax2.set_ylabel("\nSS %", fontsize=fz)
    ax2.spines["left"].set_visible(False)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.tick_params(axis="y", labelsize=fz - 3)

    # Hide unused subplot cells below the histogram grid.
    for axs in axes[8:]:
        for ax in axs:
            ax.axis("off")

    out = os.path.join(FIGURES_DIR, "altitude", "altitude_spaghetti_histograms.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
