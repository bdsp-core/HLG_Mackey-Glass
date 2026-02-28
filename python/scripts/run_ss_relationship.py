#!/usr/bin/env python
"""
Visualise the Self-Similarity (SS) vs Loop Gain (LG) relationship.

Loads per-segment intermediate CSVs produced by the SS-range extraction
pipeline, filters out invalid edge-parameter estimations, and generates
a scatter plot with:

  1. Seaborn 2nd-order polynomial regression line with 95 % CI band.
  2. scipy ``curve_fit`` quadratic model for r² computation.
  3. ``uncertainties``-based confidence intervals (unused in final plot
     but available for diagnostics).
  4. 95 % prediction band via Student-t formulation.
  5. Empirical 5th-95th percentile sliding-window range.

The resulting figure demonstrates the concave-upward relationship
between SS score and LG: higher self-similarity is associated with
disproportionately higher loop gain, reflecting ventilatory control
instability.

Usage:
    python -m scripts.run_ss_relationship
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import uncertainties as unc
import uncertainties.unumpy as unp
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit


from hlg.analysis.group import select_highest_LG_block
from hlg.analysis.statistics import prediction_band, quadratic_model
from hlg.config import config

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    # ── Configuration ─────────────────────────────────────────────────
    dataset = "mgh"
    version = "SS_range"
    smooth = "non-smooth"
    base_folder = os.path.join(config.interm_dir, smooth, f"{dataset}_{version}")

    # ── Load and concatenate per-segment CSV files ────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(11, 10))
    xs, ys, zs = [], [], []
    groups, total_seg = [], 0

    for _p, csvs in enumerate(np.sort(glob.glob(os.path.join(base_folder, "seg*.csv")))):
        group = csvs.split("/seg_")[-1].split(".csv")[0]
        groups.append(group)
        SS_data = pd.read_csv(csvs)
        if len(SS_data) == 0:
            continue
        total_seg += len(SS_data)

        SS = SS_data["SS"].values
        LG = SS_data["LG"].values
        G = SS_data["G"].values
        D = SS_data["D"].values
        valid_data = SS_data["VV"].values

        # Exclude segments where the EM algorithm hit the parameter
        # boundary (G==0.1 at the lower or upper delay limit), which
        # indicates the optimiser did not converge to a physiologically
        # meaningful solution.
        no_lower_bound = np.logical_and(G == 0.1, D == 5)
        no_upper_bound = np.logical_and(G == 0.1, D == 50)
        no_edges = ~np.logical_or(no_lower_bound, no_upper_bound)
        valid = np.logical_and(no_edges, valid_data == True)
        inds = np.logical_and(np.isfinite(SS), valid)

        xs = np.concatenate([xs, SS[inds]])
        ys = np.concatenate([ys, LG[inds]])
        zs = np.concatenate([zs, G[inds]])

    # Sort all arrays by SS (x-axis) for monotonic curve fitting.
    xs, ys, zs = xs[np.argsort(xs)], ys[np.argsort(xs)], zs[np.argsort(xs)]

    # Secondary edge filter: remove points where SS > 0 but G is at the
    # lower boundary (0.1), or SS == 0 but G is not at the boundary.
    # These represent inconsistent estimation states.
    inds = ~np.logical_or(np.logical_and(xs > 0, zs == 0.1), np.logical_and(xs == 0, zs != 0.1))
    x, y, _z, n = xs[inds], ys[inds], zs[inds], len(inds)

    # Evaluation grid for smooth fitted curves.
    xx = np.linspace(0, 1, 100)

    # ── Scatter plot with seaborn 2nd-order polynomial ────────────────
    scatter_dic = {"color": "tab:blue", "alpha": 0.2}
    poly_dic = {"color": "black", "linestyle": "solid", "label": "$2^{nd}$ order polynomial [95% CI]"}
    sns.regplot(x=x, y=y, scatter=True, order=2, ax=ax, scatter_kws=scatter_dic, line_kws=poly_dic)

    # ── Explicit curve_fit for r² and prediction bands ────────────────
    popt, pcov = curve_fit(quadratic_model, x, y)
    a, b, c = popt
    # Coefficient of determination (r²) from residual sum of squares.
    r2 = round(np.sqrt(1.0 - (sum((y - quadratic_model(x, a, b, c)) ** 2) / ((n - 1.0) * np.var(y, ddof=1)))), 2)

    # Propagate parameter covariance into fitted-curve uncertainty via
    # the ``uncertainties`` package (correlated error propagation).
    a, b, c = unc.correlated_values(popt, pcov)
    py = a * xx * xx + b * xx + c
    _nom = unp.nominal_values(py)
    _std = unp.std_devs(py)

    # 95 % prediction band via the Student-t formulation.
    lpb, upb = prediction_band(xx, x, y, popt, quadratic_model, conf=0.95)

    # ── Empirical 5th-95th percentile sliding window ──────────────────
    # A non-parametric alternative to the prediction band: compute the
    # 5th and 95th percentiles of LG within overlapping windows along
    # the SS axis.
    xi, q5, q95, win = [], [], [], 0.3
    for i in np.arange(0.325, 1 + win * 2, win):
        lo, up = i - win / 2, i + win / 2
        y_vals = y[np.logical_and(x >= lo, x < up)]
        if len(y_vals) == 0:
            continue
        xi.append(i)
        q5.append(np.quantile(y_vals, 0.05))
        q95.append(np.quantile(y_vals, 0.95))

    ax.plot(xi, q5, "k--", label="$5^{th}-95^{th}$ percentile prediction range")
    ax.plot(xi, q95, "k--")

    # ── Axis formatting ───────────────────────────────────────────────
    fz = 16
    ax.set_xlabel("\nSS", fontsize=fz)
    ax.set_ylabel("LG\n", fontsize=fz)

    xmin, xmax = 0, 1
    ymin, ymax = 0, 2.5
    margin = 0.025
    ax.set_xlim([xmin - margin * xmax, xmax + margin * xmax])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_ylim([ymin - margin * ymax, ymax + margin * ymax])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Build legend with an extra handle for r².
    handles, _ = ax.get_legend_handles_labels()
    r2_dic = {"color": "none", "linestyle": "", "label": f"$r^{{2}}$: {r2}"}
    r2_handle = Line2D([0], [0], **r2_dic)
    handles += [r2_handle]
    ax.legend(handles=handles, loc=0, frameon=False, fontsize=fz - 1, title_fontsize=fz)

    out = os.path.join(FIGURES_DIR, "ss_relationship", "SS_vs_LG_scatter.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def nrem_vs_rem_boxplot():
    """Generate a NREM vs REM LG boxplot from per-segment intermediate CSVs.

    Compares the estimated loop gain distribution between NREM and REM
    sleep segments. The 'St' column in the seg_*.csv files encodes the
    sleep stage as 'nrem' or 'rem'.
    """
    dataset = "mgh"
    version = "SS_range"
    smooth = "non-smooth"
    base_folder = os.path.join(config.interm_dir, smooth, f"{dataset}_{version}")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    med_prop = {"color": "none", "linewidth": 0}
    men_prop = {"color": "r", "linestyle": "dashed", "linewidth": 2}
    boxdic = {
        "showmeans": True,
        "meanline": True,
        "showfliers": False,
        "widths": 0.2,
        "medianprops": med_prop,
        "meanprops": men_prop,
    }

    stages = ["nrem", "rem"]
    for pos, stage in enumerate(stages):
        xs, ys, zs = [], [], []

        for _p, csvs in enumerate(np.sort(glob.glob(os.path.join(base_folder, "seg*.csv")))):
            SS_data = pd.read_csv(csvs)
            if len(SS_data) == 0:
                continue

            SS = SS_data["SS"].values
            LG = SS_data["LG"].values
            G = SS_data["G"].values
            D = SS_data["D"].values
            valid_data = SS_data["VV"].values
            Stage_data = SS_data["St"].values

            # Same edge-parameter filter as the scatter plot
            no_lower_bound = np.logical_and(G == 0.1, D == 5)
            no_upper_bound = np.logical_and(G == 0.1, D == 50)
            no_edges = ~np.logical_or(no_lower_bound, no_upper_bound)
            valid = np.logical_and(no_edges, valid_data == True)
            inds = np.logical_and(np.isfinite(SS), valid)
            stage_ind = np.logical_and(inds, Stage_data == stage)

            xs = np.concatenate([xs, SS[stage_ind]])
            ys = np.concatenate([ys, LG[stage_ind]])
            zs = np.concatenate([zs, G[stage_ind]])

        # Secondary edge filter
        xs, ys, zs = xs[np.argsort(xs)], ys[np.argsort(xs)], zs[np.argsort(xs)]
        inds = ~np.logical_or(
            np.logical_and(xs > 0, zs == 0.1),
            np.logical_and(xs == 0, zs != 0.1),
        )
        y = ys[inds]

        ax.boxplot(y, positions=[(pos + 1) * 0.3], **boxdic)

    # Layout — match the cohort boxplot style (no spines, custom y-axis)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(bottom=False, left=False)

    ax.set_xlim([0.1, 0.8])
    ax.set_xticks([0.3, 0.6])
    ax.set_xticklabels(["NREM", "REM"], fontsize=12)

    # Custom y-axis with hand-drawn line and ticks
    fz = 11
    y_ticks = [round(v, 1) for v in np.arange(0, 2.1, 0.3)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(t) for t in y_ticks], fontsize=fz)
    ax.set_ylabel("LG\n", fontsize=fz + 1)
    ax.set_ylim([-0.1, 2.0])

    x_pos = 0.15
    ax.plot([x_pos, x_pos], [-0.05, 1.95], "k", lw=0.8, clip_on=False)
    for t in y_ticks:
        ax.plot([x_pos - 0.01, x_pos], [t, t], "k", lw=0.8, clip_on=False)

    out = os.path.join(FIGURES_DIR, "ss_relationship", "NREM_vs_REM_LG_boxplot.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def swimmer_plot():
    """Generate the LG hypnogram swimmer plot (Figure 4).

    Multi-cohort version: 20 patients from each of 5 clinical cohorts,
    with group clamp labels on the right panel. Left panel shows stacked
    LG traces; right panel shows colour-coded LG bar graphs.

    This exactly reproduces the logic from the original
    EM_output_to_Group_Analysis.py __main__ block.
    """

    # Bottom-to-top order: first in list = bottom row, last = top row.
    # Desired top-to-bottom: SS OSA, high CAI, NREM OSA, REM OSA
    # So iterate: REM_OSA (bottom) → NREM_OSA → high_CAI → SS_OSA (top)
    versions = ["REM_OSA", "NREM_OSA", "high_CAI", "SS_OSA"]
    folder_names = {
        "REM_OSA": "mgh_REM_OSA",
        "NREM_OSA": "mgh_NREM_OSA",
        "high_CAI": "mgh_high_CAI",
        "SS_OSA": "mgh_SS_OSA",
        "Heart_Failure": "redeker_Heart_Failure",
    }
    smooth = "non-smooth"

    fig = plt.figure(figsize=(14, 12))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    axes = [ax1, ax2]

    height = 2
    factor = 20 * 10 * 200
    thresh1, thresh2 = 0.7, 1.0
    select = 20
    hr = 8.25
    group_shift = 0

    for v, version in enumerate(versions):
        folder = folder_names[version]
        interm_folder = os.path.join(config.interm_dir, smooth, folder)
        hypno_folder = os.path.join(interm_folder, "hypnograms")

        # Number of patients in this cohort
        len_input_files = 65 if "Heart" in version else 100

        shift = 0
        cnt = 0
        for i in range(len_input_files - select + 1, len_input_files + 1, 1):
            cnt += 1

            # Try both single-space and double-space naming conventions
            csv_path = os.path.join(hypno_folder, f"Study {i}.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join(hypno_folder, f"Study  {i}.csv")
            if not os.path.exists(csv_path):
                continue

            data = np.squeeze(pd.read_csv(csv_path).values)

            # Fill to hr hours; if recording is longer, pick the block
            # with the highest mean LG.
            block = int(hr * factor)
            LG_array = np.zeros(block) * np.nan
            if len(data) <= block:
                LG_array[: len(data)] = data
            else:
                LG_array = select_highest_LG_block(data, block)

            x = np.arange(len(LG_array)) / factor
            y = LG_array

            # ── Left panel: stacked traces ────────────────────────────
            shade = np.array(y)
            shade[np.isnan(shade)] = 0
            red = np.array(y)
            red[red < thresh2] = np.nan
            y_offset = cnt + shift + group_shift + (select * v) + 7.5 * v
            ax1.plot(x, shade + y_offset, color="lightgray", lw=1)
            ax1.plot(x, y + y_offset, color="tab:gray", lw=1.2)
            ax1.plot(x, red + y_offset, color="tab:red", lw=1.5)

            # ── Right panel: colour-coded bars ────────────────────────
            bottom_bar_i = (cnt - 0.85) * height + select * v * height + v * (height + 1)
            top_bar_i = (cnt - 0.15) * height + select * v * height + v * (height + 1)
            ax2.fill_between(x, bottom_bar_i, top_bar_i, where=np.isnan(y), facecolor="lightgray")
            ax2.fill_between(x, bottom_bar_i, top_bar_i, where=y < thresh1, facecolor="tab:gray")
            medium = np.logical_and(y >= thresh1, y < thresh2)
            ax2.fill_between(x, bottom_bar_i, top_bar_i, where=medium, facecolor="tab:orange")
            ax2.fill_between(x, bottom_bar_i, top_bar_i, where=y >= thresh2, facecolor="tab:red")

            maxi = np.nanmax(LG_array)
            shift += maxi

            # Track first/last row positions for group clamps
            if cnt == 1:
                bottom_clamp = bottom_bar_i - 1
            if cnt == select:
                top_line = cnt + shift + group_shift + (select * v) + 7.5 * v + 3.75
                top_clamp = top_bar_i + 1

        group_shift += shift

        # Group clamp brackets on the right panel
        clamp_dic = {"color": "black", "linewidth": 1, "linestyle": "solid"}
        ax2.plot([-0.2, -0.1], [bottom_clamp] * 2, **clamp_dic)
        ax2.plot([-0.2, -0.1], [top_clamp] * 2, **clamp_dic)
        ax2.plot([-0.2] * 2, [bottom_clamp, top_clamp], **clamp_dic)
        tag = version.replace("_", " ")
        y_label = (bottom_clamp + top_clamp) / 2
        ax2.text(-0.3, y_label, tag, ha="right", va="center", fontsize=11, rotation=90)

    # ── Layout: remove all spines, add custom x-axes ──────────────────
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Time\n(hr)", fontsize=11)
        ax.set_yticks([])
        ax.set_yticklabels([])

    ax1.set_ylim([-12, top_line - 2.5])
    ax1.plot([0, hr], [-12] * 2, "k", lw=2.5)
    ax2.set_ylim([-8, top_clamp + 1])
    ax2.plot([0, hr], [-8] * 2, "k", lw=2.5)

    # Legend
    fz = 11
    start = np.array([0.5, 0.75])
    tags = ["Wake", "Low LG", "Elevated LG", "High LG"]
    colors = ["lightgray", "tab:gray", "tab:orange", "tab:red"]
    text_dic = {"ha": "left", "va": "center", "fontsize": fz}
    for i, (tag, c) in enumerate(zip(tags, colors)):
        # Left panel legend (trace lines)
        if i in [0, 1]:
            lx = start + (i + 0.5) * 2
            ax1.plot(lx, [-6] * 2, color=c)
            ax1.text(lx[1] + 0.1, -6, tag, **text_dic)
        elif i == 3:
            lx = start + (i - 0.5) * 2
            ax1.plot(lx, [-6] * 2, color=c)
            ax1.text(lx[1] + 0.1, -6, tag, **text_dic)
        # Right panel legend (filled rectangles)
        rx = start + i * 2
        ax2.fill_between(rx, [-5] * 2, [-3] * 2, facecolor=c)
        ax2.text(rx[1] + 0.1, -4, tag, **text_dic)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "ss_relationship", "LG_swimmer_plot.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
    nrem_vs_rem_boxplot()
    swimmer_plot()
