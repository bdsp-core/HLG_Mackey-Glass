"""
Stable Self-Similarity (SS) analysis visualisation.

This module provides two plotting functions for the stable-SS analysis
pipeline, which detects and characterises sustained episodes of
self-similar (periodic) breathing during sleep:

1. ``plot_SS`` -- a multi-row full-night figure similar to
   ``visualization.full_night.plot_full_night``, but enhanced with
   additional annotation layers:
   - The SS convolution score trace (red when above threshold, black
     when below).
   - Chains of detected breathing oscillations (horizontal bars).
   - Stable SS regions identified by change-point detection.
   - CPAP split-point indicator (if applicable).

2. ``create_length_histogram`` -- a stacked bar chart showing the
   distribution of segment lengths for either "oscillation chains"
   or "stable SS" regions, stratified by patient SS severity group.
   This answers the question: "How long do periodic breathing episodes
   last, and does this differ between patients with mild vs. severe
   periodic breathing?"

Both functions are purely visual -- data loading and SS detection are
handled upstream by the SS segmentation pipeline (``hlg.ss``).

Source: ``Stable_SS_analysis.py`` -> ``plot_SS``, ``create_length_histogram``
"""

from __future__ import annotations

from itertools import groupby
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hlg.core.events import find_events


# ---------------------------------------------------------------------------
# Full-night SS overview plot
# ---------------------------------------------------------------------------


def plot_SS(
    data: pd.DataFrame,
    hdr: dict[str, Any],
    out_path: str = "",
) -> None:
    """Create a multi-row figure showing SS analysis results.

    The recording is divided into 10 equal-length rows (each spanning
    approximately ``total_duration / 10``).  Each row shows:

    - The abdominal RIP signal, colour-coded by sleep stage (NREM=black,
      REM=blue, wake=red).
    - The SS convolution score trace (a rolling measure of how
      "self-similar" the breathing pattern is), plotted below the signal.
      The trace is red when above the SS threshold and black when below.
    - Horizontal reference lines at SS = 0 and SS = 1 for visual
      calibration, plus a dotted line at the threshold.
    - Scored respiratory event bars (technician labels in one colour
      scheme, algorithm labels in another).
    - Chains of breathing oscillations: contiguous periods where the SS
      score exceeds the threshold for >= 2 minutes.
    - Tagged breath markers (``*`` at individual breaths that exceed the
      SS convolution threshold).
    - Stable SS regions: longer-duration episodes identified by
      change-point detection (Pelt algorithm on the median-smoothed
      SS trace).

    If the recording includes a CPAP titration (indicated by
    ``hdr['cpap_start']``), a vertical red dashed line marks the
    transition point.

    Args:
        data: DataFrame with all required columns:
            ``abd``, ``sleep_stages``, ``apnea``, ``flow_reductions``,
            ``Osc_chain``, ``stable_SS``, ``tagged``, ``ss_conv_score``,
            ``SS_trace``.
        hdr: Recording header dict (must contain ``newFs``,
            ``SS_threshold``, ``cpap_start``).
        out_path: If non-empty, the figure is saved to this path at
            900 DPI.  Otherwise it is created but not saved.

    Side Effects:
        Creates and closes a matplotlib figure.  If ``out_path`` is
        provided, saves the figure as a PNG.
    """
    _Fs: int = hdr["newFs"]
    SS_thresh: float = hdr["SS_threshold"]

    # --- Extract signal arrays ---
    signal: np.ndarray = data.abd.values
    sleep_stages: np.ndarray = data.sleep_stages.values
    y_tech: np.ndarray = data.apnea.values
    y_algo: np.ndarray = data.flow_reductions.values
    osc_chain: np.ndarray = data.Osc_chain.values
    stable_SS: np.ndarray = data.stable_SS.values
    tagged_breaths: np.ndarray = data.tagged.values
    ss_conv_score: np.ndarray = data.ss_conv_score.values
    ss_trace: np.ndarray = data.SS_trace.values

    # --- Setup figure ---
    fig: plt.Figure = plt.figure(figsize=(12, 8))
    ax: plt.Axes = fig.add_subplot(111)
    row_height: int = 30

    # Split recording into 10 equal-length rows, reversed so that the
    # earliest data appears at the top.
    nrow: int = 10
    row_ids: list[np.ndarray] = np.array_split(np.arange(len(signal)), nrow)
    row_ids.reverse()

    # --- Build stage-separated signal arrays ---
    # NREM signal.
    sleep = np.array(signal)
    sleep[np.isnan(sleep_stages)] = np.nan
    sleep[sleep_stages == 5] = np.nan
    # Wake signal.
    wake = np.zeros(signal.shape)
    wake[np.isnan(sleep_stages)] += signal[np.isnan(sleep_stages)]
    wake[sleep_stages == 5] += signal[sleep_stages == 5]
    wake[wake == 0] = np.nan
    # REM signal.
    rem = np.array(signal)
    rem[sleep_stages != 4] = np.nan

    # SS trace split at threshold for colour coding.
    SS = np.array(ss_trace)
    SS_none = np.array(ss_trace)
    SS_none[SS >= SS_thresh] = np.nan

    # =====================================================================
    # PLOT SIGNALS
    # =====================================================================
    for ri in range(nrow):
        a: float = 1

        # Respiratory signal traces.
        ax.plot(sleep[row_ids[ri]] + ri * row_height, c="k", lw=0.3, alpha=a)
        ax.plot(wake[row_ids[ri]] + ri * row_height, c="r", lw=0.3, alpha=a)
        ax.plot(rem[row_ids[ri]] + ri * row_height, c="b", lw=0.3, alpha=a)

        # SS convolution score trace (below the respiratory signal).
        offset: float = -10
        factor: int = 5
        # Red trace where SS >= threshold.
        ax.plot(
            SS[row_ids[ri]] * factor + ri * row_height + offset,
            c="r",
            lw=1,
        )
        # Black trace where SS < threshold.
        ax.plot(
            SS_none[row_ids[ri]] * factor + ri * row_height + offset,
            c="k",
            lw=1,
        )

        # Reference lines for the SS trace.
        ref = np.ones(len(row_ids[ri]))
        # SS = 0 baseline.
        ax.plot(
            ref * 0 + ri * row_height + offset,
            c="k",
            lw=0.3,
            alpha=0.2,
        )
        # SS = 1 ceiling.
        ax.plot(
            ref * 0 + ri * row_height + offset + factor,
            c="k",
            lw=0.3,
            alpha=0.2,
        )
        # SS threshold (dotted).
        ax.plot(
            ref * SS_thresh * factor + ri * row_height + offset,
            c="k",
            linestyle="dotted",
            lw=0.3,
        )

        # CPAP split-point marker.
        if hdr["cpap_start"] in row_ids[ri]:
            cpap_loc = np.where(row_ids[ri] == hdr["cpap_start"])[0]
            min_ = -20 + ri * row_height
            max_ = 20 + ri * row_height
            ax.plot(
                [cpap_loc, cpap_loc],
                [min_, max_],
                c="r",
                linestyle="dashed",
                zorder=10,
                lw=4,
            )

    # =====================================================================
    # PLOT LABELS -- multi-layer annotation
    # =====================================================================
    for yi in range(6):
        if yi == 0:
            # Technician-scored respiratory events.
            labels = y_tech
            label_color = [None, "k", "b", "b", "k", "k", None, "b"]
        elif yi == 1:
            # Algorithm-detected respiratory events.
            labels = y_algo
            label_color = [None, "b", "g", "c", "m", "r", None, "g"]
        if yi == 2:
            # Chains of breathing oscillations.
            # 1 = NREM oscillation chain (black), 2 = REM (blue).
            labels = osc_chain
            label_color = [None, "k", "b"]
        if yi == 3:
            # Tagged breaths (individual oscillation peaks).
            labels = tagged_breaths
            label_color = [None, "k"]
        if yi == 4:
            # Stable SS regions (change-point detected).
            # 1 = NREM stable SS (black), 2 = REM stable SS (blue).
            labels = stable_SS
            label_color = [None, "k", "b"]

        for ri in range(nrow):
            loc_counter: int = 0
            for i, j in groupby(labels[row_ids[ri]]):
                len_j: int = len(list(j))

                if np.isfinite(i) and label_color[int(i)] is not None:
                    if yi < 1:
                        # Scored respiratory events.
                        _sub = 0 if int(i) < 7 else 2
                        minus = 3 if "fc811d4b" not in out_path else 4
                        ax.plot(
                            [loc_counter, loc_counter + len_j],
                            [ri * row_height - minus * (2**yi)] * 2,
                            c=label_color[int(i)],
                            lw=1,
                        )
                        if int(i) == 7:
                            ax.plot(
                                [loc_counter, loc_counter + len_j],
                                [ri * row_height - minus * (2**1)] * 2,
                                c="m",
                                lw=1,
                            )
                    if yi == 2:
                        # Oscillation chain bars.
                        ax.plot(
                            [loc_counter, loc_counter + len_j],
                            [ri * row_height + 8] * 2,
                            c=label_color[int(i)],
                            lw=3,
                            alpha=1,
                        )
                    if yi == 3:
                        # Tagged breath markers.
                        c_score = np.round(
                            ss_conv_score[row_ids[ri]][loc_counter],
                            2,
                        )
                        if np.isfinite(c_score):
                            ax.text(
                                loc_counter,
                                ri * row_height + 10,
                                "*",
                                c="k",
                                ha="center",
                            )
                            ax.text(
                                loc_counter,
                                ri * row_height + 15,
                                str(c_score),
                                ha="center",
                                fontsize=3,
                            )
                    if yi == 4:
                        # Stable SS region bars.
                        ax.plot(
                            [loc_counter, loc_counter + len_j],
                            [ri * row_height + 5] * 2,
                            c=label_color[int(i)],
                            lw=3,
                            alpha=1,
                        )

                loc_counter += len_j

    # --- Final layout ---
    ax.set_xlim([0, max(len(x) for x in row_ids)])
    ax.axis("off")

    plt.tight_layout()

    if len(out_path) > 0:
        plt.savefig(out_path, dpi=900)
    plt.close()


# ---------------------------------------------------------------------------
# Segment length histogram
# ---------------------------------------------------------------------------


def create_length_histogram(
    sim_df: pd.DataFrame,
    result: list[tuple[pd.DataFrame | None, dict[str, Any] | None]],
    version: str = "Osc_chain",
) -> None:
    """Plot a stacked histogram of SS segment lengths by patient group.

    For each patient recording, the lengths (in minutes) of all
    detected episodes of the chosen ``version`` type are computed.
    These are then aggregated by patient SS severity group and displayed
    as a stacked bar chart, with each group's contribution shown in a
    different colour (blue -> red gradient from low to high SS burden).

    The median segment length per group is included in the legend to
    quantify the typical episode duration.  The overall median across
    all groups is printed to the console.

    The histogram uses 0.5-minute bins for oscillation chains and
    1-minute bins for stable SS regions, reflecting the different
    temporal scales of these two segmentation methods.

    Args:
        sim_df: Metadata DataFrame containing at least an ``'SS group'``
            column that maps each patient to an SS severity group.
        result: List of ``(data, hdr)`` tuples, one per recording.
            ``data`` is the DataFrame with ``Osc_chain`` and/or
            ``stable_SS`` columns; ``hdr`` must contain ``'newFs'``,
            ``'patient_tag'``, and ``'group'``.  ``(None, None)`` entries
            are skipped (recordings that failed loading).
        version: Which segmentation to histogram.  Either
            ``'Osc_chain'`` (chains of breathing oscillations) or
            ``'stable_SS'`` (change-point-detected stable SS regions).

    Side Effects:
        Creates a matplotlib figure (not saved -- the caller is
        responsible for ``plt.savefig()``).
    """
    # --- Initialise per-group length dictionary ---
    # Keys are the unique SS groups from the metadata, sorted
    # alphabetically.  Values start as None and are filled with
    # concatenated length arrays.
    len_dic: dict[str | None, np.ndarray | None] = dict.fromkeys(np.sort(sim_df["SS group"].dropna().unique()))

    # --- Collect segment lengths from each recording ---
    for data, hdr in result:
        if data is None or hdr is None:
            continue

        # Compute the length (in minutes) of each detected episode.
        # find_events returns (start, end) tuples; duration in samples
        # is (end - start), converted to minutes via newFs.
        lens: np.ndarray = np.array([(end - st) / hdr["newFs"] / 60 for st, end in find_events(data[version] == 1)])

        # Insert into the group dictionary.
        group: str = hdr["group"]
        if len_dic[group] is None:
            len_dic[group] = lens
        else:
            len_dic[group] = np.concatenate([len_dic[group], lens])

    # --- Build the stacked bar chart ---
    fig: plt.Figure = plt.figure(figsize=(9, 6))
    ax: plt.Axes = fig.add_subplot(111)

    # Colour gradient from cool (low SS) to warm (high SS).
    colors: list[str] = ["blue", "lightskyblue", "khaki", "darkorange", "red"]

    # Bin width depends on the segmentation type: finer bins for
    # oscillation chains (which tend to be shorter) than for stable SS.
    bar_width: float = 0.5 if version == "Osc_chain" else 1
    lw: float = 0.1
    xs: np.ndarray = np.arange(0, 50 + bar_width, bar_width)
    bottom: np.ndarray = np.zeros(len(xs))

    total_lens: list[float] = []
    for i, group_key in enumerate(len_dic.keys()):
        lens = len_dic[group_key]
        if lens is None:
            continue

        total_lens += lens.tolist()

        # Count segments falling into each bin.
        ys: np.ndarray = np.array([sum(np.logical_and(lens >= x, lens < x + bar_width)) for x in xs])

        # --- Build legend label ---
        # First group: "< upper_bound", last group: "> lower_bound",
        # middle groups: "lower - upper".
        if i == 0:
            lab = "< " + group_key.split("-")[-1]
        elif i == len(len_dic) - 1:
            lab = "> " + group_key.split("-")[0][-3:]
        else:
            lab = group_key.replace("SS ", "").replace("-", " - ")
        label = f"{lab}   [{round(np.median(lens), 1)} min]"

        # Stacked bars.
        ax.bar(
            xs,
            ys,
            color=colors[i],
            bottom=bottom,
            ec="k",
            width=bar_width,
            lw=lw,
            label=label,
        )

        # Accumulate bottom for stacking.
        bottom = ys if i == 0 else bottom + ys

    # Print overall median to console.
    median: float = round(np.median(total_lens), 2)
    print(f"{version}:\nMedian length across all segments: {median}\n")

    # --- Layout ---
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim([-1, 51])
    ax.set_ylim([0, 1.05 * max(bottom)])
    ax.set_xlabel("Window length\n(minutes)", fontsize=11)
    ax.set_ylabel("# of segments", fontsize=11)

    title: str = "Patient bins based on ratio of expressed HLG [median segment length]"
    ax.legend(
        loc=0,
        facecolor="k",
        title=title,
        ncol=2,
        handletextpad=0.8,
        frameon=False,
        fontsize=10,
        title_fontsize=11,
    )
