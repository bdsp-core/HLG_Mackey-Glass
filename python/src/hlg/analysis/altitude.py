"""
Altitude-dependent Loop Gain (LG) analysis.

This module implements the extraction and histogram-based visualisation
pipeline for the altitude study, where the same patients were recorded
at multiple elevations (sea level, 5 000 ft, 8 000 ft, 13 000 ft).
The key scientific question is whether loop gain increases with altitude
-- which would confirm that hypoxia-driven chemoreflex sensitisation is
a major driver of periodic breathing at altitude.

Three functions are provided:

1. ``extract_EM_output`` -- reads the EM output CSVs for the altitude
   ("rt") dataset, groups per-segment LG estimates by patient and
   altitude, computes the self-similarity percentage from the SS HDF5
   output, and writes per-recording intermediate CSVs.

2. ``create_histogram_bars`` -- converts a per-recording LG distribution
   into normalised histogram bars suitable for the altitude grid figure.
   Invalid segments (those that failed the flow-reduction event count
   criterion) are placed in a separate grey "invalid" bar.

3. ``plot_histogram_bins`` -- renders one cell of the altitude x patient
   histogram grid, including the bar chart and a coloured marker at a
   user-specified percentile (typically the 95th).

Statistical utilities (``quadratic_model``, ``prediction_band``) are
imported from ``hlg.analysis.statistics`` for any downstream
curve-fitting that uses the altitude data.

Source: ``EM_output_to_Alitude_Relationship.py``
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hlg.analysis.statistics import (
    sort_dic_keys,
)
from hlg.core.events import find_events
from hlg.em.postprocessing import post_process_EM_output
from hlg.io.readers import load_SS_percentage
from hlg.ss.scoring import convert_ss_seg_scores_into_arrays


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_EM_output(
    input_files: list[str],
    interm_folder: str,
    version: str,
    hf5_folder: str,
    error_thresh: float = 1.8,
) -> None:
    """Extract per-segment EM estimates for the altitude dataset.

    Unlike the MGH group extraction (``hlg.analysis.group``), the
    altitude pipeline does *not* assign patients to SS severity groups.
    Instead, each recording is identified by its ``patient_tag`` (which
    encodes the patient ID and altitude condition, e.g.
    ``'P40-3-Sea level'``), and results are written to individual CSVs
    named ``{tag}-{SS%}.csv``.

    The self-similarity percentage (SS%) is appended to the filename so
    that downstream altitude figures can read it directly without
    reopening the HDF5 file.

    The extraction follows the same 8-step pattern as the group pipeline
    but with altitude-specific differences:

    * No arousal loading (the altitude dataset uses a different scoring
      format).
    * Per-patient rather than per-group output (each CSV is one
      recording at one altitude).
    * The SS percentage is loaded from the HDF5 file and the
      flow-reduction trace is used for segment validity.

    Args:
        input_files: EM output CSV paths for the altitude ("rt") dataset.
        interm_folder: Directory for intermediate CSV output.
        version: Analysis version string (typically ``'Altitude'``).
        hf5_folder: Directory containing the SS ``.hf5`` output files.
        error_thresh: RMSE threshold for segment exclusion (default 1.8).
    """
    # Per-group accumulators.
    LG_data: dict[str, np.ndarray] = {}
    G_data: dict[str, np.ndarray] = {}
    D_data: dict[str, np.ndarray] = {}
    GxD_data: dict[str, np.ndarray] = {}
    SS_data: dict[str, np.ndarray] = {}
    valid_data: dict[str, np.ndarray] = {}
    SS_percentages: list[float] = []

    for idx, input_file in enumerate(input_files):
        num = input_file.split("/Study ")[-1].split(".csv")[0]
        print(
            f"Extracting Study {num} ({idx + 1}/{len(input_files)}) ..",
            end="\r",
        )

        # Read EM output and identify the patient/altitude group.
        data = pd.read_csv(input_file)
        # The altitude dataset stores the altitude condition in the
        # patient_tag field with lower-case 'a' for "Altitude" which is
        # capitalised to match the canonical group naming convention.
        group: str = data.loc[0, "patient_tag"].replace("a", "A")

        # Convert sparse segment SS scores to dense array.
        data = convert_ss_seg_scores_into_arrays(data)

        # Post-process EM output (LG outlier smoothing).
        data = post_process_EM_output(data)

        # Extract header fields.
        hdr: dict[str, Any] = {"Study_num": f"Study {num}"}
        for col in ["patient_tag", "Fs", "original_Fs"]:
            hdr[col] = data.loc[0, col]
            data = data.drop(columns=col)

        # Load the self-similarity percentage and flow-reduction trace
        # from the SS HDF5 output.
        SS_pct, resp = load_SS_percentage(hf5_folder, hdr["patient_tag"])
        SS_percentages.append(SS_pct)

        # --- Collect per-segment parameters ---
        Errors: list[float] = []
        Vmaxs: list[float] = []
        LGs: list[float] = []
        Gs: list[float] = []
        Ds: list[float] = []
        Ls: list[float] = []
        SSs_seg: list[float] = []
        valid_seg: list[bool] = []

        for stage in ["nrem", "rem"]:
            starts = data[f"{stage}_starts"].dropna().values.astype(int)
            ends = data[f"{stage}_ends"].dropna().values.astype(int)

            for start, end in zip(starts, ends):
                loc = np.where(data[f"{stage}_starts"] == start)[0][0]
                Errors.append(round(data.loc[loc, "rmse_Vo"], 2))
                Ls.append(data.loc[loc, f"L_{stage}"])
                Vmaxs.append(round(data.loc[loc, "Vmax"], 2))
                LGs.append(data.loc[loc, f"LG_{stage}_corrected"])
                Gs.append(data.loc[loc, f"G_{stage}"])
                Ds.append(data.loc[loc, f"D_{stage}"])
                # Validity: >= 5 flow-reduction events within the segment.
                valid_seg.append(len(find_events(resp[start:end] > 0)) >= 5)
                SSs_seg.append(data.loc[start, "SS_score"])

        # Apply error threshold.
        inds = np.array(Errors) < error_thresh
        LG_data[group] = np.array(LGs)[inds]
        G_data[group] = np.array(Gs)[inds]
        D_data[group] = np.array(Ds)[inds]
        GxD_data[group] = np.array(Gs)[inds] * np.array(Ds)[inds]
        SS_data[group] = np.array(SSs_seg)[inds]
        valid_data[group] = np.array(valid_seg)[inds]

    # --- Sort and write per-recording CSVs ---
    sorted_dics = sort_dic_keys([LG_data, G_data, D_data, GxD_data, SS_data, valid_data])
    names = ["LG_data", "G_data", "D_data", "GxD_data", "SS_data", "valid_data"]

    for group, SS_pct in zip(LG_data.keys(), SS_percentages):
        df = pd.DataFrame([], dtype=float)
        for dic, name in zip(sorted_dics, names):
            df[name] = dic[group]
        os.makedirs(interm_folder, exist_ok=True)
        # Encode the SS percentage in the filename for easy access
        # by the downstream altitude figure script.
        df.to_csv(
            f"{interm_folder}/{group}-{SS_pct}.csv",
            header=df.columns,
            index=None,
            mode="w+",
        )


# ---------------------------------------------------------------------------
# Histogram construction
# ---------------------------------------------------------------------------


def create_histogram_bars(
    LG_all: np.ndarray,
    valids: np.ndarray,
    percentile: float,
) -> tuple[np.ndarray, float]:
    """Convert a per-recording LG distribution into normalised bars.

    The histogram has a fixed set of bins:
    - Bin 0 (grey): proportion of *invalid* segments (those that didn't
      meet the >= 5-event criterion).
    - Bins 1-N: LG values from 0.0 to ``max_edge`` in steps of 0.1.
    - Final bin: everything >= ``max_edge`` (overflow).

    Each bar height is the percentage of *all* segments (valid + invalid)
    that fall into that bin.  This ensures the bars always sum to 100 %.

    The ``percentile`` value (typically 0.95) is computed over the *full*
    LG array (with invalid segments set to 0) to provide a summary
    statistic that accounts for both the severity and prevalence of
    high LG.  Setting invalid segments to 0 rather than NaN is a
    conservative choice: it pulls the percentile down when many segments
    are invalid, reflecting genuine uncertainty in the LG estimate.

    Args:
        LG_all: Array of per-segment LG values (includes both valid and
            invalid segments).
        valids: Boolean array; ``True`` for segments that passed the
            validity criterion.
        percentile: Quantile to compute (e.g. 0.95 for the 95th
            percentile).

    Returns:
        A tuple ``(bins, pct)`` where:

        * **bins** -- 1-D array of normalised bar heights (percentages,
          summing to ~100 %).
        * **pct** -- the computed percentile value for the full array.
    """
    # Only valid segments contribute to the main histogram bins.
    LG: np.ndarray = LG_all[valids]

    # Upper edge of the last named bin.
    max_edge: float = 1.4
    edges: np.ndarray = np.arange(0, max_edge, 0.1)

    # +2 for the "invalid" bin at index 0 and the "overflow" bin at the end.
    bins: np.ndarray = np.zeros(len(edges) + 2)

    # Bin 0: count of invalid segments.
    bins[0] = sum(valids == False)

    # Named bins: [0.0, 0.1), [0.1, 0.2), ..., [1.3, 1.4).
    for e, edge in enumerate(edges[:-1]):
        lo, up = edge, edges[e + 1]
        count = sum(np.logical_and(LG >= lo, LG < up))
        bins[e + 1] = count

    # Overflow bin: everything >= max_edge.
    bins[-1] = sum(LG >= max_edge)

    # Normalise to percentages of total segments (valid + invalid).
    for i in range(len(bins)):
        bins[i] = bins[i] / len(valids) * 100

    # Compute the percentile over the full array with invalids set to 0.
    LG_for_pct = np.array(LG_all, copy=True)
    LG_for_pct[~valids] = 0
    pct: float = np.quantile(LG_for_pct, percentile)

    return bins, pct


# ---------------------------------------------------------------------------
# Histogram plotting
# ---------------------------------------------------------------------------


def plot_histogram_bins(
    bins: np.ndarray,
    pct: float,
    axes: np.ndarray,
    row: int,
    col: int,
    tag: str,
    height: str,
    c: str,
    fz: int,
) -> None:
    """Render one cell of the altitude x patient histogram grid.

    Each cell shows:
    - A bar chart of LG bin percentages (black bars for valid, grey for
      invalid).
    - A coloured inverted-triangle marker at the computed percentile
      value, with a numeric label below it.

    The grid layout uses ``row`` for patient index and ``col`` for
    altitude level.  Row 0 gets the altitude title and row 7 (the
    bottom row) gets x-axis tick labels; all other rows suppress x-ticks
    to keep the figure compact.

    Args:
        bins: Normalised bar heights from ``create_histogram_bars``.
        pct: Percentile value (plotted as a marker).
        axes: 2-D array of matplotlib ``Axes`` (the full grid).
        row: Row index in the grid (patient number).
        col: Column index in the grid (altitude level).
        tag: Patient label (e.g. ``'#1'``).
        height: Altitude label (e.g. ``'Sea level'``).
        c: Marker colour (one per patient for visual distinction).
        fz: Base font size.
    """
    ax: plt.Axes = axes[row, col]
    n_bins: int = len(bins)
    width: float = 0.1
    x_range: np.ndarray = np.arange(0, n_bins / 10, width) - 0.1

    # --- Bar chart ---
    # Main bars (valid segments): solid black.
    ax.bar(x_range[1:], bins[1:], color="k", width=0.85 * width, align="edge")
    # First bar (invalid segments): grey with slight transparency.
    ax.bar(
        x_range[0],
        bins[0],
        color="grey",
        width=0.85 * width,
        align="edge",
        alpha=0.75,
    )

    # --- Percentile marker ---
    # Inverted triangle at y=90 (near the top of the 0-100 % range).
    ax.plot(pct, 90, marker="v", ms=6, color=c, alpha=0.75)
    txt = round(pct, 1)
    if txt == 0:
        txt = 0
    ax.text(pct, 70, txt, ha="center", va="top", fontsize=fz - 3)

    # --- Axis layout ---
    if row != 7:
        # Suppress x-ticks for interior rows.
        ax.set_xticks([])
        if row == 0:
            # Top row: show altitude label as column title.
            ax.set_title(height, fontsize=fz)
    else:
        # Bottom row: show x-axis with LG tick marks.
        xx = [0, 0.5, 1, x_range[-1]]
        xran = [0, 0.5, 1.0, x_range[-1]]
        ax.set_xticks(xx)
        ax.set_xticklabels(xran, fontsize=fz - 3)
        ax.set_xlabel("LG", fontsize=fz)

    # y-axis: percentage scale 0-100 %.
    yy = [0, 50, 100]
    ax.set_yticks(yy)
    ax.set_yticklabels([], fontsize=fz - 3)

    if col == 0:
        # Left-most column: add percentage labels and patient tag.
        ax.text(-0.25, -5, "0%", ha="right", va="bottom", fontsize=10)
        ax.text(-0.25, 105, "100%", ha="right", va="top", fontsize=10)
        ax.set_ylabel(
            f"{tag}          ",
            rotation="horizontal",
            ha="right",
            fontsize=fz,
        )

    # Clean up spines.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 100)
