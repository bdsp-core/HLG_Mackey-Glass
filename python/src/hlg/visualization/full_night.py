"""
Full-night overview and LG-hook figures.

This module produces a compressed, multi-row overview figure of an
entire PSG recording.  The recording is split into 1-hour blocks, each
displayed as a horizontal row.  Rows are stacked bottom-to-top so that
the earliest part of the night appears at the top -- matching the
conventional hypnogram reading direction.

Three signal layers are plotted per row:

1. **Respiratory signal** -- the abdominal RIP trace, colour-coded by
   sleep stage: black for NREM, blue for REM, red for wake.
2. **Respiratory event labels** -- horizontal bars for scored apneas
   and hypopneas, plus markers for detected breathing oscillations.
3. **LG hooks** -- bracket annotations showing the span of each EM
   segment, annotated with the estimated loop gain value.  Segments
   with fewer than 4 flow-reduction events are drawn with reduced
   opacity to indicate lower confidence.

Supporting functions:

* ``plot_full_night`` -- the main orchestrator.
* ``add_LG_hooks`` -- draws the LG bracket annotations.
* ``find_row_location`` -- maps a global sample index to its (x, row)
  position in the multi-row layout.

Source: ``EM_output_to_Figures.py`` -> ``plot_full_night``,
``add_LG_hooks``, ``find_row_location``
"""

from __future__ import annotations

from itertools import groupby
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hlg.core.events import find_events
from hlg.io.readers import load_sim_output

from hlg.em.postprocessing import match_EM_with_SS_output, remove_excessive_wake
from hlg.reporting import create_report


def plot_full_night(
    EM_data: pd.DataFrame,
    EM_hdr: dict[str, Any],
    figure_path: str,
    hf5_folder: str,
    csv_file: str,
    dataset: str = "mgh",
    plot_all_tagged: bool = False,
) -> None:
    """Create a full-night multi-row overview figure.

    The recording is divided into 1-hour blocks (at the resampled
    10 Hz rate).  Each block occupies one horizontal row in the figure.
    The rows are plotted from bottom (earliest) to top (latest) so that
    reading top-to-bottom follows chronological order.

    The figure includes:
    - Colour-coded respiratory signal (NREM=black, REM=blue, wake=red).
    - Respiratory event bars (apneas, hypopneas) below the signal.
    - Self-similarity oscillation markers (``o`` / ``'``).
    - LG estimation hooks (see ``add_LG_hooks``).
    - A legend at the bottom with sleep-state line types, event types,
      and a scale bar.

    Before plotting, excessive wake at the beginning and end of the
    recording is trimmed (> 1 hour of trailing wake is cut to keep the
    figure compact), and a summary report is generated for the header.

    Args:
        EM_data: Full-night EM output DataFrame (post-processed, with
            SS scores expanded).
        EM_hdr: Recording header dict.
        figure_path: Output path (without extension; ``.pdf`` is
            appended).
        hf5_folder: Directory containing the SS ``.hf5`` files.
        csv_file: Path to the metadata CSV.
        dataset: Dataset identifier (default ``'mgh'``).
        plot_all_tagged: If ``True``, show all tagged breaths regardless
            of their SS convolution score.  If ``False`` (default), only
            breaths above the SS threshold are shown.

    Side Effects:
        Saves a high-DPI PDF to ``figure_path + '.pdf'``.
    """
    # --- Match with SS pipeline output ---
    sim_path, _ = match_EM_with_SS_output(EM_data, dataset, csv_file)
    path: str = hf5_folder + sim_path + ".hf5"

    # Load the full SS pipeline output for this recording.
    SS_data, hdr = load_sim_output(path)
    assert len(SS_data) == len(EM_data), "matching SS output does not match EM data"

    # Trim excessive wake at the start/end of the recording.
    EM_data, SS_data = remove_excessive_wake(EM_data, SS_data, hdr["newFs"])

    # Generate a summary report (AHI, RDI, etc.) for the figure header.
    SS_data = SS_data.rename(
        columns={"self similarity": "T_sim", "sleep_stages": "stage"},
    )
    _, summary_report = create_report(SS_data, hdr)

    # --- Extract signal arrays ---
    signal: np.ndarray = SS_data.abd.values.astype(float)
    sleep_stages: np.ndarray = SS_data.stage.values.astype(float)
    y_algo: np.ndarray = SS_data.flow_reductions.values.astype(float)
    _below_u: np.ndarray = np.array(EM_data.d_i_ABD_smooth < 1).astype(int)
    tagged_breaths: np.ndarray = SS_data.tagged.values.astype(float)
    ss_conv_score: np.ndarray = SS_data.ss_conv_score.values.astype(float)
    selfsim: np.ndarray = SS_data.T_sim.values.astype(int)

    # --- Define row layout ---
    # Each row spans 1 hour at 10 Hz.
    fs: int = hdr["newFs"]
    block: int = 60 * 60 * fs
    row_ids: list[np.ndarray] = [np.arange(i * block, (i + 1) * block) for i in range(len(signal) // block + 1)]
    # Reverse so the earliest data is at the top.
    row_ids.reverse()
    # Adjust the last (now first) row to extend to end of recording.
    row_ids[0] = np.arange(row_ids[0][0], len(SS_data))
    nrow: int = len(row_ids)

    # --- Setup figure ---
    fig: plt.Figure = plt.figure(figsize=(12, 8))
    ax: plt.Axes = fig.add_subplot(111)
    row_height: int = 16

    # --- Build stage-separated signal arrays ---
    # NREM signal: mask wake and REM.
    sleep = np.array(signal)
    sleep[np.isnan(sleep_stages)] = np.nan
    sleep[sleep_stages == 5] = np.nan
    # Wake signal: only during wake (stage 5) or unscored (NaN).
    wake = np.zeros(signal.shape)
    wake[np.isnan(sleep_stages)] += signal[np.isnan(sleep_stages)]
    wake[sleep_stages == 5] += signal[sleep_stages == 5]
    wake[wake == 0] = np.nan
    # REM signal: only during REM (stage 4).
    rem = np.array(signal)
    rem[sleep_stages != 4] = np.nan

    # =====================================================================
    # PLOT SIGNALS -- one trace per row
    # =====================================================================
    for ri in range(nrow):
        ax.plot(sleep[row_ids[ri]] + ri * row_height, c="k", lw=0.3, alpha=0.75)
        ax.plot(wake[row_ids[ri]] + ri * row_height, c="r", lw=0.3, alpha=0.5)
        ax.plot(rem[row_ids[ri]] + ri * row_height, c="b", lw=0.3, alpha=0.5)

        if ri == nrow - 1:
            _max_y = (
                np.nanmax(
                    [
                        sleep[row_ids[ri]],
                        wake[row_ids[ri]],
                        rem[row_ids[ri]],
                    ]
                )
                + ri * row_height
            )

    # =====================================================================
    # PLOT LABELS -- event bars, tagged breaths, self-similarity
    # =====================================================================
    for yi in range(3):
        if yi == 0:
            # Respiratory events (apneas/hypopneas).
            labels = y_algo
            label_color = [None, "b", "b", "b", "m"]
        if yi == 1:
            # Tagged breathing oscillations.
            labels = tagged_breaths
            label_color = [None, "k", "r"]
        if yi == 2:
            # Self-similarity flag.
            labels = selfsim
            label_color = [None, "b"]

        for ri in range(nrow):
            loc_counter = 0
            for i, j in groupby(labels[row_ids[ri]]):
                len_j = len(list(j))
                if not np.isnan(i) and label_color[int(i)] is not None:
                    if yi == 0:
                        # Apnea bars below the signal.
                        shift = 3.5 if i == 1 else 4
                        ax.plot(
                            [loc_counter, loc_counter + len_j],
                            [ri * row_height - shift] * 2,
                            c=label_color[int(i)],
                            lw=1.5,
                            alpha=1,
                        )
                    if yi == 1:
                        # Tagged breath markers.
                        tag_marker = "o" if i == 1 else "'"
                        c_score = np.round(ss_conv_score[row_ids[ri]][loc_counter], 2)
                        c_colour, sz = ("b", 6) if c_score >= hdr["SS_threshold"] else ("k", 8)
                        if c_score >= hdr["SS_threshold"] or plot_all_tagged:
                            offset = -5
                            ax.text(
                                loc_counter,
                                ri * row_height + offset,
                                tag_marker,
                                c=c_colour,
                                ha="center",
                                va="center",
                                fontsize=sz,
                            )
                    if yi == 2:
                        # Self-similarity bar (currently a no-op for
                        # fill_between, matching the original code).
                        pass
                loc_counter += len_j

    # =====================================================================
    # PLOT EM SEGMENTS -- LG hooks
    # =====================================================================
    add_LG_hooks(EM_data, SS_data, EM_hdr, row_ids, nrow, row_height, fs, ax)

    # =====================================================================
    # LAYOUT
    # =====================================================================
    ax.set_xlim([0, max(len(x) for x in row_ids)])
    ax.axis("off")

    len_x: int = len(row_ids[-1])
    fz: int = 11
    offset = row_height * (nrow - 1) + 17
    dx: int = len_x // 10

    # Summary report in the header area.
    for i, key in enumerate(summary_report.keys()):
        tag_text = key.replace("detected ", "") + ":\n" + str(summary_report[key].values[0])
        ax.text(i * dx, offset, tag_text, fontsize=7, ha="left", va="bottom")

    # --- Bottom legend ---
    y_legend: float = -10

    # Sleep-state line types.
    line_types = ["NREM", "REM", "Wake"]
    line_colors = ["k", "b", "r"]
    for i, (color, e_type) in enumerate(zip(line_colors, line_types)):
        x = 60 * fs + 200 * fs * i
        ax.plot([x, x + 50 * fs], [y_legend] * 2, c=color, lw=0.8)
        ax.text(x + 25 * fs, y_legend - 3, e_type, fontsize=fz, c=color, ha="center", va="top")

    # Event types.
    event_types = ["Apnea", "Hypopnea"]
    label_colors_legend = ["b", "m"]
    for i, (color, e_type) in enumerate(zip(label_colors_legend, event_types)):
        x = 200 * fs * (len(line_types) + 0.5) + 300 * fs * (i + 1)
        ax.plot([x, x + 100 * fs], [y_legend] * 2, c=color, lw=2)
        ax.text(x + 50 * fs, y_legend - 3, e_type, fontsize=fz, ha="center", va="top")

    # Duration scale bar.
    duration: int = 5
    ax.plot([len_x - 60 * fs * duration, len_x], [y_legend] * 2, color="k", lw=1)
    ax.plot([len_x - 60 * fs * duration] * 2, [y_legend - 0.5, y_legend + 0.5], color="k", lw=1)
    ax.plot([len_x] * 2, [y_legend - 0.5, y_legend + 0.5], color="k", lw=1)
    ax.text(
        len_x - 60 * fs * (duration / 2),
        y_legend + 1,
        f"{duration} min",
        color="k",
        fontsize=fz,
        ha="center",
        va="bottom",
    )
    ax.text(
        len_x - 60 * fs * (duration / 2),
        y_legend - 1,
        "(abd RIP)",
        color="k",
        fontsize=8,
        ha="center",
        va="top",
    )

    # Detected SS oscillation marker.
    tag_text = "Detected SS\nbreathing oscillation"
    ax.text(
        len_x - 60 * fs * (duration / 2) - 2 * dx,
        y_legend - 3,
        tag_text,
        color="k",
        fontsize=fz - 1,
        ha="center",
        va="top",
    )
    ax.text(
        len_x - 60 * fs * (duration / 2) - 2 * dx,
        y_legend,
        "o",
        c="b",
        fontsize=fz,
        ha="center",
        va="bottom",
    )

    # Estimated LG bracket legend.
    tag_text = "Estimated LG"
    dur_samples = 8 * 60 * fs
    left = len_x - 4.75 * dx
    right = left + dur_samples
    ax.text(
        left + dur_samples / 2,
        y_legend - 3,
        tag_text,
        color="k",
        fontsize=fz - 1,
        ha="center",
        va="top",
    )
    ax.plot([left, right], [y_legend] * 2, color="k", lw=0.5)
    ax.plot([left] * 2, [y_legend - 1, y_legend], color="k", lw=0.5)
    ax.plot([right] * 2, [y_legend - 1, y_legend], color="k", lw=0.5)

    # --- Save ---
    plt.tight_layout()
    plt.savefig(fname=figure_path + ".pdf", format="pdf", dpi=1200)
    plt.close()


# ---------------------------------------------------------------------------
# LG hook annotations
# ---------------------------------------------------------------------------


def add_LG_hooks(
    data: pd.DataFrame,
    SS_data: pd.DataFrame,
    hdr: dict[str, Any],
    row_ids: list[np.ndarray],
    nrow: int,
    row_height: int,
    fs: int,
    ax: plt.Axes,
) -> None:
    """Draw LG estimation brackets on the full-night figure.

    For each NREM/REM segment, a horizontal bracket is drawn at the
    top of its row, annotated with the LG value.  If the LG was
    corrected by the post-processing outlier smoother, the annotation
    shows ``original -> corrected``.

    Segments where the EM fit is unreliable (< 4 flow-reduction events)
    are drawn with reduced opacity (``alpha=0.3``) so they are visually
    de-emphasised but still visible for diagnostic purposes.

    When a segment spans a row boundary (i.e., crosses a 1-hour mark),
    the bracket is split across two rows with the LG label placed on
    whichever side has more room.

    Args:
        data: EM output DataFrame (full night).
        SS_data: SS pipeline output DataFrame (aligned with ``data``).
        hdr: Recording header dict.
        row_ids: List of per-row sample-index arrays.
        nrow: Number of rows.
        row_height: Vertical spacing between rows in the figure.
        fs: Sampling frequency (Hz).
        ax: Matplotlib ``Axes`` to draw on.
    """
    len_x: int = len(row_ids[-1])

    for stage in ["nrem", "rem"]:
        starts = data[f"{stage}_starts"].dropna().values.astype(int)
        ends = data[f"{stage}_ends"].dropna().values.astype(int)
        if len(starts) == 0:
            continue

        LGs = np.round(data[f"LG_{stage}"].values[: len(starts)], 2)
        LGs_c = np.round(data[f"LG_{stage}_corrected"].values[: len(starts)], 2)

        for i, (st, end, LG, LG_c) in enumerate(zip(starts, ends, LGs, LGs_c)):
            # Map segment start/end to (x-position, row-index) in the
            # multi-row layout.
            x_st, y_st = find_row_location(st, row_ids)
            up_st = y_st * row_height + 0.425 * row_height

            x_end, y_end = find_row_location(end, row_ids)
            up_end = y_end * row_height + 0.425 * row_height

            # Alternate vertical offset to reduce overlap between
            # adjacent segment brackets.
            yy = up_st if i % 2 == 0 else up_st + 0.075 * row_height
            yy_ = up_end if i % 2 == 0 else up_end + 0.075 * row_height
            hook = yy - 0.05 * row_height
            hook_ = yy_ - 0.05 * row_height
            shift = yy - 0.025 * row_height if i % 2 == 0 else yy + 0.01 * row_height
            shift_ = yy_ - 0.025 * row_height if i % 2 == 0 else yy_ + 0.01 * row_height
            va = "top" if i % 2 == 0 else "bottom"

            # Segments with < 4 apneas are drawn faded.
            LG_alpha = 1 if len(find_events(SS_data.loc[st:end, "flow_reductions"] > 0)) >= 4 else 0.3
            hook_alpha = 0.5 if LG_alpha == 1 else 0.3

            # Annotation text: show correction arrow if LG was smoothed.
            tag_text = LG if LG == LG_c or np.isnan(LG) else f"{LG} --> {LG_c}"

            if y_st == y_end:
                # Segment fits entirely within one row.
                ax.plot(
                    [x_st + 15 * fs, x_end - 15 * fs],
                    [yy, yy],
                    "k",
                    lw=0.5,
                    alpha=hook_alpha,
                )
                ax.plot(
                    [x_st + 15 * fs, x_st + 15 * fs],
                    [yy, hook],
                    "k",
                    lw=0.5,
                    alpha=hook_alpha,
                )
                ax.plot(
                    [x_end - 15 * fs, x_end - 15 * fs],
                    [yy, hook],
                    "k",
                    lw=0.5,
                    alpha=hook_alpha,
                )
                x = (x_st + x_end) / 2
                ax.text(x, shift, tag_text, fontsize=6, ha="center", va=va, alpha=LG_alpha)
            else:
                # Segment crosses a row boundary -- draw split brackets.
                ax.plot(
                    [x_st + 15 * fs, len_x],
                    [yy, yy],
                    "k",
                    lw=0.5,
                    alpha=hook_alpha,
                )
                ax.plot(
                    [x_st + 15 * fs, x_st + 15 * fs],
                    [yy, hook],
                    "k",
                    lw=0.5,
                    alpha=hook_alpha,
                )
                ax.plot(
                    [0, x_end - 15 * fs],
                    [yy_, yy_],
                    "k",
                    lw=0.5,
                    alpha=hook_alpha,
                )
                ax.plot(
                    [x_end - 15 * fs, x_end - 15 * fs],
                    [yy_, hook_],
                    "k",
                    lw=0.5,
                    alpha=hook_alpha,
                )
                # Place label on whichever side has more room.
                half_win = 4 * 60 * fs
                if x_st + half_win <= len_x:
                    x = x_st + half_win
                    ax.text(x, shift, tag_text, fontsize=6, ha="center", va=va, alpha=LG_alpha)
                else:
                    x = x_end - half_win
                    ax.text(x, shift_, tag_text, fontsize=6, ha="center", va=va, alpha=LG_alpha)


# ---------------------------------------------------------------------------
# Row location finder
# ---------------------------------------------------------------------------


def find_row_location(
    loc: int,
    row_ids: list[np.ndarray],
) -> tuple[int, int]:
    """Map a global sample index to its (x, row) position.

    Each row in the full-night figure contains a subset of the
    recording's samples (one hour).  Given a global sample index, this
    function finds which row it belongs to and its position within that
    row.

    Args:
        loc: Global sample index (0-based).
        row_ids: List of per-row index arrays (as produced by
            ``plot_full_night``).

    Returns:
        A tuple ``(x, row)`` where ``x`` is the within-row index and
        ``row`` is the row number.

    Raises:
        Exception: If the index is not found in any row.
    """
    for i, row in enumerate(row_ids):
        match = np.where(loc == row)[0]
        if len(match) == 0:
            continue
        return match[0], i

    raise Exception("No matching index found..!")
