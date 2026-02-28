"""
Stable Self-Similarity detection and change-point analysis.

This module is extracted from ``Stable_SS_analysis.py`` and contains
the two core computation functions.  Plotting helpers from the same
source file are migrated separately to ``hlg.visualization``.

The module identifies regions of **sustained periodic breathing** in an
overnight recording using two complementary approaches:

1. ``compute_osc_chains`` -- a heuristic rolling-window detector that
   flags contiguous runs of above-threshold SS convolution score.
2. ``compute_change_points_ruptures`` -- a statistical change-point
   detector (Pelt algorithm from the ``ruptures`` library) that
   partitions the smoothed SS trace into piecewise-stationary segments
   and labels those with non-zero median as "stable SS".

Both functions annotate the input DataFrame with new indicator columns
(``Osc_chain`` and ``stable_SS`` respectively) whose values are:

* **0** -- no stable periodic breathing
* **1** -- stable periodic breathing during NREM
* **2** -- stable periodic breathing during REM

The distinction between NREM and REM is clinically important because
loop gain (LG) manifests differently in each state: NREM oscillations
are driven by high LG, while REM oscillations suggest severe
instability because REM normally suppresses periodic breathing.

Minimum duration filter
-----------------------
Both functions discard short bursts (< 2 min for osc chains, < 3 min
for change-point segments) to avoid false-positive detections from
transient arousals or movement artifact.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import ruptures as rpt

from hlg.core.events import find_events


def compute_osc_chains(
    data: pd.DataFrame,
    hdr: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect oscillation chains via rolling-window SS thresholding.

    An "oscillation chain" is a sustained epoch where the SS convolution
    score exceeds the detection threshold.  The algorithm works in three
    steps:

    1. **Threshold** -- mark every sample where ``ss_conv_score >=
       SS_threshold`` as belonging to an oscillation.
    2. **Dilate** -- apply a 3-minute rolling maximum to bridge brief
       gaps (e.g. arousals that momentarily break the periodic pattern).
    3. **Label** -- assign 1 for NREM and 2 for REM, then remove any
       chain shorter than 2 minutes (likely artifact).

    Args:
        data: DataFrame with at least ``sleep_stages`` and
            ``ss_conv_score`` columns.
        hdr: Header dict that must contain ``SS_threshold`` (float, the
            detection threshold) and ``newFs`` (int, sampling rate in
            Hz).

    Returns:
        A 2-tuple ``(data, hdr)`` where ``data`` has a new
        ``Osc_chain`` column (values 0 / 1 / 2) and ``hdr`` is passed
        through unchanged.

    Notes:
        The ``REM_breakpoints`` column is an intermediate mask used to
        separate NREM (0) from REM (1) samples.  It is written to
        ``data`` as a side-effect and may be useful for downstream
        visualisation.
    """
    # ── Build REM breakpoint mask ────────────────────────────────────
    # This mask distinguishes asleep-NREM (0) from REM (1) and marks
    # wake/unscored as NaN so they can be excluded later.
    data["REM_breakpoints"] = np.nan
    notnan = np.where(data.sleep_stages > 0)[0]
    data.loc[notnan, "REM_breakpoints"] = 0
    rem = np.where(data.sleep_stages == 4)[0]
    data.loc[rem, "REM_breakpoints"] = 1

    # ── Threshold the SS convolution score ───────────────────────────
    data["Osc_chain"] = 0
    data.loc[np.where(data["ss_conv_score"] >= hdr["SS_threshold"])[0], "Osc_chain"] = 1

    # ── Dilate with 3-minute rolling max ─────────────────────────────
    # A rolling maximum over a 3-minute window bridges transient dips
    # (arousals, movement artifact) that would otherwise fragment a
    # single sustained oscillation into many short pieces.
    data["Osc_chain"] = data["Osc_chain"].rolling(int(3 * 60 * hdr["newFs"]), center=True).max()

    # ── Label REM oscillation chains as 2 ────────────────────────────
    data.loc[
        np.where(np.logical_and(data["REM_breakpoints"] == 1, data["Osc_chain"] == 1))[0],
        "Osc_chain",
    ] = 2

    # Wake / unscored samples should not carry an oscillation label.
    data.loc[np.where(data["REM_breakpoints"].isna())[0], "Osc_chain"] = 0

    # ── Minimum duration filter (2 minutes) ──────────────────────────
    # Short bursts are likely arousals or noise rather than true
    # sustained periodic breathing.
    for i in range(1, 3):
        for st, end in find_events(data["Osc_chain"] == i):
            if end - st < 2 * 60 * hdr["newFs"]:
                data.loc[st:end, "Osc_chain"] = 0

    return data, hdr


def compute_change_points_ruptures(
    data: pd.DataFrame,
    hdr: dict[str, Any],
) -> pd.DataFrame:
    """Detect stable SS regions using Pelt change-point detection.

    This function provides a more statistically rigorous alternative to
    ``compute_osc_chains``.  It smooths the SS convolution score,
    applies the Pelt algorithm with an RBF kernel to find change points,
    then labels intervals whose median smoothed score is positive as
    "stable SS".

    Algorithm steps:

    1. **Threshold** -- zero out sub-threshold SS values (noise floor
       removal).
    2. **Smooth** -- apply a 3-minute centered rolling median to remove
       breath-to-breath variability and highlight the envelope.
    3. **Downsample** -- reduce to 1 Hz (``scale = 10 * Fs``) for
       computational efficiency in the change-point search.
    4. **Pelt** -- run ``ruptures.Pelt`` with an RBF kernel and penalty 4
       to find change points in the downsampled trace.
    5. **Label** -- for each inter-change-point interval, compute the
       median of the smoothed SS trace.  Intervals with positive median
       are marked as stable (1 = NREM, 2 = REM).
    6. **Filter** -- discard stable-SS regions shorter than the smoothing
       window (3 min).

    Args:
        data: DataFrame with at least ``ss_conv_score`` and
            ``sleep_stages`` columns.
        hdr: Header dict that must contain ``SS_threshold`` (float) and
            ``newFs`` (int, Hz).

    Returns:
        The input DataFrame with a new ``stable_SS`` column (values
        0 / 1 / 2).

    Notes:
        The ``SS_trace`` column (smoothed, threshold-applied SS) is
        written to ``data`` as a side-effect.  Downstream visualisation
        code expects it.
    """
    # ── Threshold and smooth ─────────────────────────────────────────
    SS_trace: np.ndarray = data["ss_conv_score"].values.copy()

    # Zero out sub-threshold values to eliminate low-level noise that
    # would bias the rolling median upward.
    SS_trace[SS_trace < hdr["SS_threshold"]] = np.nan

    # 3-minute centered rolling median smooths breath-to-breath
    # variability while preserving the onset/offset of sustained
    # oscillation epochs.
    win: int = int(3 * 60 * hdr["newFs"])
    SS_trace = np.squeeze(
        pd.DataFrame(data=SS_trace).rolling(win, min_periods=1, center=True).median().fillna(0).values
    )

    # ── Downsample for Pelt efficiency ───────────────────────────────
    # The Pelt algorithm is O(n^2) in the worst case.  Downsampling from
    # 10 Hz to ~1 Hz (factor of 10) makes it tractable for 8-hour
    # recordings while preserving the multi-minute structure we care
    # about.
    scale: int = int(10 * hdr["newFs"])
    detector = rpt.Pelt(model="rbf").fit(SS_trace[::scale])

    # Penalty of 4 was empirically tuned: lower values over-segment
    # (too many change points), higher values under-segment (miss
    # transitions between stable and unstable epochs).
    change_points: np.ndarray = detector.predict(pen=4)

    # Scale change-point indices back to the original sampling rate.
    change_points = np.array(change_points) * scale

    # ── Label stable intervals ───────────────────────────────────────
    data["stable_SS"] = 0

    for i, loc0 in enumerate(change_points[:-1]):
        loc1 = change_points[i + 1]
        # An interval is "stable SS" if its median smoothed score is
        # positive (i.e. sustained above-threshold oscillations).
        if data.loc[loc0:loc1, "SS_trace"].median() > 0:
            data.loc[loc0:loc1, "stable_SS"] = 1

    # ── Mark REM stable-SS as category 2 ─────────────────────────────
    rem = np.where(np.logical_and(data.sleep_stages == 4, data.stable_SS == 1))[0]
    data.loc[rem, "stable_SS"] = 2

    # ── Minimum duration filter ──────────────────────────────────────
    # Discard stable-SS regions shorter than the smoothing window
    # (3 min) -- these are likely transient.
    for i in range(1, 3):
        for st, end in find_events(data["stable_SS"] == i):
            if end - st < win:
                data.loc[st:end, "stable_SS"] = 0

    return data
