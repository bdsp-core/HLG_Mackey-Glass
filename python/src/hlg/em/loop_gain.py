"""
Per-sample loop gain array reconstruction from segment-level EM estimates.

The Estimation Model (EM) fits a physiological ventilatory-control model
to each 8-minute sleep segment and produces a single scalar loop gain
(LG) value per segment.  For downstream analyses (hypnograms, histograms,
CPAP-response prediction) we need a *sample-level* LG trace aligned
with the original PSG time axis.

This module provides ``create_total_LG_array``, which "paints" each
segment's corrected LG value onto every sample that falls within that
segment's [start, end) window.  Wake epochs and unscored epochs are
masked with ``NaN`` so they are excluded from any subsequent averaging.

Source: ``Recreate_LG_array.py``

No local package dependencies -- only numpy is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_total_LG_array(data: pd.DataFrame) -> np.ndarray:
    """Build a sample-level loop gain trace from segment-level estimates.

    For every NREM and REM segment identified by the EM, the function
    assigns the segment's *corrected* loop gain value (after outlier
    smoothing in ``post_process_EM_output``) to every sample within
    that segment.  Samples that belong to wake (stage == 5) or have
    no scored stage (NaN) are set to ``NaN`` so that downstream
    aggregation (e.g. histogram binning) naturally ignores them.

    The resulting array has the same length as the input DataFrame and
    can be directly attached as a new column (``data['total_LG']``).

    Args:
        data: EM output DataFrame that must contain at minimum:

            * ``nrem_starts``, ``nrem_ends`` -- start/end indices of
              NREM segments (may contain trailing NaN padding).
            * ``rem_starts``, ``rem_ends`` -- same for REM segments.
            * ``LG_nrem_corrected``, ``LG_rem_corrected`` -- the
              outlier-corrected loop gain for each segment.
            * ``Stage`` -- AASM sleep stage per sample (0 = wake,
              1-3 = NREM, 4 = REM, 5 = unscored/artifact, NaN =
              missing).

    Returns:
        A 1-D numpy array of length ``len(data)`` where each element
        is the corrected loop gain of its enclosing segment, or
        ``NaN`` for wake / unscored / unsegmented samples.
    """
    total_array: np.ndarray = np.zeros(len(data))

    # Iterate over both sleep macro-stages.  NREM and REM segments are
    # stored in separate column pairs because the EM fits them with
    # different physiological constraints (e.g. upper-airway gain
    # differs between NREM and REM).
    for stage in ["nrem", "rem"]:
        # Segment boundaries are stored as floats with NaN padding
        # beyond the actual number of segments -- drop the padding.
        starts: np.ndarray = data[f"{stage}_starts"].dropna().values.astype(int)
        ends: np.ndarray = data[f"{stage}_ends"].dropna().values.astype(int)

        for i, (start, end) in enumerate(zip(starts, ends)):
            # Each row ``i`` in the segment column corresponds to
            # one segment; fetch its corrected LG scalar.
            LG: float = data.loc[i, f"LG_{stage}_corrected"]

            # "Paint" the scalar LG across every sample in [start, end).
            total_array[start:end] = LG

    # Mask wake and unscored epochs.  Stage == 5 is the AASM code for
    # "movement / artifact / unscored" and NaN indicates epochs the
    # sleep stager could not classify.  Both are physiologically
    # meaningless for loop-gain estimation.
    total_array[np.isnan(data.Stage)] = np.nan
    total_array[data.Stage == 5] = np.nan

    return total_array
