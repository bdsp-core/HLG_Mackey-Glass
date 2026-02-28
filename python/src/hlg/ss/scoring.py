"""
Self-Similarity segment score conversion.

This module converts per-segment SS scores (stored as sparse start/end
pairs) into a dense, sample-level ``SS_score`` column on the data
DataFrame.  It is extracted from ``Convert_SS_seg_scores.py``.

Background
----------
The SS segmentation step (``hlg.ss.segmentation``) computes one SS
score per overlapping time segment (typically 8-minute blocks with 50 %
overlap).  Those scores are stored compactly as three parallel columns
per sleep-stage group (NREM / REM):

    * ``{stage}_starts`` -- segment start indices
    * ``{stage}_ends``   -- segment end indices
    * ``{stage}_SS_score`` -- scalar score for each segment

This module "paints" those segment-level scores back onto every
individual sample so that downstream analyses can work with a
continuous time-series of SS quality.

Index convention
----------------
The original code uses 1-based indexing for starts/ends (Lua / MATLAB
heritage).  When assigning into the DataFrame with ``.loc``, the
expression ``st-1 : end-2`` converts to the correct 0-based inclusive
slice that matches the original segment boundaries.
"""

from __future__ import annotations


import numpy as np
import pandas as pd


def convert_ss_seg_scores_into_arrays(data: pd.DataFrame) -> pd.DataFrame:
    """Expand per-segment SS scores into a sample-level column.

    For each sleep-stage group (NREM and REM), reads the segment
    boundaries and scores, then writes the score value into every sample
    within that segment.  Where segments overlap (due to 50 % stride),
    the **last** written value wins -- this is consistent with the
    original pipeline behaviour and means later-starting segments take
    precedence in overlap regions.

    Args:
        data: DataFrame that **must** contain the following columns:

            * ``nrem_starts``, ``nrem_ends``, ``nrem_SS_score``
            * ``rem_starts``, ``rem_ends``, ``rem_SS_score``

            A new column ``SS_score`` should be pre-initialised (e.g.
            filled with NaN or 0) before calling this function; if it
            does not exist, assignment will create it.

    Returns:
        The same DataFrame with the ``SS_score`` column filled in and
        the per-stage score columns dropped.

    Notes:
        The original code called ``data.drop(columns=[...])`` without
        ``inplace=True`` and without re-assigning, so the per-stage
        columns were **not** actually removed.  This behaviour is
        preserved here to maintain exact algorithmic equivalence.
    """
    for stage in ["nrem", "rem"]:
        starts: np.ndarray = data[f"{stage}_starts"].dropna().values.astype(int)
        ends: np.ndarray = data[f"{stage}_ends"].dropna().values.astype(int)
        seg_scores: np.ndarray = data[f"{stage}_SS_score"].values

        for st, end, score in zip(starts, ends, seg_scores):
            # Convert from 1-based to 0-based inclusive range.
            # st-1 is the first sample of the segment.
            # end-2 is the last sample (one before the exclusive end).
            data.loc[st - 1 : end - 2, "SS_score"] = score

        # NOTE: In the original code this drop was not captured (no
        # inplace=True and no reassignment).  We preserve that exact
        # behaviour intentionally -- the column remains in the DataFrame.
        data.drop(columns=[f"{stage}_SS_score"])

    return data
