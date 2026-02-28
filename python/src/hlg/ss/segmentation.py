"""
NREM/REM segmentation for Self-Similarity analysis.

This module is extracted from ``SS_output_to_EM_input.py`` and handles
two tasks:

1. **Segmentation** -- ``segment_data_based_on_nrem`` divides an entire
   night's data into overlapping fixed-length blocks aligned to NREM
   and REM episodes.
2. **Scoring** -- ``compute_SS_score_per_segement`` computes the average
   Self-Similarity convolution score within each of those blocks.

Together they prepare the segment-level SS quality metrics that feed
into the Estimation Model (EM) input pipeline.

Segmentation strategy
---------------------
For each sleep-stage type (NREM stages 1-3 and REM stage 4), the
algorithm:

1. Identifies contiguous bouts of that stage using ``find_events``.
2. For each bout, determines how many full ``block_size``-minute blocks
   fit inside it.  Bouts shorter than one block are discarded.
3. Centers the blocks within the bout (the remaining time is split
   evenly as padding on both sides).
4. Uses a 50 % overlap stride (``block / 2``) to maximise the number of
   segments extracted from each bout.

This produces the highest number of EM-input segments while ensuring
every segment is entirely within a single sleep-stage bout, which is
critical because mixing NREM and REM data would violate the
physiological assumptions of the loop-gain estimation model.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from hlg.core.events import find_events


def segment_data_based_on_nrem(
    data: pd.DataFrame,
    Fs: int,
    block_size: int = 8,
) -> dict[str, np.ndarray]:
    """Divide a recording into overlapping blocks aligned to NREM / REM.

    Scans the ``Stage`` column for contiguous NREM (1-3) and REM (4)
    bouts, then tiles each bout with overlapping fixed-length segments.

    Args:
        data: DataFrame that must contain a ``Stage`` column with AASM
            sleep-stage labels (0 = wake, 1-3 = NREM, 4 = REM, 5 =
            unscored).
        Fs: Sampling frequency in Hz (typically 10 Hz after
            downsampling).
        block_size: Segment length in **minutes**.  Default is 8 min,
            chosen to contain ~100-160 breaths at normal respiratory
            rates -- enough for reliable loop-gain estimation while being
            short enough to capture within-night variability.

    Returns:
        A dict with four keys:

        * ``nrem_starts`` -- int array of segment start indices (NREM)
        * ``nrem_ends``   -- int array of segment end indices (NREM)
        * ``rem_starts``  -- int array of segment start indices (REM)
        * ``rem_ends``    -- int array of segment end indices (REM)

    Notes:
        The 50 % overlap stride means adjacent segments share half their
        data.  This is deliberate: it doubles the number of EM
        estimates and smooths within-night trends, at the cost of
        correlation between adjacent segments.
    """
    stages: np.ndarray = data.Stage.values

    # NREM: stages 1 (N1), 2 (N2), 3 (N3) -- all non-REM sleep.
    nrem: np.ndarray = np.logical_and(stages > 0, stages < 4)
    # REM: stage 4.
    rem: np.ndarray = stages == 4

    seg_dic: dict[str, np.ndarray] = {}

    # Convert block size from minutes to samples.
    block: int = int(block_size * 60 * Fs)

    for SS, tag in zip([nrem, rem], ["nrem", "rem"]):
        starts: np.ndarray = np.array([])

        for st, end in find_events(SS):
            # How many full blocks fit in this bout?
            blocks: int = (end - st) // block
            if blocks == 0:
                # Bout is shorter than one block -- skip it.
                continue

            # Center the blocks within the bout by splitting the
            # leftover time equally on both sides.
            shift: float = ((end - st) - blocks * block) / 2

            # Generate segment start positions with 50 % overlap.
            starts = np.concatenate([starts, np.arange(st + shift, end - block, block / 2).tolist()])

        seg_dic[f"{tag}_starts"] = starts.astype(int)
        seg_dic[f"{tag}_ends"] = starts.astype(int) + block

    return seg_dic


def compute_SS_score_per_segement(
    data: pd.DataFrame,
    seg_dic: dict[str, np.ndarray],
) -> dict[str, list[float]]:
    """Compute the average SS convolution score for each segment.

    For every segment (defined by ``seg_dic``), computes the mean of the
    ``ss_conv_score`` column.  Segments with a valid (non-NaN) mean are
    kept as-is.  Segments whose mean is NaN (typically because all
    samples are NaN) are assigned a score of **0** if the segment
    contains 4 or more respiratory events -- the rationale being that
    frequent apneas disrupt the periodicity detector but still indicate
    unstable breathing.

    Args:
        data: DataFrame containing at least ``ss_conv_score`` and
            ``Apnea`` columns.
        seg_dic: Segment dictionary as returned by
            ``segment_data_based_on_nrem``, with ``{stage}_starts`` and
            ``{stage}_ends`` keys.

    Returns:
        A dict with keys ``"rem"`` and ``"nrem"``, each mapping to a
        list of float scores (one per segment).

    Notes:
        The function name intentionally preserves the original
        misspelling ("segement") for backward compatibility with
        downstream code that may reference it by name.
    """
    SS_seg_scores: dict[str, list[float]] = {}

    for SS in ["rem", "nrem"]:
        SS_list: list[float] = []

        for st, end in zip(seg_dic[f"{SS}_starts"], seg_dic[f"{SS}_ends"]):
            SS_score: float = np.round(data.loc[st:end, "ss_conv_score"].mean(), 2)

            if not np.isnan(SS_score):
                SS_list.append(SS_score)
            else:
                # If the segment has >= 4 apnea events despite a NaN SS
                # score, assign 0 rather than NaN.  This captures
                # segments where periodic breathing is disrupted by
                # frequent obstructive events -- the SS detector fails
                # but the clinical picture is still one of ventilatory
                # instability.
                if len(find_events(data.loc[st:end, "Apnea"] > 0)) >= 4:
                    SS_list.append(0)
                else:
                    SS_list.append(SS_score)

        SS_seg_scores[SS] = SS_list

    return SS_seg_scores
