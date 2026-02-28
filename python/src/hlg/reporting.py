"""
Clinical report generation and output persistence.

After the Self-Similarity (SS) pipeline has scored a recording, the
results are saved to an HDF5 file (``save_output``) and summarised in
a two-part clinical report (``create_report``):

* **Full report** -- one row per second of recording, with start/end
  sample indices for mapping back to the original PSG.
* **Summary report** -- a single-row table with derived indices:
  signal duration, detected central apneas and hypopneas, CAI, CAHI,
  and the overall Self-Similarity percentage.

The report mirrors the output tables that clinicians use to triage
patients for CPAP titration: a high CAHI or SS% indicates significant
periodic breathing and guides treatment decisions.

Source: ``Save_and_Report.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hlg.io.writers import write_to_hdf5_file
from hlg.core.events import find_events


def save_output(
    data: pd.DataFrame,
    hdr: dict[str, Any],
    out_file: str,
    channels: list[str],
) -> None:
    """Persist SS pipeline output to an HDF5 file.

    Collects the requested signal channels plus standard derived
    columns (apnea mask, flow reductions, sleep stages, self-similarity
    score, and the tagged/scored arrays) into a single DataFrame and
    writes it to ``out_file`` via the canonical HDF5 writer.  Recording
    metadata from ``hdr`` is appended as constant-valued columns (the
    legacy HDF5 schema convention used throughout the pipeline).

    Args:
        data: SS pipeline output DataFrame.  Expected to contain at
            least ``resp``, ``flow_reductions``, ``stage``, ``T_sim``,
            ``TAGGED``, and ``ss_conv_score`` columns, plus any signal
            channels listed in ``channels``.
        hdr: Recording header dictionary.  Every key/value pair is
            written as a constant column in the output HDF5 file so
            that metadata travels with the data.
        out_file: Filesystem path for the output ``.hf5`` file.
        channels: List of raw signal channel names to include (e.g.
            ``['abd', 'chest']``).  The function also always attempts
            to include ``'spo2'`` and ``'arousal'`` if present.
    """
    df: pd.DataFrame = pd.DataFrame([])

    # Copy requested signal channels plus SpO2 and arousal if available.
    for ch in channels + ["spo2", "arousal"]:
        if ch in data.columns:
            df[ch] = data[ch].values

    # The legacy naming convention stores the respiratory event mask
    # under "resp" in memory but "apnea" in the HDF5 file.
    if "resp" in data.columns:
        df["apnea"] = data.resp.values

    df["flow_reductions"] = data.flow_reductions.values
    df["sleep_stages"] = data.stage.values
    df["self similarity"] = data.T_sim.values
    df["tagged"] = data.TAGGED.values
    df["ss_conv_score"] = data.ss_conv_score.values

    # Flatten header metadata into the DataFrame as constant columns
    # (one scalar repeated across all rows).  This is the HDF5 schema
    # convention expected by ``load_sim_output``.
    for key in hdr.keys():
        df[key] = hdr[key]

    write_to_hdf5_file(df, out_file, overwrite=True)


def create_report(
    output_data: pd.DataFrame,
    hdr: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a clinical summary report from SS pipeline output.

    The report has two components:

    1. **Full report** -- one row per *second* of recording.  Each row
       carries start/end sample indices (at the original PSG sampling
       rate) so that the clinician can jump to any second in the raw
       recording.

    2. **Summary report** -- a single row with aggregate indices:

       * Signal duration (hours of sleep).
       * Count of detected central apneas and hypopneas.
       * Central Apnea Index (CAI) -- apneas per hour.
       * Central Apnea-Hypopnea Index (CAHI) -- total events per hour.
       * Self-Similarity percentage (SS%) -- fraction of recording
         classified as periodic breathing.

    Event classification:
        An event is a contiguous region where the SS convolution score
        exceeds the ``SS_threshold``.  Events overlapping any wake
        epoch (stage == 0) are excluded.  An event is labelled "apnea"
        if complete flow cessation (flow_reductions == 1) occurs within
        the first 30 seconds; otherwise it is labelled "hypopnea".

    Args:
        output_data: SS pipeline output DataFrame.  Must contain
            ``flow_reductions``, ``T_sim``, ``stage``, and
            ``ss_conv_score`` columns.
        hdr: Recording header dict.  Must contain ``Fs`` (original
            sampling frequency), ``newFs`` (resampled frequency, always
            10 Hz), and ``SS_threshold``.

    Returns:
        A 2-tuple ``(full_report, summary_report)`` of DataFrames.
    """
    originalFs: int = hdr["Fs"]
    newFs: int = hdr["newFs"]

    # The report is at 1-second resolution.
    finalFs: int = 1

    original_cols: list[str] = ["flow_reductions", "T_sim", "stage"]
    data: pd.DataFrame = pd.DataFrame([], columns=["start_idx", "end_idx"] + original_cols)

    # Downsample each signal column from newFs (10 Hz) to 1 Hz by
    # repeating at finalFs then decimating.  This is a simple
    # nearest-neighbour downsampling -- appropriate because these are
    # categorical/binary signals, not continuous waveforms.
    for sig in original_cols:
        image: np.ndarray = np.repeat(output_data[sig].values, finalFs)
        image = image[::newFs]
        data[sig] = image

    # Build sample-index columns that map each 1-second row back to
    # the original PSG timeline at the full sampling rate.
    factor: int = originalFs // finalFs
    ind0: np.ndarray = np.arange(0, data.shape[0]) * factor
    ind1: np.ndarray = np.concatenate([ind0[1:], [ind0[-1] + factor]])

    data["second"] = range(len(data))
    data["start_idx"] = ind0
    data["end_idx"] = ind1

    # ── Event detection and classification ───────────────────────────
    # Detect contiguous regions above the SS threshold, then classify
    # each as apnea vs. hypopnea based on whether complete flow
    # cessation occurs within the first 30 seconds of the event.
    lengths: dict[str, int] = {"apnea": 0, "hypopnea": 0}

    for st, end in find_events(output_data["ss_conv_score"] >= hdr["SS_threshold"]):
        # Exclude events that overlap with wake epochs -- respiratory
        # events during wake are not clinically scored.
        if any(data.loc[st:end, "stage"] == 0):
            continue

        tag: str = "hypopnea"
        # Check the first 30 seconds (at newFs) of the event for
        # complete flow cessation (flow_reductions == 1 indicates a
        # full apnea; < 1 indicates partial reduction = hypopnea).
        if np.any(output_data.loc[st : st + 30 * newFs, "flow_reductions"] == 1):
            tag = "apnea"

        lengths[tag] += 1

    _SS_num: int = len(find_events(output_data["ss_conv_score"] >= hdr["SS_threshold"]))

    # ── Summary report ────────────────────────────────────────────────
    summary_report: pd.DataFrame = pd.DataFrame([])

    # Duration of sleep in hours (stage != 0 means scored as sleep or
    # NREM/REM; this excludes wake but includes all other stages).
    duration: float = sum(data.stage != 0) / finalFs / 3600
    summary_report["signal duration (h)"] = [np.round(duration, 2)]

    summary_report["detected central apneas"] = [lengths["apnea"]]
    summary_report["detected central hypopneas"] = [lengths["hypopnea"]]

    # CAI: central apnea index (apneas per hour of sleep).
    summary_report["cai"] = [np.round(lengths["apnea"] / duration, 1)]

    # CAHI: central apnea-hypopnea index (total events per hour).
    summary_report["cahi"] = [np.round(sum(lengths.values()) / duration, 1)]

    # SS%: percentage of recording classified as periodic breathing.
    summary_report["SS%"] = np.round((np.sum(data["T_sim"] == 1) / (len(data))) * 100, 1)

    # Drop the raw signal columns from the full report -- they were
    # only needed for event detection above.
    for col in original_cols:
        if col in data.columns:
            data = data.drop(columns=col)

    full_report: pd.DataFrame = pd.concat([data, summary_report], axis=1)

    return full_report, summary_report
