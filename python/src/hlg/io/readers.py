"""
HDF5 file readers for Self-Similarity (SS) pipeline output.

This module consolidates **all** HDF5-reading logic that was previously
duplicated across three scripts (``SS_output_to_EM_input.py``,
``Stable_SS_analysis.py``, ``update_MGH_info.py``).  The canonical
implementation is ``load_sim_output``, which reads the fixed-schema
``.hf5`` files produced by the SS analysis pipeline and returns a tidy
DataFrame plus a header dictionary.

A second reader, ``load_SS_percentage``, is a lightweight convenience
function that extracts only the self-similarity percentage and
flow-reduction trace from a single recording's output file.  It was
extracted from ``EM_output_extraction.py`` and
``EM_output_to_Alitude_Relationship.py``.

Design rationale
----------------
* **Single source of truth** -- all downstream consumers now call into
  ``hlg.io.readers`` instead of maintaining their own copy of the
  loading logic.  This eliminates drift between the three previous
  copies.
* **Header extraction pattern** -- PSG metadata (patient tag, test type,
  sampling rate, etc.) is stored as length-1 columns in the HDF5 file
  (a legacy convention).  We peel them out into a dict so the returned
  DataFrame contains only time-series data.
* **Chest->abdomen fallback** -- some recordings store the respiratory
  effort signal under ``chest`` instead of ``abd`` (the canonical name).
  The function transparently renames to maintain a uniform schema.
"""

from __future__ import annotations

import glob
from typing import Any

import h5py
import numpy as np
import pandas as pd

from hlg.core.sleep_metrics import compute_sleep_metrics


# ---------------------------------------------------------------------------
# Default column names for the SS pipeline HDF5 schema
# ---------------------------------------------------------------------------

# Header columns carry per-recording metadata (scalar values), not
# time-series data.  They are extracted into the ``hdr`` dict.
_HDR_COLS: list[str] = [
    "patient_tag",
    "test_type",
    "rec_type",
    "cpap_start",
    "Fs",
    "SS_threshold",
]

# Signal columns carry the actual time-series arrays at 10 Hz.
_DEFAULT_SIGNAL_COLS: list[str] = [
    "abd",
    "chest",
    "spo2",
    "apnea",
    "arousal",
    "sleep_stages",
    "flow_reductions",
    "tagged",
    "self similarity",
    "ss_conv_score",
]


def load_sim_output(
    path: str,
    cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load Self-Similarity pipeline output from an HDF5 file.

    Reads the ``.hf5`` file produced by the SS analysis pipeline and
    returns a ``(data, hdr)`` tuple.  ``data`` is a DataFrame whose
    columns are the requested time-series signals (default: all SS
    outputs).  ``hdr`` is a dict of scalar metadata extracted from the
    file (patient ID, sampling rate, sleep-metric summaries, etc.).

    This function is the **canonical** reader and replaces three
    near-identical copies that previously existed in the codebase.

    Args:
        path: Filesystem path to the ``.hf5`` file.
        cols: Explicit list of column names to load.  If ``None`` or
            empty, a default set of signal + header columns is used
            (see module-level ``_DEFAULT_SIGNAL_COLS`` and
            ``_HDR_COLS``).

    Returns:
        A 2-tuple ``(data, hdr)`` where:

        * **data** -- ``pd.DataFrame`` of shape ``(n_samples, n_signals)``
          containing only time-series columns (header columns are
          removed).  If sleep stages are present, an additional boolean
          column ``patient_asleep`` is added.
        * **hdr** -- ``dict`` with at least ``newFs`` (resampled
          frequency, always 10 Hz) and any scalar metadata found in the
          file.  When sleep stages are available, ``RDI``, ``AHI``,
          ``CAI``, and ``sleep_time`` are computed and included.

    Raises:
        OSError: If the HDF5 file cannot be opened.

    Notes:
        * Columns listed in ``cols`` that do not exist in the file are
          silently skipped -- this makes the function tolerant of older
          pipeline versions that may not have produced every column.
        * If the ``abd`` column is entirely NaN but ``chest`` is not,
          the chest signal is used as a substitute and the column is
          renamed to ``abd``.  This handles recordings where only a
          thoracic belt was available.
    """
    if cols is None or len(cols) == 0:
        cols = _DEFAULT_SIGNAL_COLS + _HDR_COLS

    data = pd.DataFrame([], columns=cols)

    # ── Read raw arrays from HDF5 ────────────────────────────────────
    f = h5py.File(path, "r")
    for key in cols:
        if key not in f.keys():
            # Silently skip missing columns for backward compatibility
            # with older pipeline versions.
            continue
        vals = f[key][:]
        data[key] = vals
    f.close()

    # ── Chest -> abdomen fallback ─────────────────────────────────────
    # Some sites record only a thoracic belt and store it under "chest"
    # rather than "abd".  Downstream code always expects "abd", so we
    # transparently rename when the abdominal trace is empty.
    if "abd" in data.columns and "chest" in data.columns and all(data.abd.isna()) and not all(data.chest.isna()):
        print('"CHEST" replaced "ABD"!')
        data["abd"] = data.chest.values
        data = data.drop(columns=["chest"])

    # ── Peel header columns into a dict ──────────────────────────────
    # Pipeline convention: metadata is stored as length-N columns whose
    # values are identical across all rows.  We take the first row and
    # move them out of the DataFrame so it contains only numeric series.
    hdr: dict[str, Any] = {}

    # The downstream resampled rate is always 10 Hz, regardless of the
    # original PSG sampling rate stored in the file.
    hdr["newFs"] = 10

    for hf in _HDR_COLS:
        if hf not in data.columns:
            continue
        val = data.loc[0, hf]
        # HDF5 stores strings as bytes -- decode to native Python str.
        try:
            val = val.decode("utf-8")
        except (AttributeError, UnicodeDecodeError):
            pass
        hdr[hf] = val
        data = data.drop(columns=[hf])

    # ── Derive sleep metrics if staging is available ─────────────────
    # Sleep stages follow AASM encoding: 0 = wake, 1-3 = NREM, 4 = REM,
    # 5 = unscored/artifact.  "Asleep" is any stage 1-4 (inclusive).
    if "sleep_stages" in data.columns:
        data["patient_asleep"] = np.logical_and(data.sleep_stages < 5, data.sleep_stages > 0)
        # Compute standard sleep-disordered-breathing indices.
        RDI, AHI, CAI, sleep_time = compute_sleep_metrics(data.apnea, data.sleep_stages, exclude_wake=True)
        hdr["RDI"] = RDI
        hdr["AHI"] = AHI
        hdr["CAI"] = CAI
        hdr["sleep_time"] = sleep_time

    return data, hdr


def load_SS_percentage(
    hf5_folder: str,
    ID: str,
) -> tuple[float, np.ndarray]:
    """Load the Self-Similarity percentage for a single recording.

    This is a lightweight reader that opens one HDF5 output file, reads
    only the four columns needed to compute the percentage of sleep time
    that is "self-similar" (i.e. periodic breathing), and returns that
    percentage together with the raw flow-reduction trace.

    The Self-Similarity percentage is defined as::

        SS% = (samples where SS == 1) / (samples where patient is asleep) x 100

    rounded to one decimal place.

    This was previously duplicated in ``EM_output_extraction.py`` and
    ``EM_output_to_Alitude_Relationship.py``.

    Args:
        hf5_folder: Directory containing the ``.hf5`` output files.
            Must include a trailing path separator (e.g. ``"output/"``).
        ID: Recording identifier.  The function searches for a file
            whose name ends in ``{ID}.hf5`` within ``hf5_folder``.

    Returns:
        A 2-tuple ``(SS, flow_reductions)`` where:

        * **SS** -- Self-similarity percentage (float, 0-100, 1 d.p.).
        * **flow_reductions** -- 1-D numpy array of the flow-reduction
          signal from the HDF5 file.

    Raises:
        AssertionError: If zero or more than one matching file is found.
    """
    cols: list[str] = [
        "patient_tag",
        "self similarity",
        "sleep_stages",
        "flow_reductions",
    ]
    data = pd.DataFrame([], columns=cols)

    # Locate the single matching file by glob pattern.
    path = [p for p in glob.glob(hf5_folder + "*.hf5") if f"{ID}.hf5" in p]
    assert len(path) == 1, f"No matching SS output file found in {hf5_folder}"

    f = h5py.File(path[0], "r")
    for key in cols:
        data[key] = f[key][:]
    f.close()

    # "Asleep" follows AASM staging: stages 1-4 (NREM + REM).
    patient_asleep = np.logical_and(data["sleep_stages"] > 0, data["sleep_stages"] < 5)

    # SS percentage: fraction of asleep samples with SS flag == 1.
    SS: float = np.round((np.sum(data["self similarity"] == 1) / (sum(patient_asleep))) * 100, 1)

    return SS, data.flow_reductions.values
