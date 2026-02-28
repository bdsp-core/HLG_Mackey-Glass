"""
Loop gain histogram computation and CPAP-response prediction.

The EM produces a single LG value per 8-minute sleep segment.  To
characterise a patient's *distribution* of loop gain across the night
(rather than a single summary statistic), the pipeline bins the
sample-level LG trace into 8-minute epochs, computes the mean LG per
epoch, and then builds a 10-bar histogram over the range [0, 1] with
step 0.1.

The resulting "histogram bars" capture the shape of the LG distribution
and have been shown to discriminate CPAP responders from non-responders
better than the median LG alone (because two patients can share the same
median LG but have very different LG variability).

The prediction model is a simple nearest-mean classifier: a patient's
histogram is compared (total absolute error) to the average histograms
of known CPAP successes and failures, and the difference in error scores
gives a "LG Bar" score (positive -> more similar to success).

Source: ``EM_output_histograms.py`` (compute / load / predict functions).
"""

from __future__ import annotations

import glob

import h5py
import numpy as np
import pandas as pd


def compute_histogram(
    data: pd.DataFrame,
    hdr: dict,
    bar_folder: str,
) -> list[float]:
    """Bin a patient's sample-level LG trace into a 10-bar histogram.

    The algorithm:

    1. Divides the recording into non-overlapping 8-minute epochs
       (matching the EM segment duration).
    2. Discards epochs where the patient is awake for >= 80 % of the
       epoch (too little sleep to yield a meaningful LG).
    3. For each surviving epoch, computes the mean of the non-NaN LG
       samples.
    4. Bins the epoch means into 10 equal-width bins spanning [0, 1].

    The bars are also persisted to an HDF5 file for later batch loading
    (see ``load_histogram_bars``).

    Args:
        data: EM output DataFrame.  Must contain ``Stage``, ``total_LG``
            columns.  A ``patient_asleep`` column is added internally.
        hdr: Recording header dict.  Must contain ``Fs`` (sampling
            frequency) and ``Study_num`` (used as the output filename).
        bar_folder: Output directory for the per-patient HDF5 bar files
            (with trailing separator).

    Returns:
        A list of 10 floats representing the percentage of epochs
        falling into each LG bin [0-0.1), [0.1-0.2), ..., [0.9-1.0].
    """
    # Mark every sample as asleep or awake using AASM staging.
    # Stages 1-4 = asleep; 0 = wake; 5 = unscored.
    data["patient_asleep"] = np.logical_and(data.Stage.values > 0, data.Stage.values < 5)

    # 8-minute epoch size in samples (matches the EM segment length).
    epoch_size: int = int(round(8 * 60 * hdr["Fs"]))
    epoch_inds: np.ndarray = np.arange(0, len(data) - epoch_size + 1, epoch_size)

    # Build index arrays for each epoch (one array of sample indices per
    # epoch).
    seg_ids: list[np.ndarray] = [np.arange(x, x + epoch_size) for x in epoch_inds]

    # Samples where a valid (non-NaN) LG estimate exists.
    score_present: np.ndarray = data.total_LG.dropna().index.values

    bin_means: np.ndarray = np.zeros(len(seg_ids))
    for i, seg_id in enumerate(seg_ids):
        # Skip epochs with < 20 % sleep -- insufficient data for a
        # reliable LG average.
        if data.loc[seg_id, "patient_asleep"].sum() < 0.20 * epoch_size:
            bin_means[i] = -1
            continue

        # Skip epochs that have no scored LG samples at all.
        if not any(s in seg_id for s in score_present):
            continue

        bin_means[i] = data.loc[seg_id, "total_LG"].dropna().mean()

    bars: list[float] = histogram_bins_to_bars(bin_means)
    save_histogram_bars(np.array(bars), hdr["Study_num"], bar_folder)

    return bars


def histogram_bins_to_bars(bins: np.ndarray) -> list[float]:
    """Convert epoch-mean LG values into a 10-bar percentage histogram.

    Epochs marked with -1 (insufficient sleep) are excluded before
    binning.  Each bar represents the percentage of remaining epochs
    whose mean LG falls within a 0.1-wide bin from 0.0 to 1.0.

    Args:
        bins: 1-D array of epoch mean LG values.  Negative values
            (sentinel for "insufficient sleep") are filtered out.

    Returns:
        A list of 10 floats (percentages summing to ~100 %).
    """
    # Remove sentinel values (epochs with too much wake).
    bins = bins[bins >= 0]

    step: float = 0.1
    steps: np.ndarray = np.arange(0, 1.1, step)

    bars: list[float] = []
    for block in steps[:-1]:
        # The +0.0001 epsilon on the upper bound ensures that values
        # landing exactly on a bin edge are captured (floating-point
        # tolerance).
        percentage: float = (
            sum(np.logical_and(np.array(bins) >= block, np.array(bins) < block + step + 0.0001)) / len(bins) * 100
        )
        bars.append(percentage)

    return bars


def save_histogram_bars(
    LG_bars: np.ndarray,
    ID: str,
    bar_output_folder: str,
) -> None:
    """Persist a patient's LG histogram bars to an HDF5 file.

    Each patient gets a single ``.hf5`` file containing a ``LG_bars``
    dataset (10-element float32 array, gzip-compressed).

    Args:
        LG_bars: 1-D array of 10 histogram bar percentages.
        ID: Study identifier string (e.g. ``'Study 42'``), used as the
            filename stem.
        bar_output_folder: Output directory (with trailing separator).
    """
    out_file: str = bar_output_folder + ID + ".hf5"
    with h5py.File(out_file, "w") as f:
        dtypef: str = "float32"
        dXy = f.create_dataset("LG_bars", shape=LG_bars.shape, dtype=dtypef, compression="gzip")
        dXy[:] = LG_bars.astype(float)


def load_histogram_bars(
    bar_output_folder: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load pre-computed LG histogram bars for the CPAP success/failure cohorts.

    Expects the ``bar_output_folder`` to contain two sub-directories:
    ``CPAP_success/`` and ``CPAP_failure/``, each holding per-patient
    HDF5 files named ``Study {i}.hf5`` (1-indexed).

    The files are loaded in numeric order so that bar index ``i``
    corresponds to Study ``i+1``.

    Args:
        bar_output_folder: Root directory containing the ``CPAP_success``
            and ``CPAP_failure`` sub-folders (with trailing separator).

    Returns:
        A 2-tuple ``(bars_success, bars_failure)`` where each element
        is a list of 1-D numpy arrays (one per patient).
    """
    bars_success: list[np.ndarray] = []
    bars_failure: list[np.ndarray] = []

    for version in ["CPAP_success", "CPAP_failure"]:
        bar_folder: str = f"{bar_output_folder}{version}/"
        bar_files: list[str] = glob.glob(bar_folder + "*.hf5")

        bars: list[np.ndarray] = []
        for i in range(1, len(bar_files) + 1):
            # Files are named "Study {i}.hf5" -- match by that suffix.
            bar_file: str = [f for f in bar_files if f"Study {i}.hf5" in f][0]
            with h5py.File(bar_file, "r") as f:
                bars.append(f["LG_bars"][:])

        if "success" in version.lower():
            bars_success = bars
        else:
            bars_failure = bars

    return bars_success, bars_failure


def predict_CPAP_SUCCESS_from_bars(
    df: pd.DataFrame,
    bars: np.ndarray,
    bars_success: list[np.ndarray],
    bars_failure: list[np.ndarray],
) -> pd.DataFrame:
    """Predict CPAP response using histogram-based nearest-mean classification.

    For each patient, computes the total absolute error between their
    LG histogram and the average success histogram, then the average
    failure histogram.  The "LG Bar" score is defined as:

        LG Bar = error_to_success - error_to_failure

    A *negative* LG Bar means the patient's histogram is closer to the
    success template (predicting CPAP success); a *positive* value
    predicts failure.

    Args:
        df: Patient-level DataFrame to augment.  A new ``'LG Bar'``
            column is written.
        bars: 2-D array of shape ``(n_patients, 10)`` with per-patient
            histogram bars.
        bars_success: List of 10-element arrays from the success cohort
            (as loaded by ``load_histogram_bars``).
        bars_failure: List of 10-element arrays from the failure cohort.

    Returns:
        The input DataFrame with a new ``'LG Bar'`` column.
    """
    # Compute cohort-average histograms (one 10-element template each).
    bars_s: np.ndarray = np.mean(bars_success, axis=0)
    bars_f: np.ndarray = np.mean(bars_failure, axis=0)

    for i in range(len(df)):
        print(f"Comparing histogram bars: #{i}/{len(df)}", end="\r")
        _, s_score = custom_error(bars[i, :], ref=bars_s)
        _, f_score = custom_error(bars[i, :], ref=bars_f)
        df.loc[df.index[i], "LG Bar"] = s_score - f_score

    return df


def custom_error(
    bar: np.ndarray,
    ref: np.ndarray,
) -> tuple[float, float]:
    """Compute mean and total absolute error between two histogram bar arrays.

    This is deliberately a simple L1 distance -- no weighting by bin
    position.  The total error is used as the prediction score because
    it penalises wide distributional differences more than the mean
    error does.

    Args:
        bar: 1-D array (10 elements) -- the patient's histogram.
        ref: 1-D array (10 elements) -- the reference template.

    Returns:
        A 2-tuple ``(mean_error, total_error)``.
    """
    error: list[float] = []
    for b, r in zip(bar, ref):
        error.append(abs(b - r))

    mean_error: float = np.mean(error)
    total_error: float = sum(error)

    return mean_error, total_error
