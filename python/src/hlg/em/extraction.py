"""
Parallel extraction and aggregation of Estimation Model (EM) output.

This module is the main entry point for converting per-patient EM CSV
files into cohort-level aggregate tables.  It:

1. Reads each patient's EM CSV output.
2. Converts SS segment scores into aligned arrays.
3. Applies LG outlier smoothing (``post_process_EM_output``).
4. Links to the matching SS HDF5 output for arousal and flow-reduction
   data.
5. Reconstructs the sample-level LG trace and computes the LG
   histogram.
6. Extracts per-segment metrics (LG, G, D, SS score, etc.) and filters
   by model fit quality (RMSE threshold).
7. Aggregates results across all patients into a single CSV and per-SS-
   score-bin CSVs.

Heavy lifting is parallelised across CPU cores via ``multiprocessing``.

Source: ``EM_output_extraction.py`` (parallel version).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from hlg.core.events import find_events
from hlg.em.postprocessing import (
    add_arousals,
    match_EM_with_SS_output,
    post_process_EM_output,
)
from hlg.ss.scoring import convert_ss_seg_scores_into_arrays
from hlg.io.readers import load_sim_output
from hlg.em.loop_gain import create_total_LG_array
from hlg.em.histograms import compute_histogram
from hlg.config import config


def extract_EM_output(
    input_files: list[str],
    interm_folder: str,
    hf5_folder: str,
    version: str,
    dataset: str,
    csv_file: str,
    bar_folder: str,
) -> None:
    """Parallel extraction of EM output across all patients in a cohort.

    Spawns a process pool that calls ``process_EM_output`` for each
    patient file, then collects and concatenates the per-segment metrics
    into a single ``all_segments.csv`` table.  Additionally, segments are
    stratified by SS score into 0.2-wide bins and saved as separate CSVs
    for downstream per-severity analyses.

    The function also builds a ``SS_dic`` dictionary that bins every
    valid segment by its Self-Similarity score (in 0.2 increments from
    0.0 to 1.0).  Each bin gets its own CSV with columns for SS score,
    LG, controller gain (G), plant gain (D), validity flag, and sleep
    stage.

    Args:
        input_files: List of filesystem paths to per-patient EM CSV
            output files (one per patient).
        interm_folder: Directory for intermediate/aggregate output
            files (with trailing separator).
        hf5_folder: Directory containing the SS ``.hf5`` output files.
        version: Pipeline version string (e.g. ``'MGH'``,
            ``'Simulation'``, ``'CPAP'``).
        dataset: Dataset identifier (``'mgh'``, ``'redeker'``, etc.).
        csv_file: Path to the cohort-level CSV metadata file.
        bar_folder: Output directory for LG histogram bar HDF5 files.
    """
    # Accumulators for cohort-level arrays.
    LG_data: list[float] = []
    G_data: list[float] = []
    D_data: list[float] = []
    GxD_data: list[float] = []
    SS_data: list[float] = []
    valid_data: list[float] = []
    ID_data: list[float] = []
    Stages_data: list[float] = []

    # Per-SS-score-bin accumulators.  The EM's Self-Similarity score
    # runs from 0 to 1; we bin in 0.2 increments (5 bins total).
    SS_dic: dict[str, list[float]] = {}
    for i in np.arange(0, 10, 2):
        for param in ["SS", "LG", "G", "D", "VV", "St"]:
            SS_dic[f"seg_{i / 10}-{(i + 2) / 10}_{param}"] = []

    # Launch parallel workers -- one per patient file.
    num_workers: int = cpu_count()
    pool: Pool = Pool(num_workers)

    process_args: list[tuple] = [
        (input_file, interm_folder, hf5_folder, version, dataset, csv_file, bar_folder) for input_file in input_files
    ]

    for i, (data_dic) in enumerate(pool.starmap(process_EM_output, process_args)):
        num = process_args[i][1]

        # Concatenate this patient's per-segment arrays to the cohort
        # accumulators.
        LG_data = np.concatenate([LG_data, data_dic["LGs"]])
        G_data = np.concatenate([G_data, data_dic["Gs"]])
        D_data = np.concatenate([D_data, data_dic["Ds"]])
        GxD_data = np.concatenate([GxD_data, data_dic["Gs"] * data_dic["Ds"]])
        SS_data = np.concatenate([SS_data, data_dic["SSs"]])
        valid_data = np.concatenate([valid_data, data_dic["valids"]])
        ID_data = np.concatenate([ID_data, np.ones(len(data_dic["valids"])) * int(num.strip())])
        Stages_data = np.concatenate([Stages_data, data_dic["Stages"]])

        # Bin each segment by its SS score into one of the 0.2-wide
        # bins for stratified downstream analysis.
        for LG, G, D, SS, VV, ST in zip(
            data_dic["LGs"], data_dic["Gs"], data_dic["Ds"], data_dic["SSs"], data_dic["valids"], data_dic["Stages"]
        ):
            for j in np.arange(0, 10, 2):
                ran: str = f"seg_{j / 10}-{(j + 2) / 10}"
                if SS >= j and SS < j + 0.2:
                    break
            SS_dic[ran + "_SS"].append(SS)
            SS_dic[ran + "_LG"].append(LG)
            SS_dic[ran + "_G"].append(G)
            SS_dic[ran + "_D"].append(D)
            SS_dic[ran + "_VV"].append(VV)
            SS_dic[ran + "_St"].append(ST)

    pool.close()
    pool.join()

    # ── Write cohort-level aggregate CSV ─────────────────────────────
    sorted_data: list[np.ndarray] = [
        LG_data,
        G_data,
        D_data,
        GxD_data,
        SS_data,
        valid_data,
        ID_data,
        Stages_data,
    ]
    names: list[str] = [
        "LG_data",
        "G_data",
        "D_data",
        "GxD_data",
        "SS_data",
        "valid_data",
        "ID_data",
        "Stages_data",
    ]
    out_path: str = f"{interm_folder}/all_segments.csv"

    df: pd.DataFrame = pd.DataFrame([], dtype=float)
    for dat, name in zip(sorted_data, names):
        df[name] = dat

    os.makedirs(interm_folder, exist_ok=True)
    df.to_csv(out_path, header=df.columns, index=None, mode="w+")

    # ── Write per-SS-score-bin CSVs ──────────────────────────────────
    for i in np.arange(0, 10, 2):
        df = pd.DataFrame([], dtype=float)
        ran = f"seg_{i / 10}-{(i + 2) / 10}"
        for param in ["SS", "LG", "G", "D", "VV", "St"]:
            df[param] = SS_dic[f"{ran}_{param}"]
        df.to_csv(
            f"{interm_folder}{ran}.csv",
            header=df.columns,
            index=None,
            mode="w+",
        )


def process_EM_output(
    input_file: str,
    interm_folder: str,
    hf5_folder: str,
    version: str,
    dataset: str,
    csv_file: str,
    bar_folder: str,
) -> dict[str, np.ndarray]:
    """Process a single patient's EM CSV and extract per-segment metrics.

    This is the worker function called in parallel by
    ``extract_EM_output``.  For one patient it:

    1. Reads the EM CSV and reconstructs segment-level arrays.
    2. Applies LG outlier smoothing.
    3. Links to the matching SS HDF5 output.
    4. Reconstructs the sample-level LG hypnogram and computes the LG
       histogram.
    5. Iterates over every NREM and REM segment to extract LG, G, D,
       model-fit error, SS score, and whether the segment contains
       >= 5 respiratory events (the ``valid`` flag).
    6. Filters out segments whose RMSE exceeds the configured error
       threshold.

    Args:
        input_file: Path to the patient's EM CSV file.
        interm_folder: Directory for intermediate output (hypnograms).
        hf5_folder: Directory containing SS ``.hf5`` files.
        version: Pipeline version string.
        dataset: Dataset identifier.
        csv_file: Path to cohort CSV metadata.
        bar_folder: Output directory for LG histogram bar files.

    Returns:
        A dict with keys ``'LGs'``, ``'Gs'``, ``'Ds'``, ``'SSs'``,
        ``'valids'``, ``'Stages'`` -- each a 1-D numpy array with one
        element per segment that passed the error threshold filter.
    """
    data: pd.DataFrame = pd.read_csv(input_file)

    # Convert segment-level SS scores back into aligned per-sample
    # arrays so they can be joined with the EM data.
    data = convert_ss_seg_scores_into_arrays(data)

    # Smooth outlier LG values using a sliding-window median/mean
    # filter (populates LG_{stage}_corrected columns).
    data = post_process_EM_output(data)

    # Extract the study number from the filename.  Handles both
    # Unix ('/') and Windows ('\\') path separators.
    num: str = input_file.split("/Study")[-1].split("\\Study")[-1].split(".csv")[0].strip()

    hdr: dict[str, Any] = {"Study_num": f"Study {num}"}
    for col in ["patient_tag", "Fs", "original_Fs"]:
        hdr[col] = data.loc[0, col]

    # For the MGH cohort, merge arousal annotations and extract the
    # clinical group assignment.
    if "MGH" in version:
        _, hdr["SS group"] = add_arousals(data, version, "mgh", hf5_folder, csv_file)

    # ── Link to matching SS output ────────────────────────────────────
    sim_path, _ = match_EM_with_SS_output(data, dataset, csv_file)
    path: str = hf5_folder + sim_path + ".hf5"

    # Try loading the flow-reduction channel first; fall back to the
    # older (apnea + cpap_start) schema if not available.
    failed: bool = False
    try:
        SS_df, SS_hdr = load_sim_output(path, ["flow_reductions"])
    except Exception:
        failed = True

    if failed or "flow_reductions" not in SS_df.columns:
        SS_df, SS_hdr = load_sim_output(path, ["apnea", "cpap_start"])

    assert len(SS_df) > 0.99 * len(data), "matching SS output does not match EM data"

    # For CPAP studies, truncate both DataFrames at the CPAP start
    # so only the diagnostic (pre-treatment) portion is analysed.
    if "CPAP" in version:
        data = data.loc[: SS_hdr["cpap_start"], :].copy()
        SS_df = SS_df.loc[: SS_hdr["cpap_start"], :].copy()

    # ── Build sample-level LG trace and histogram ────────────────────
    total_LG: np.ndarray = create_total_LG_array(data)
    total_LG_df: pd.DataFrame = pd.DataFrame(total_LG, columns=["LG_hypno"])

    hypno_folder: str = os.path.join(interm_folder, "hypnograms/")
    os.makedirs(hypno_folder, exist_ok=True)
    out_path: str = os.path.join(hypno_folder, f"Study {num}.csv")
    total_LG_df.to_csv(out_path, header=total_LG_df.columns, index=None, mode="w+")

    data["total_LG"] = total_LG
    compute_histogram(data, hdr, bar_folder)

    # ── Per-segment metric extraction ────────────────────────────────
    Errors: list[float] = []
    Vmaxs: list[float] = []
    LGs: list[float] = []
    Gs: list[float] = []
    Ds: list[float] = []
    Ls: list[float] = []
    SSs_seg: list[float] = []
    valid_seg: list[bool] = []
    Stages: list[str] = []

    for stage in ["nrem", "rem"]:
        starts: np.ndarray = data[f"{stage}_starts"].dropna().values.astype(int)
        ends: np.ndarray = data[f"{stage}_ends"].dropna().values.astype(int)

        for start, end in zip(starts, ends):
            # For CPAP studies, skip segments that extend beyond the
            # CPAP-start boundary.
            if "CPAP" in version and end > SS_hdr["cpap_start"]:
                continue

            loc: int = np.where(data[f"{stage}_starts"] == start)[0][0]

            Errors.append(round(data.loc[loc, "rmse_Vo"], 2))
            Ls.append(data.loc[loc, f"L_{stage}"])
            Vmaxs.append(round(data.loc[loc, "Vmax"], 2))
            LGs.append(data.loc[loc, f"LG_{stage}_corrected"])
            Gs.append(data.loc[loc, f"G_{stage}"])
            Ds.append(data.loc[loc, f"D_{stage}"])

            # A segment is "valid" if it contains at least 5
            # respiratory events (flow reductions > 0).  Segments
            # with fewer events lack sufficient signal for a
            # reliable LG fit.
            valid_seg.append(len(find_events(SS_df.loc[start:end] > 0)) >= 5)
            SSs_seg.append(data.loc[start, "SS_score"])
            Stages.append(stage)

    # ── Quality filter: discard poor model fits ──────────────────────
    # Segments with RMSE above the configured threshold are excluded
    # from cohort-level analyses.  The threshold (default 1.8) was
    # empirically calibrated to reject fits distorted by artifacts
    # while retaining >=90 % of segments in clean recordings.
    error_thresh: float = config.error_threshold
    inds: np.ndarray = np.array(Errors) < error_thresh

    LGs_arr: np.ndarray = np.array(LGs)[inds]
    Gs_arr: np.ndarray = np.array(Gs)[inds]
    Ds_arr: np.ndarray = np.array(Ds)[inds]
    SSs_arr: np.ndarray = np.array(SSs_seg)[inds]
    valids_arr: np.ndarray = np.array(valid_seg)[inds]
    Stages_arr: np.ndarray = np.array(Stages)[inds]

    data_dic: dict[str, np.ndarray] = {
        "LGs": LGs_arr,
        "Gs": Gs_arr,
        "Ds": Ds_arr,
        "SSs": SSs_arr,
        "valids": valids_arr,
        "Stages": Stages_arr,
    }

    return data_dic
