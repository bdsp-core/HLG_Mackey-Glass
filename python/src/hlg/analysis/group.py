"""
Group-level EM output extraction and cohort comparison analysis.

This module extracts per-segment loop-gain (LG), controller-gain (gamma),
and circulation-delay (tau) estimates from the EM output CSV files,
organises them by clinical cohort, and saves intermediate results for
downstream box-plot and statistical comparison figures.

Two key functions are provided:

1. ``extract_EM_output`` -- the main extraction pipeline.  For each
   study, it:
   - Reads the EM output CSV and converts sparse SS segment scores
     into dense arrays.
   - Post-processes the EM output (outlier smoothing of LG estimates).
   - Optionally adds arousal annotations from the SS HDF5 output.
   - Matches each recording to its SS pipeline output and validates
     segment alignment.
   - Reconstructs a sample-level LG "hypnogram" and saves it.
   - Collects per-segment parameters (LG, G, D, RMSE, SS, validity)
     and writes per-cohort and per-SS-bin CSV summaries.

2. ``select_highest_LG_block`` -- selects the contiguous block with the
   highest mean LG from a full-night LG array.  Used for swimmer-plot
   visualisations where recordings of different lengths need to be
   aligned to a fixed display window.

Statistical significance testing for cohort comparisons is handled by
``add_statistical_significance`` from ``hlg.analysis.statistics``.

Source: ``EM_output_to_Group_Analysis.py`` (``extract_EM_output``,
``select_highest_LG_block``, ``add_statistical_significance``).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from hlg.core.events import find_events
from hlg.em.loop_gain import create_total_LG_array
from hlg.io.readers import load_sim_output
from hlg.ss.scoring import convert_ss_seg_scores_into_arrays


# ---------------------------------------------------------------------------
# Placeholder imports for functions that may not yet be migrated.
# The original code imported these from EM_output_to_Figures.py; they
# are expected to live in the hlg.em or hlg.io subpackages once fully
# migrated.  For now we import from their canonical new locations.
# ---------------------------------------------------------------------------
# ``post_process_EM_output`` -- outlier smoothing of consecutive LG
# values using a sliding-window median/mean threshold.
# ``add_arousals`` -- attaches arousal annotations from the SS HDF5
# output and retrieves the SS group label.
# ``match_EM_with_SS_output`` -- finds the matching SS pipeline HDF5
# file for a given EM recording by patient tag.
#
# These are assumed to be available via top-level package imports;
# adjust the import path as migration progresses.
from hlg.em.postprocessing import (
    add_arousals,
    match_EM_with_SS_output,
    post_process_EM_output,
)


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_EM_output(
    input_files: list[str],
    interm_folder: str,
    hf5_folder: str,
    version: str,
    dataset: str,
    csv_file: str,
    error_thresh: float = 1.8,
) -> None:
    """Extract per-segment EM estimates, group them, and write CSVs.

    This is the "sequential" extraction pipeline.  It processes each
    study file one at a time, accumulating results into flat arrays
    (per-segment) and a per-SS-bin dictionary, then writes both to CSV.

    The extraction proceeds in the following stages for each study:

    1. **Read & convert** -- load the EM output CSV and expand the
       compact segment-level SS scores into a dense ``SS_score`` column.
    2. **Post-process** -- apply the sliding-window outlier correction to
       the raw LG estimates (``post_process_EM_output``).
    3. **Arousal & group assignment** -- for MGH datasets, load arousal
       annotations from the SS HDF5 and retrieve the patient's SS group
       label.
    4. **SS pipeline matching** -- find the corresponding ``.hf5`` file
       and extract the flow-reduction trace for segment validity checks.
    5. **LG hypnogram** -- reconstruct a sample-level LG trace and save
       it to a CSV for swimmer-plot visualisation.
    6. **Segment loop** -- for each NREM/REM segment, extract the RMSE,
       LG, G, D, and segment SS score.  Validity is defined as having
       >= 5 flow-reduction events within the segment (the minimum needed
       for reliable LG estimation).
    7. **Error thresholding** -- segments with RMSE > ``error_thresh``
       are excluded from all downstream summaries.
    8. **Output** -- write ``all_segments.csv`` (flat) and per-SS-bin
       CSVs (``seg_0.0-0.2.csv``, etc.) to ``interm_folder``.

    Args:
        input_files: List of filesystem paths to EM output CSV files.
        interm_folder: Directory for intermediate CSV output files.
        hf5_folder: Directory containing the SS pipeline ``.hf5`` files.
        version: Analysis version string (e.g. ``'MGH_NREM_OSA_V2'``).
            Controls whether arousals are loaded (only for MGH versions).
        dataset: Dataset identifier (``'mgh'``, ``'redeker'``, ``'rt'``,
            ``'bdsp'``).  Passed to ``match_EM_with_SS_output``.
        error_thresh: RMSE threshold above which segments are excluded
            (default 1.8, empirically determined).
    """
    # Flat arrays for the "all_segments" output.
    LG_data: np.ndarray = np.array([])
    G_data: np.ndarray = np.array([])
    D_data: np.ndarray = np.array([])
    GxD_data: np.ndarray = np.array([])
    SS_data: np.ndarray = np.array([])
    valid_data: np.ndarray = np.array([])

    # Per-SS-bin dictionary.  SS scores are binned into 0.2-wide ranges
    # (0.0-0.2, 0.2-0.4, …, 0.8-1.0) and each bin stores lists of the
    # segment-level parameters.  The bin labels use the convention
    # "seg_{lo}-{hi}_{param}".
    SS_dic: dict[str, list] = {}
    for i in np.arange(0, 10, 2):
        for param in ["SS", "LG", "G", "D", "VV"]:
            SS_dic[f"seg_{i / 10}-{(i + 2) / 10}_{param}"] = []

    # --- Main extraction loop ---
    for idx, input_file in enumerate(input_files):
        # Extract the study number from the file path.
        num = input_file.split("/Study")[-1].split(".csv")[0]
        print(
            f"Extracting Study {num} ({idx + 1}/{len(input_files)}) ..   ",
            end="\r",
        )

        # 1. Read EM output CSV.
        data = pd.read_csv(input_file)

        # 2. Convert sparse segment-level SS scores into a dense column.
        data = convert_ss_seg_scores_into_arrays(data)

        # 3. Post-process EM output (sliding-window LG outlier correction).
        data = post_process_EM_output(data)

        # 4. Extract header fields.
        hdr: dict[str, Any] = {"Study_num": f"Study {num}"}
        for col in ["patient_tag", "Fs", "original_Fs"]:
            hdr[col] = data.loc[0, col]

        # 5. Add arousals (MGH datasets only).
        if "MGH" in version:
            _, hdr["SS group"] = add_arousals(data, version, "mgh", hf5_folder, csv_file)

        # 6. Match with SS pipeline output and validate alignment.
        sim_path, _ = match_EM_with_SS_output(data, dataset, csv_file)
        path = hf5_folder + sim_path + ".hf5"
        SS_df, SS_hdr = load_sim_output(path, ["flow_reductions"])
        assert len(SS_df) > 0.99 * len(data), "matching SS output does not match EM data"

        # 7. Reconstruct sample-level LG hypnogram and save.
        total_LG = create_total_LG_array(data)
        total_LG_df = pd.DataFrame(total_LG, columns=["LG_hypno"])
        hypno_folder = interm_folder + "/hypnograms/"
        os.makedirs(hypno_folder, exist_ok=True)
        out_path = f"{hypno_folder}Study {num}.csv"
        total_LG_df.to_csv(
            out_path,
            header=total_LG_df.columns,
            index=None,
            mode="w+",
        )

        # 8. Iterate over NREM and REM segments to collect parameters.
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
                # Locate the row containing this segment's parameters.
                loc = np.where(data[f"{stage}_starts"] == start)[0][0]

                Errors.append(round(data.loc[loc, "rmse_Vo"], 2))
                Ls.append(data.loc[loc, f"L_{stage}"])
                Vmaxs.append(round(data.loc[loc, "Vmax"], 2))
                LGs.append(data.loc[loc, f"LG_{stage}_corrected"])
                Gs.append(data.loc[loc, f"G_{stage}"])
                Ds.append(data.loc[loc, f"D_{stage}"])

                # Segment validity: require >= 5 flow-reduction events.
                # This threshold ensures the EM had enough breathing
                # oscillations to produce a reliable LG estimate.
                valid_seg.append(len(find_events(SS_df.loc[start:end, "flow_reductions"] > 0)) >= 5)
                SSs_seg.append(data.loc[start, "SS_score"])

        # 9. Apply error threshold -- exclude poorly-fit segments.
        inds = np.array(Errors) < error_thresh
        LGs_arr = np.array(LGs)[inds]
        Gs_arr = np.array(Gs)[inds]
        Ds_arr = np.array(Ds)[inds]
        SSs_arr = np.array(SSs_seg)[inds]
        valids_arr = np.array(valid_seg)[inds]

        # Accumulate into flat arrays.
        LG_data = np.concatenate([LG_data, LGs_arr])
        G_data = np.concatenate([G_data, Gs_arr])
        D_data = np.concatenate([D_data, Ds_arr])
        GxD_data = np.concatenate([GxD_data, Gs_arr * Ds_arr])
        SS_data = np.concatenate([SS_data, SSs_arr])
        valid_data = np.concatenate([valid_data, valids_arr])

        # Bin segments by SS score into 0.2-wide ranges.
        for LG, G, D, SS, VV in zip(LGs_arr, Gs_arr, Ds_arr, SSs_arr, valids_arr):
            for i in np.arange(0, 10, 2):
                ran = f"seg_{i / 10}-{(i + 2) / 10}"
                if SS >= i and SS < i + 0.2:
                    break
            SS_dic[ran + "_SS"].append(SS)
            SS_dic[ran + "_LG"].append(LG)
            SS_dic[ran + "_G"].append(G)
            SS_dic[ran + "_D"].append(D)
            SS_dic[ran + "_VV"].append(VV)

    # --- Write flat "all segments" CSV ---
    sorted_data = [LG_data, G_data, D_data, GxD_data, SS_data, valid_data]
    names = ["LG_data", "G_data", "D_data", "GxD_data", "SS_data", "valid_data"]

    out_path = f"{interm_folder}/all_segments.csv"
    df = pd.DataFrame([], dtype=float)
    for dat, name in zip(sorted_data, names):
        df[name] = dat
    os.makedirs(interm_folder, exist_ok=True)
    df.to_csv(out_path, header=df.columns, index=None, mode="w+")

    # --- Write per-SS-bin CSVs ---
    for i in np.arange(0, 10, 2):
        df = pd.DataFrame([], dtype=float)
        ran = f"seg_{i / 10}-{(i + 2) / 10}"
        for param in ["SS", "LG", "G", "D", "VV"]:
            df[param] = SS_dic[f"{ran}_{param}"]
        df.to_csv(
            f"{interm_folder}{ran}.csv",
            header=df.columns,
            index=None,
            mode="w+",
        )


# ---------------------------------------------------------------------------
# Highest-LG block selection
# ---------------------------------------------------------------------------


def select_highest_LG_block(
    data: np.ndarray,
    block: int,
) -> np.ndarray:
    """Select the contiguous block with the highest mean LG.

    Slides a window of size ``block`` samples across the LG array with
    a stride of 30 minutes (at 10 Hz = 18 000 samples), computes the
    mean LG in each position (ignoring NaN/wake), and returns the block
    with the maximum mean.

    This is used for the "swimmer-plot" visualisation where all
    recordings are displayed at a fixed time-window width (e.g. 8.25 h).
    For recordings longer than the display window, only the most
    clinically relevant portion (highest overall LG burden) is shown.

    The 30-minute stride provides a good balance between computational
    efficiency and spatial resolution -- shifting by less than 30 min
    rarely changes which block wins.

    Args:
        data: 1-D array of per-sample LG values (may contain NaN for
            wake/unscored epochs).
        block: Window size in samples (e.g. ``int(8.25 * 60 * 60 * 10)``
            for 8.25 hours at 10 Hz).

    Returns:
        A 1-D numpy array of length ``block`` containing the LG values
        from the highest-mean window.
    """
    # Stride of 30 minutes at 10 Hz.
    step = int(0.5 * 60 * 60 * 10)
    means: list[float] = []

    for i in range(0, 100, step):
        # Stop if the window extends beyond the data.
        if i * step + block > len(data):
            break
        means.append(np.nanmean(data[i * step : i * step + block]))

    # Select the window position with the highest mean LG.
    loc = np.argmax(means)
    LG_array = data[loc * step : loc * step + block]

    return LG_array
