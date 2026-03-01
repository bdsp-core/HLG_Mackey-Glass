"""
Per-study EM pipeline — Python equivalent of MATLAB ``mainRun.m``.

Reads a study CSV, iterates over all NREM and REM segments, runs the
EM algorithm on each segment, computes loop gain, reconstructs arousals,
scales the estimated ventilation, and writes everything back into the
DataFrame with the same column structure as the MATLAB output.

Output columns added per segment:
  - ``D_nrem/D_rem``: estimated delay in seconds
  - ``L_nrem/L_rem``: CO2 production rate (fixed at 0.05)
  - ``Alpha_nrem/Alpha_rem``: arousal mixing factor
  - ``G_nrem/G_rem``: estimated gamma (Hill exponent)
  - ``LG_nrem/LG_rem``: computed loop gain
  - ``rmse_Vo``: RMSE between observed and scaled estimated ventilation
  - ``Vmax``: ventilation scaling factor
  - ``Vo_est1/2``, ``Vo_est_scaled1/2``, ``Arousal1/2``: waveforms
    (doubled for overlapping segment windows)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hlg.em.arousal import heaviside
from hlg.em.em_algorithm import run_em_on_segment
from hlg.em.loop_gain_calc import compute_loop_gain


def process_study(
    csv_path: str,
    version: str = "non-smooth",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the EM algorithm on all segments of a single study.

    Direct port of MATLAB ``main_run.m``.

    Args:
        csv_path: Path to the study CSV (produced by the Python SS
            segmentation pipeline).
        version: ``"smooth"`` or ``"non-smooth"`` — selects which drive
            signal column to use.
        verbose: If True, print progress per segment.

    Returns:
        The input DataFrame augmented with EM estimation columns.
    """
    T = pd.read_csv(csv_path)
    Fs = int(T["Fs"].iloc[0])
    w = 5 * Fs  # Arousal window width: 5 seconds
    L = 0.05  # CO2 production rate

    gamma_init = 0.5
    tau_init = 15 * Fs  # 15 seconds * Fs

    # Extract segment boundaries
    nrem_starts = T["nrem_starts"].dropna().values.astype(int)
    nrem_ends = T["nrem_ends"].dropna().values.astype(int)
    rem_starts = T["rem_starts"].dropna().values.astype(int)
    rem_ends = T["rem_ends"].dropna().values.astype(int)
    assert len(nrem_starts) == len(nrem_ends), "Uneven NREM segment indices"
    assert len(rem_starts) == len(rem_ends), "Uneven REM segment indices"

    # Add output columns (all NaN initially)
    n = len(T)
    for col in [
        "D_rem",
        "L_rem",
        "Alpha_rem",
        "LG_rem",
        "G_rem",
        "D_nrem",
        "L_nrem",
        "Alpha_nrem",
        "LG_nrem",
        "G_nrem",
        "rmse_Vo",
        "Vmax",
        "Vo_est1",
        "Vo_est2",
        "Vo_est_scaled1",
        "Vo_est_scaled2",
        "Arousal1",
        "Arousal2",
    ]:
        T[col] = np.nan

    # Process NREM then REM
    for stage_tag, starts, ends in [
        ("nrem", nrem_starts, nrem_ends),
        ("rem", rem_starts, rem_ends),
    ]:
        for s_idx in range(len(starts)):
            # Segment boundaries (MATLAB 1-based → Python 0-based)
            starting = int(starts[s_idx])
            ending = int(ends[s_idx]) - 1
            if starting == 0:
                starting = 1
                ending += 1
            ending = min(ending, n - 1)

            if verbose:
                print(
                    f"  {stage_tag.upper()} segment {s_idx + 1}/{len(starts)} [{starting}:{ending}]",
                    end="\r",
                )

            T_seg = T.iloc[starting : ending + 1].copy().reset_index(drop=True)

            # Run EM on this segment
            upAlpha, upgamma, uptau, Vo_est, h, u_min = run_em_on_segment(T_seg, w, L, gamma_init, tau_init, version)

            # Extract final-iteration parameters
            G_est = float(upgamma[-1])
            D_est = float(uptau[-1]) / Fs  # Convert samples to seconds
            L_est = 0.05
            Alpha_est = float(upAlpha[-1])

            # Compute loop gain
            LG_est = compute_loop_gain(L_est, G_est, u_min)

            # Reconstruct arousal signal for storage
            K_seg = len(T_seg)
            t_ar = np.where(T_seg["arousal_locs"].values != 0)[0]
            t = np.arange(1, K_seg + 1, dtype=np.float64)
            Arousal = np.zeros(K_seg, dtype=np.float64)
            for idx in range(len(t_ar)):
                centre = t_ar[idx] + 1
                square = heaviside(t, centre - w / 2) - heaviside(t - w / 2, centre)
                Arousal += h[idx] * square

            # Scale ventilation
            non_arousal = Arousal == 0
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = T_seg["Ventilation_ABD"].values[non_arousal] / Vo_est[non_arousal]
            valid = np.isfinite(ratio) & (ratio <= 5.0)
            Scale = float(np.mean(ratio[valid])) if np.sum(valid) > 0 else 1.0
            Vd = Vo_est - Arousal
            Vo_est_scaled = Vd * Scale + Arousal

            # Store waveforms (slot 1 or 2 for overlapping windows)
            seg_slice = slice(starting, ending + 1)
            if T["Vo_est1"].iloc[starting : ending + 1].isna().all():
                T.loc[T.index[seg_slice], "Vo_est1"] = Vo_est
                T.loc[T.index[seg_slice], "Vo_est_scaled1"] = Vo_est_scaled
                T.loc[T.index[seg_slice], "Arousal1"] = Arousal
            else:
                T.loc[T.index[seg_slice], "Vo_est2"] = Vo_est
                T.loc[T.index[seg_slice], "Vo_est_scaled2"] = Vo_est_scaled
                T.loc[T.index[seg_slice], "Arousal2"] = Arousal

            # Store scalar parameters and RMSE
            rmse = float(np.sqrt(np.mean((T_seg["Ventilation_ABD"].values - Vo_est_scaled) ** 2)))
            T.loc[T.index[s_idx], "rmse_Vo"] = rmse
            T.loc[T.index[s_idx], "Vmax"] = Scale

            if stage_tag == "nrem":
                T.loc[T.index[s_idx], "D_nrem"] = D_est
                T.loc[T.index[s_idx], "L_nrem"] = L_est
                T.loc[T.index[s_idx], "Alpha_nrem"] = Alpha_est
                T.loc[T.index[s_idx], "G_nrem"] = G_est
                T.loc[T.index[s_idx], "LG_nrem"] = LG_est
            else:
                T.loc[T.index[s_idx], "D_rem"] = D_est
                T.loc[T.index[s_idx], "L_rem"] = L_est
                T.loc[T.index[s_idx], "Alpha_rem"] = Alpha_est
                T.loc[T.index[s_idx], "G_rem"] = G_est
                T.loc[T.index[s_idx], "LG_rem"] = LG_est

        if verbose:
            print(f"  {stage_tag.upper()}: {len(starts)} segments done.          ")

    return T
