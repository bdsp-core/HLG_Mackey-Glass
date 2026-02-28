"""
Per-segment EM output figure generation.

This module produces a detailed diagnostic figure for each 8-minute
sleep segment that the Estimation Model (EM) has analysed.  The figure
is a multi-panel "strip chart" that stacks the following signals
vertically on a shared time axis:

1. **Manual arousals** -- horizontal bars at scored arousal locations.
2. **Estimated arousals** -- bars at EM-inferred arousal locations.
3. **Manual respiratory events** -- colour-coded by event type (RERA,
   hypopnea, mixed/central/obstructive apnea).
4. **Disturbance (U)** -- the ventilatory drive disturbance signal,
   combining the flow-reduction index with the arousal contribution.
5. **Abdominal effort** -- raw respiratory inductance plethysmography.
6. **SpO2** -- pulse oximetry trace with artefact clamping.
7. **Ventilation** -- observed (black) and EM-modelled (blue) traces,
   with arousal-related ventilation bursts highlighted.
8. **Modelled CO2** -- the EM's internal CO2 state variable, shown both
   instantaneously and shifted by the circulation delay tau.

A legend box at the bottom summarises the estimated parameters:
LG, gamma (controller gain), tau (delay), v_max, L (metabolic rate),
alpha (arousal fraction), and RMSE.

Source: ``EM_output_to_Figures.py`` -> ``plot_EM_output_per_segment``
"""

from __future__ import annotations

import os
from itertools import groupby
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from hlg.core.events import find_events
from hlg.io.readers import load_sim_output

from hlg.em.postprocessing import match_EM_with_SS_output, post_process_estimated_arousals


def plot_EM_output_per_segment(
    data_og: pd.DataFrame,
    hdr: dict[str, Any],
    metric_map: dict[str, str],
    start: int,
    end: int,
    stage: str,
    arousal_dur: float,
    out_folder: str,
    Ut_smooth: str,
    hf5_folder: str,
    csv_file: str,
    dataset: str = "mgh",
) -> None:
    """Create a detailed strip-chart figure for one EM-analysed segment.

    The figure provides a visual "audit trail" that lets the clinician
    or researcher verify the quality of the EM fit by comparing the
    observed ventilation trace with the model's reconstruction, and by
    inspecting the inferred arousal timing and CO2 dynamics.

    The ``data_og`` DataFrame should be the *full-night* EM output
    (after SS score conversion and LG post-processing).  The segment
    is sliced from ``start`` to ``end`` (1-based indexing, consistent
    with the original pipeline convention).

    **CO2 computation** -- the EM models CO2 as a first-order ODE driven
    by metabolic production (L) and cleared by delayed ventilation:

        dx/dt = L - v_o(t - tau) * x(t)

    where x(t) is the CO2 level, v_o is the estimated ventilation, and
    tau is the circulation delay.  The ODE is integrated with forward
    Euler at the segment's sampling rate.

    Args:
        data_og: Full-night EM output DataFrame.
        hdr: Recording header dict (must contain ``'Fs'``,
            ``'Study_num'``, ``'patient_tag'``).
        metric_map: Mapping of clinical metric column names to display
            labels (e.g. ``{'Sex': 'Sex', 'age': 'Age', ...}``).
        start: Segment start index (1-based).
        end: Segment end index (1-based, exclusive).
        stage: Sleep stage label (``'nrem'`` or ``'rem'``).
        arousal_dur: Arousal duration in seconds (used by
            ``post_process_estimated_arousals``).
        out_folder: Directory for saving the figure PNG.
        Ut_smooth: Smoothing variant (``'smooth'`` or ``'non-smooth'``)
            controlling which disturbance trace column is used.
        hf5_folder: Directory of SS ``.hf5`` files.
        csv_file: Path to the metadata CSV for patient lookup.
        dataset: Dataset identifier (default ``'mgh'``).

    Side Effects:
        Saves a high-DPI PNG to ``out_folder`` (filename encodes the
        segment parameters).  If the file already exists, returns
        immediately (skip-if-exists pattern for incremental re-runs).
    """
    # --- Slice the segment from the full-night DataFrame ---
    # The original code uses 1-based indexing; .loc[start-1:end-2]
    # converts to the correct 0-based inclusive slice.
    data: pd.DataFrame = data_og.loc[start - 1 : end - 2].reset_index(drop=True).copy()
    fs: float = hdr["Fs"]
    fz: int = 14

    # --- Match with SS pipeline output for segment validity ---
    sim_path, _ = match_EM_with_SS_output(data_og, dataset, csv_file)
    path: str = hf5_folder + sim_path + ".hf5"
    SS_df, SS_hdr = load_sim_output(path, ["apnea"])
    assert len(SS_df) == len(data_og), "matching SS output does not match EM data"
    SS_df = SS_df.loc[start - 1 : end - 2].reset_index(drop=True).copy()
    # A segment needs >= 4 apnea events to be considered valid.
    min_events: bool = len(find_events(SS_df.apnea > 0)) >= 4

    # --- Extract segment-level EM parameters ---
    loc: int = np.where(data_og[f"{stage}_starts"] == start)[0][0]
    SS_seg: float = round(data.loc[0, "SS_score"], 2)
    Error: float = round(data_og.loc[loc, "rmse_Vo"], 2)
    Vmax: float = round(data_og.loc[loc, "Vmax"], 2)
    LG: float = round(data_og.loc[loc, f"LG_{stage}_corrected"], 2)
    G: float = data_og.loc[loc, f"G_{stage}"]
    Delay: float = data_og.loc[loc, f"D_{stage}"]
    L: float = data_og.loc[loc, f"L_{stage}"]
    Alpha: float = data_og.loc[loc, f"Alpha_{stage}"]

    # Mark excluded segments with a prefix in the filename.
    ex: str = "" if min_events else "Exc. "
    tag: str = f"{stage} {ex}LG({LG}) g({G}) d({Delay}) a({Alpha}) -{SS_seg}- {start}-{end}.png"
    out_path: str = out_folder + tag
    if os.path.exists(out_path):
        return

    # --- Post-process estimated arousals ---
    # Separates the modelled ventilation into drive (Vd) and arousal
    # components, and applies the scaling correction.
    data = post_process_estimated_arousals(data, arousal_dur * fs)

    # --- Extract signal arrays ---
    ABD: np.ndarray = data.ABD.values
    Vo: np.ndarray = data.Ventilation_ABD.values
    Vd_est_scaled: np.ndarray = data.Vd_est_scaled.values
    Vo_est_scaled: np.ndarray = data.Vo_est_corrected.values
    a_locs: np.ndarray = data.Aest_loc.values.astype(float)
    di_abd: np.ndarray = data.d_i_ABD.values if Ut_smooth == "non-smooth" else data.d_i_ABD_smooth.values
    # Combined disturbance: flow-reduction index plus arousal fraction.
    Disturbance: np.ndarray = di_abd + Alpha * (1 - di_abd)
    spo2: np.ndarray = data.SpO2.values.astype(float)

    # Scored labels.
    y_tech: np.ndarray = data.Apnea.values
    _y_algo: np.ndarray = data.Apnea_algo.values
    arousals: np.ndarray = data.Arousals.values if "Arousals" in data.columns else np.zeros(len(data))

    # =====================================================================
    # CO2 modelling via forward Euler integration
    # =====================================================================
    # The EM's internal model treats chemosensory drive as proportional
    # to CO2 level x(t), which evolves according to:
    #     dx/dt = L - v_o(t - tau) * x(t)
    # where L is the constant metabolic CO2 production rate, v_o is the
    # estimated ventilation, and tau is the plant delay (circulation time
    # from lung to chemoreceptor).
    dt: float = 1.0 / fs
    delay_steps: int = int(Delay * fs)
    initial_CO2: float = 1.0
    CO2: np.ndarray = np.zeros(len(Vo_est_scaled))
    CO2[0] = initial_CO2

    for i in range(1, len(Vo_est_scaled)):
        # Use the ventilation value from tau seconds ago.
        if i - delay_steps >= 0:
            v_delayed = Vo_est_scaled[i - delay_steps]
        else:
            v_delayed = Vo_est_scaled[0]
        dCO2 = L - v_delayed * CO2[i - 1]
        CO2[i] = CO2[i - 1] + dCO2 * dt

    # =====================================================================
    # Figure construction
    # =====================================================================
    fig: plt.Figure = plt.figure(figsize=(18, 8))
    ax: plt.Axes = fig.add_subplot(111)
    label_txt_dic: dict[str, Any] = {
        "fontsize": fz,
        "ha": "right",
        "va": "center",
    }

    # ----- Abdominal effort trace -----
    # Normalised to span a fixed vertical range (``factor * 2``).
    factor: int = 4
    maxi: float = np.nanmax(ABD) - np.nanmin(ABD)
    ABD_n: np.ndarray = ABD / maxi * factor * 2
    ax.plot(ABD_n, c="k", lw=0.5, alpha=0.75)
    # Peak/trough amplitudes for layout reference.
    max_y: float = np.median(ABD_n[find_peaks(ABD_n, distance=fs)[0]])
    min_y: float = -np.median(ABD_n[find_peaks(-ABD_n, distance=fs)[0]])
    ax.text(-10 * fs, 0, "Abdominal effort", **label_txt_dic)

    # ----- Arousal traces -----
    arousals[arousals != 1] = np.nan
    a_locs[a_locs != 1] = np.nan
    offset: float = max_y + 8.5
    ax.axhline(offset, c="k", lw=0.5, linestyle="dashed")
    ax.axhline(offset - 1, c="k", lw=0.5, linestyle="dashed")
    ax.plot(arousals * offset, c="k", lw=4)
    ax.plot(a_locs * offset - 1, c="k", lw=4)
    ax.text(-10 * fs, offset, "Manual arousals", **label_txt_dic)
    ax.text(-10 * fs, offset - 1, "Estimated arousals", **label_txt_dic)

    # ----- Manual respiratory event labels -----
    # Colour coding: 1=RERA(uncoloured), 2=hypopnea(blue),
    # 3=mixed(green), 4=central(cyan), 5=obstructive(magenta),
    # 6=RERA(red), 7=hypopnea(blue).
    label_color = [None, "b", "g", "c", "m", "r", None, "b"]
    offset = max_y + 6.5
    ax.axhline(offset, c="k", lw=0.5, linestyle="dashed")
    ax.text(-10 * fs, offset, "Manual resp. events", **label_txt_dic)

    loc_counter: int = 0
    for i, j in groupby(y_tech):
        len_j: int = len(list(j))
        if np.isfinite(i) and label_color[int(i)] is not None:
            ax.plot(
                [loc_counter, loc_counter + len_j],
                [offset] * 2,
                c=label_color[int(i)],
                lw=3,
            )
        loc_counter += len_j

    # ----- Disturbance (U) trace -----
    offset = max_y + 3.75
    factor = 2
    ax.text(-2 * fs, offset + factor, "1", fontsize=fz - 5, ha="right", va="center")
    ax.text(-2 * fs, offset, "0", fontsize=fz - 5, ha="right", va="center")
    # Raw disturbance index (semi-transparent) and combined disturbance.
    ax.plot(di_abd * factor + offset, c="k", lw=1, alpha=0.25)
    ax.plot(Disturbance * factor + offset, c="k", lw=2, alpha=0.5)
    ax.text(-10 * fs, offset + 1, "Disturbance ($U$)", **label_txt_dic)
    # Shaded background for the [0, 1] range.
    ax.fill_between([0, len(data)], offset, offset + factor, fc="k", alpha=0.1)

    # ----- SpO2 trace -----
    offset = min_y - 8.5
    factor = 5
    # Clamp implausible values (< 80 %) to NaN.
    spo2[np.less(spo2, 80, where=np.isfinite(spo2))] = np.nan
    if any(np.isfinite(spo2)):
        spo2_n: np.ndarray = (spo2 - np.nanmin(spo2)) / (np.nanmax(spo2) - np.nanmin(spo2)) * factor
        ax.plot(spo2_n + offset, c="y", lw=1)
    ax.text(-10 * fs, offset + factor / 2, "SpO$_{2}$", **label_txt_dic)
    maxi_spo2, mini_spo2 = ("NaN", "NaN") if np.all(np.isnan(spo2)) else (int(np.nanmax(spo2)), int(np.nanmin(spo2)))
    ax.text(-2 * fs, offset + factor, f"{maxi_spo2}%", fontsize=fz - 5, ha="right", va="top")
    ax.text(-2 * fs, offset, f"{mini_spo2}%", fontsize=fz - 5, ha="right", va="bottom")

    # ----- Ventilation traces (observed + modelled) -----
    offset_V: float = min_y - 17
    factor = 5
    maxi_v: float = max(np.nanmax(Vo), np.nanmax(Vo_est_scaled))
    mini_v: float = 0
    Vo_n: np.ndarray = (Vo - mini_v) / (maxi_v - mini_v) * factor
    Vo_est_scaled_n: np.ndarray = (Vo_est_scaled - mini_v) / (maxi_v - mini_v) * factor
    Vd_est_scaled_n: np.ndarray = (Vd_est_scaled - mini_v) / (maxi_v - mini_v) * factor
    # Align zero baseline by subtracting the near-minimum.
    Vo_est_scaled_n -= np.nanquantile(Vo_est_scaled_n, 0.002)
    Vd_est_scaled_n -= np.nanquantile(Vd_est_scaled_n, 0.002)
    Vo_est_scaled_n[Vo_est_scaled_n < 0] = 0
    Vd_est_scaled_n[Vd_est_scaled_n < 0] = 0

    # Separate arousal-driven ventilation bursts from drive-only trace.
    Var_est_scaled_n: np.ndarray = np.array(Vo_est_scaled_n)
    same = Vd_est_scaled_n == Var_est_scaled_n
    Vd_est_scaled_n[~same] = np.nan
    # Stitch transitions so the arousal burst lines connect smoothly.
    for st, ed in find_events(np.isfinite(Var_est_scaled_n)):
        Var_est_scaled_n[st] = Vd_est_scaled_n[st]
        Var_est_scaled_n[ed : ed + 2] = Vd_est_scaled_n[ed : ed + 2]

    ax.plot(Vo_n + offset_V, c="k", lw=2, alpha=1)
    ax.plot(Var_est_scaled_n + offset_V, c="b", linestyle="solid", lw=2)
    ax.plot(Vd_est_scaled_n + offset_V, c="b", linestyle="solid", lw=2)
    ax.text(-10 * fs, offset_V + factor / 2, "Ventilation", **label_txt_dic)
    ax.axhline(offset_V, c="k", lw=0.5, linestyle="dashed")
    ax.axhline(offset_V + factor, c="k", lw=0.5, linestyle="dashed")

    # Shade arousal locations in the ventilation panel.
    yl = [offset_V] * 2
    yu = [max(np.nanmax(Vo_n), np.nanmax(Vo_est_scaled_n))] * 2
    for st, ed in find_events(data["Aest_loc"] == 1):
        ax.fill_between([st, ed], yl, yu + offset_V, color="k", alpha=0.1, ec="none")

    # ----- CO2 trace -----
    offset_CO2: float = min_y - 12
    factor_CO2: int = 3

    # Mask the initial calibration period (2.5 x tau) where the Euler
    # integration has not yet converged.
    mask_duration: float = Delay * 2.5
    mask_steps: int = int(mask_duration * fs)

    # Normalise CO2 using post-calibration values.
    CO2_n: np.ndarray = (
        (CO2 - np.nanmin(CO2[mask_steps:])) / (np.nanmax(CO2[mask_steps:]) - np.nanmin(CO2[mask_steps:])) * factor_CO2
    )

    # Non-delayed (instantaneous) CO2.
    CO2_non_delayed: np.ndarray = np.copy(CO2_n)
    CO2_non_delayed[:mask_steps] = np.nan

    # Delayed CO2: shifted by tau to show the signal as it arrives at
    # the peripheral chemoreceptor.
    CO2_delayed: np.ndarray = np.roll(CO2_n, delay_steps)
    CO2_delayed[: delay_steps + mask_steps] = np.nan

    ax.plot(
        CO2_non_delayed + offset_CO2,
        c="k",
        lw=1,
        linestyle="dashed",
        alpha=0.1,
        label="CO2 (non-shifted)",
    )
    ax.plot(
        CO2_delayed + offset_CO2,
        c="b",
        lw=1.5,
        linestyle="dashed",
        alpha=0.75,
        label="CO2 (shifted by tau)",
    )
    ax.text(-10 * fs, offset_CO2 + factor_CO2 / 2, "Modeled CO2 (x)", **label_txt_dic)

    # Double-headed arrow showing the delay tau.
    first_CO2_index: int = np.where(np.isfinite(CO2_delayed))[0][0]
    max_CO2_value: float = CO2_delayed[first_CO2_index] + 0.25
    left_index: int = first_CO2_index - delay_steps
    ax.annotate(
        "",
        xy=(first_CO2_index, max_CO2_value + offset_CO2),
        xytext=(left_index, max_CO2_value + offset_CO2),
        arrowprops={"arrowstyle": "<->", "color": "k", "lw": 1},
    )
    mid_x: float = (first_CO2_index + left_index) / 2
    mid_y: float = max_CO2_value + offset_CO2 + 0.25
    va: str = "bottom"
    if mid_y > 4:
        mid_y = max_CO2_value + offset_CO2 - 0.25
        va = "top"
    ax.text(mid_x, mid_y, "$\\tau$", fontsize=fz - 2, ha="center", va=va)

    # Shade the calibration region.
    yu_co2 = [offset_CO2] * 2
    yl_co2 = [offset_CO2 + factor_CO2] * 2
    ax.fill_between([0, left_index], yl_co2, yu_co2, color="k", alpha=0.1, ec="none")
    ax.text(
        fs,
        offset_CO2 + 0.25,
        "Calibrating..",
        fontsize=fz - 4,
        fontstyle="italic",
        ha="left",
        va="bottom",
    )

    # ----- Final layout -----
    ax.set_xlim([-5, len(data) + 5])
    ax.axis("off")

    # --- Legend box: global metrics + SS ---
    len_x: int = len(data)
    metric_map["SS"] = "SS"
    hdr["SS"] = SS_seg
    dx: int = len_x // 30
    y_legend: float = max_y + 10
    for i, metric in enumerate(metric_map.values()):
        x = 30 + dx * i * 2
        ax.text(x + dx / 2, y_legend, metric, fontsize=fz, ha="center", va="bottom")
        ax.text(x + dx / 2, y_legend - 0.25, hdr[metric], fontsize=fz - 2, ha="center", va="top")

    # Event-type legend (top right).
    event_types = ["RERA", "Hypopnea", "Mixed\napnea", "Central\napnea", "Obstructive\napnea"]
    label_colors_legend = ["r", "m", "c", "g", "b"]
    dx = len_x // 25
    for i, (color, e_type) in enumerate(zip(label_colors_legend, event_types)):
        x = len_x - 20 - dx * i * 2
        ax.plot([x, x - dx], [y_legend - 0.5] * 2, c=color, lw=3)
        ax.text(x - dx / 2, y_legend, e_type, fontsize=fz - 2, ha="center", va="bottom")

    # Ventilation line legend (below ventilation panel).
    lines = ["Observed", "Modeled"]
    line_colors = ["k", "b"]
    line_styles = ["solid", "solid"]
    dx = int(len_x // 22.5)
    offset_legend: float = offset_V - 1
    for i, (line_label, c_legend, ls) in enumerate(zip(lines, line_colors, line_styles)):
        x = dx * i * 1.5
        ax.plot([x, x + dx], [offset_legend + 0.25] * 2, c=c_legend, lw=2, linestyle=ls)
        ax.text(x + dx / 2, offset_legend, line_label, fontsize=fz - 2, ha="center", va="top")

    # EM parameter summary (centre bottom).
    scores = [
        r"$\bf{LG}$",
        "$\\gamma$",
        "$\\tau$",
        "$v_{max}$",
        "$L$",
        "$\\alpha$",
        "RMSE",
    ]
    values = [
        r"$\bf{" + str(LG) + "}$",
        G,
        f"{Delay} sec",
        Vmax,
        L,
        Alpha,
        Error,
    ]
    for i, (tag_label, val) in enumerate(zip(scores, values)):
        minus = np.floor(len(scores) / 2)
        x = len_x / 2 + ((i - minus) * 2) * dx
        ax.text(x, offset_legend - 0.5, tag_label, fontsize=fz, ha="center", va="bottom")
        ax.text(x, offset_legend - 0.75, val, fontsize=fz - 2, ha="center", va="top")

    # Duration scale bar (bottom right).
    sec: int = 30
    ax.plot([len_x - sec * fs, len_x], [offset_legend + 0.25] * 2, color="k", lw=1.5)
    ax.plot([len_x - sec * fs] * 2, [offset_legend, offset_legend + 0.5], color="k", lw=1.5)
    ax.plot([len_x] * 2, [offset_legend, offset_legend + 0.5], color="k", lw=1.5)
    ax.text(
        len_x - sec / 2 * fs,
        offset_legend,
        f"{sec} sec\n({stage})",
        color="k",
        fontsize=fz - 2,
        ha="center",
        va="top",
    )

    # --- Save figure ---
    plt.savefig(out_path, dpi=900)
    plt.close()
