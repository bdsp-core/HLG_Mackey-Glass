#!/usr/bin/env python
"""
Reproduce the paper figures from the 4 example patients.

Generates:
  - Figure 1: Per-segment EM fit panels (re-runs EM live) for all 4 patients
  - Figure 2: Full-night overview with LG hooks for all 4 patients

Figure 1 panel mapping (from paper):
  Panel A: Study 99 (High CAI), NREM seg 1,  rows 9000-13800,  LG~1.87
  Panel B: Study 99 (High CAI), NREM seg 14, rows 82200-87000, LG~3.85
  Panel C: Study 97 (HLG OSA),  NREM seg 8,  rows 64800-69600, LG~0.97
  Panel D: Study 97 (HLG OSA),  NREM seg 20, rows 158400-163200, LG~1.10
  Panel E: Study  5 (NREM OSA), NREM seg 6,  rows 54600-59400, LG~0.11
  Panel F: Study  7 (HLG OSA),  NREM seg 14, rows 75300-80100, LG~0.39

Data required in data/paper_examples/:
  - figure1_studies.csv          (metadata mapping)
  - NREM_OSA_Study_5.csv         (patient d1c690da)
  - HLG_OSA_Study_7.csv          (patient 7b1e2d31)
  - HLG_OSA_Study_97.csv         (patient 39fc3416)
  - high_CAI_Study_99.csv        (patient 4e504ee4)

Optional for full-night figures (H5 files in data/paper_examples/h5/):
  - sub-S0001111922082_ses-1.h5  (Study 5)
  - sub-S0001111985952_ses-1.h5  (Study 7)
  - sub-S0001114591660_ses-1.h5  (Study 97)
  - sub-S0001116587855_ses-1.h5  (Study 99)

If H5 files are not available, full-night figures are generated directly
from the CSV data (slightly simplified but visually equivalent).

Usage:
    python -m scripts.run_paper_figures
    python -m scripts.run_paper_figures --figure1-only
    python -m scripts.run_paper_figures --fullnight-only
    python -m scripts.run_paper_figures --paper-panels
"""

import argparse
import os
import time
from itertools import groupby

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from hlg.em.arousal import heaviside
from hlg.em.em_algorithm import run_em_on_segment
from hlg.em.loop_gain_calc import compute_loop_gain

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "paper_examples")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "paper")

STUDIES = [
    {"csv": "NREM_OSA_Study_5.csv", "group": "NREM OSA", "study": 5},
    {"csv": "HLG_OSA_Study_7.csv", "group": "HLG OSA", "study": 7},
    {"csv": "HLG_OSA_Study_97.csv", "group": "HLG OSA", "study": 97},
    {"csv": "high_CAI_Study_99.csv", "group": "High CAI", "study": 99},
]

BDSP_MAP = {
    5: "sub-S0001111922082_ses-1.h5",
    7: "sub-S0001111985952_ses-1.h5",
    97: "sub-S0001114591660_ses-1.h5",
    99: "sub-S0001116587855_ses-1.h5",
}

# Exact paper Figure 1 panel-to-segment mapping.
# Each entry: (study_number, 0-based NREM segment index)
PAPER_PANELS = {
    "A": (99, 0),   # High CAI seg 1  → LG ~1.87
    "B": (99, 13),  # High CAI seg 14 → LG ~3.85
    "C": (97, 7),   # HLG OSA  seg 8  → LG ~0.97
    "D": (97, 19),  # HLG OSA  seg 20 → LG ~1.10
    "E": (5, 5),    # NREM OSA seg 6  → LG ~0.11
    "F": (7, 13),   # HLG OSA  seg 14 → LG ~0.39
}


def _find_events(arr):
    """Find contiguous regions of True values → list of (start, end) tuples."""
    arr = np.asarray(arr, dtype=bool)
    if len(arr) == 0:
        return []
    diff = np.diff(arr.astype(int))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)
    if arr[0]:
        starts.insert(0, 0)
    if arr[-1]:
        ends.append(len(arr))
    return list(zip(starts, ends))


def _load_demographics():
    """Load patient demographics from figure1_studies.csv."""
    csv_path = os.path.join(DATA_DIR, "figure1_studies.csv")
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    demo = {}
    for _, row in df.iterrows():
        demo[int(row["study"])] = {
            "Sex": row["Sex"],
            "Age": int(round(row["age"])),
            "AHI": round(row["AHI_3pct"], 1),
            "OAI": round(row["OAI"], 1),
            "CAI": round(row["CAI_3pct"], 1),
            "MAI": 0.0,
            "HI": round(row.get("SS_pct", 0), 2),
        }
    return demo


# ─────────────────────────────────────────────────────────────────────
# Figure 1: Per-segment EM panels
# ─────────────────────────────────────────────────────────────────────

def plot_segment_panel(
    seg: pd.DataFrame,
    upAlpha: np.ndarray,
    upgamma: np.ndarray,
    uptau: np.ndarray,
    V_o_est: np.ndarray,
    h: np.ndarray,
    u_min: float,
    Fs: int,
    stage: str,
    out_path: str,
    demographics: dict | None = None,
    ss_score: float | None = None,
) -> None:
    """Generate Figure 1 style multi-panel segment plot.

    Matches the original MATLAB publication figure including:
    - Manual + estimated arousal panels
    - Disturbance 0/1 scale labels
    - SpO2 min/max percentage labels
    - Grey arousal shading on the ventilation panel
    - Two CO2 traces (non-delayed + delayed) with calibration period
    - Double-headed tau arrow annotation
    - Patient demographics header
    - Full 7-parameter box (LG, gamma, tau, V_max, L, alpha, RMSE)
    """
    K = len(seg)
    fs = Fs
    fz = 14

    gamma = float(upgamma[-1])
    tau_sec = float(uptau[-1]) / Fs
    tau_samples = int(uptau[-1])
    alpha = float(upAlpha[-1])
    L = 0.05
    LG = compute_loop_gain(L, gamma, u_min)

    # Reconstruct arousal signal
    dit = seg["arousal_locs"].values.astype(np.float64)
    t_ar = np.where(dit != 0)[0]
    t = np.arange(1, K + 1, dtype=np.float64)
    w = 5 * Fs
    Arousal = np.zeros(K, dtype=np.float64)
    for idx in range(len(t_ar)):
        centre = t_ar[idx] + 1
        square = heaviside(t, centre - w / 2) - heaviside(t - w / 2, centre)
        Arousal += h[idx] * square

    # Scale ventilation estimate
    V_o = seg["Ventilation_ABD"].values.astype(np.float64)
    non_arousal = Arousal == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = V_o[non_arousal] / V_o_est[non_arousal]
    valid = np.isfinite(ratio) & (ratio <= 5.0)
    Scale = float(np.mean(ratio[valid])) if np.sum(valid) > 0 else 1.0

    Vd_est = V_o_est - Arousal
    Vo_est_scaled = Vd_est * Scale + Arousal
    Vd_est_scaled = Vd_est * Scale

    rmse = float(np.sqrt(np.mean((V_o - Vo_est_scaled) ** 2)))
    Vmax = round(float(np.nanmax(Vo_est_scaled)), 2)

    ABD = seg["ABD"].values.astype(np.float64)
    di_abd = seg["d_i_ABD"].values.astype(np.float64)
    Disturbance = di_abd + alpha * (1.0 - di_abd)
    spo2 = seg["SpO2"].values.astype(np.float64)
    y_tech = seg["Apnea"].values.astype(np.float64)

    # Arousal masks
    a_locs = (Arousal > 0).astype(float)
    a_locs_bool = Arousal > 0
    a_locs[a_locs != 1] = np.nan

    # CO2 via Euler integration
    dt = 1.0 / fs
    delay_steps = tau_samples
    CO2 = np.zeros(K)
    CO2[0] = 1.0
    for i in range(1, K):
        v_delayed = Vo_est_scaled[i - delay_steps] if i >= delay_steps else Vo_est_scaled[0]
        dCO2 = L - v_delayed * CO2[i - 1]
        CO2[i] = CO2[i - 1] + dCO2 * dt

    time_min = np.arange(K) / fs / 60
    t0, t1 = time_min[0], time_min[-1]
    len_x = time_min[-1]

    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(111)
    label_txt_dic = {"fontsize": fz, "ha": "right", "va": "center"}

    # ── Panel 1: Abdominal effort ─────────────────────────────────────
    factor = 4
    maxi = np.nanmax(ABD) - np.nanmin(ABD)
    ABD_n = ABD / maxi * factor * 2 if maxi > 0 else ABD
    peaks_up = find_peaks(ABD_n, distance=fs)[0]
    peaks_dn = find_peaks(-ABD_n, distance=fs)[0]
    max_y = np.median(ABD_n[peaks_up]) if len(peaks_up) > 0 else 2
    min_y = -np.median(ABD_n[peaks_dn]) if len(peaks_dn) > 0 else -2

    ax.plot(time_min, ABD_n, c="k", lw=0.5, alpha=0.75)
    ax.text(time_min[0] - 0.3, 0, "Abdominal effort", **label_txt_dic)

    # ── Panel 2: Arousals (manual + estimated) ────────────────────────
    offset_ar = max_y + 8.5
    if "Arousals" in seg.columns:
        manual_ar = seg["Arousals"].values.astype(float).copy()
        manual_ar[manual_ar != 1] = np.nan
        ax.plot([t0, t1], [offset_ar, offset_ar], c="k", lw=0.5, linestyle="dashed")
        ax.plot(time_min, manual_ar * offset_ar, c="k", lw=4)
        ax.text(time_min[0] - 0.3, offset_ar, "Manual arousals", **label_txt_dic)
        offset_est = offset_ar - 1
        ax.plot([t0, t1], [offset_est, offset_est], c="k", lw=0.5, linestyle="dashed")
        ax.plot(time_min, a_locs * offset_est, c="k", lw=4)
        ax.text(time_min[0] - 0.3, offset_est, "Estimated arousals", **label_txt_dic)
    else:
        ax.plot([t0, t1], [offset_ar, offset_ar], c="k", lw=0.5, linestyle="dashed")
        ax.plot(time_min, a_locs * offset_ar, c="k", lw=4)
        ax.text(time_min[0] - 0.3, offset_ar, "Estimated arousals", **label_txt_dic)

    # ── Panel 3: Manual respiratory events ────────────────────────────
    label_color = [None, "b", "g", "c", "m", "r", None, "b"]
    offset_events = max_y + 6.5
    ax.plot([t0, t1], [offset_events, offset_events], c="k", lw=0.5, linestyle="dashed")
    ax.text(time_min[0] - 0.3, offset_events, "Manual resp. events", **label_txt_dic)
    loc = 0
    for val, grp in groupby(y_tech):
        len_j = len(list(grp))
        if np.isfinite(val) and int(val) < len(label_color) and label_color[int(val)] is not None:
            t_start = loc / fs / 60
            t_end = (loc + len_j) / fs / 60
            ax.plot([t_start, t_end], [offset_events] * 2, c=label_color[int(val)], lw=3)
        loc += len_j

    # ── Panel 4: Disturbance U(t) with 0/1 labels ────────────────────
    offset_u = max_y + 3.75
    factor_u = 2
    ax.text(time_min[0] - 0.05, offset_u + factor_u, "1", fontsize=fz - 5, ha="right", va="center")
    ax.text(time_min[0] - 0.05, offset_u, "0", fontsize=fz - 5, ha="right", va="center")
    ax.plot(time_min, di_abd * factor_u + offset_u, c="k", lw=1, alpha=0.25)
    ax.plot(time_min, Disturbance * factor_u + offset_u, c="k", lw=2, alpha=0.5)
    ax.text(time_min[0] - 0.3, offset_u + 1, "Disturbance ($U$)", **label_txt_dic)
    ax.fill_between(time_min, offset_u, offset_u + factor_u, fc="k", alpha=0.1)

    # ── Panel 5: SpO2 with min/max % labels ───────────────────────────
    offset_spo2 = min_y - 8.5
    factor_spo2 = 5
    spo2_clean = spo2.copy()
    spo2_clean[spo2_clean < 80] = np.nan
    has_spo2 = np.any(np.isfinite(spo2_clean))
    spo2_max = int(np.nanmax(spo2_clean)) if has_spo2 else "NaN"
    spo2_min = int(np.nanmin(spo2_clean)) if has_spo2 else "NaN"
    if has_spo2:
        spo2_range = np.nanmax(spo2_clean) - np.nanmin(spo2_clean)
        spo2_n = (spo2_clean - np.nanmin(spo2_clean)) / spo2_range * factor_spo2 if spo2_range > 0 else np.full_like(spo2_clean, factor_spo2 / 2)
        ax.plot(time_min, spo2_n + offset_spo2, c="y", lw=1)
    ax.text(time_min[0] - 0.3, offset_spo2 + factor_spo2 / 2, "SpO$_{2}$", **label_txt_dic)
    ax.text(time_min[0] - 0.05, offset_spo2 + factor_spo2, f"{spo2_max}%", fontsize=fz - 5, ha="right", va="top")
    ax.text(time_min[0] - 0.05, offset_spo2, f"{spo2_min}%", fontsize=fz - 5, ha="right", va="bottom")

    # ── Panel 6: Ventilation + grey arousal shading ───────────────────
    offset_V = min_y - 17
    factor_V = 5
    maxi_v = max(np.nanmax(V_o), np.nanmax(Vo_est_scaled))
    Vo_n = V_o / maxi_v * factor_V if maxi_v > 0 else V_o
    Vo_est_n = Vo_est_scaled / maxi_v * factor_V if maxi_v > 0 else Vo_est_scaled

    ax.plot(time_min, Vo_n + offset_V, c="k", lw=2, alpha=1)
    ax.plot(time_min, Vo_est_n + offset_V, c="b", lw=2, alpha=0.7)
    ax.text(time_min[0] - 0.3, offset_V + factor_V / 2, "Ventilation", **label_txt_dic)
    ax.plot([t0, t1], [offset_V, offset_V], c="k", lw=0.5, linestyle="dashed")
    ax.plot([t0, t1], [offset_V + factor_V, offset_V + factor_V], c="k", lw=0.5, linestyle="dashed")

    # Grey arousal shading bars
    yu_bar = offset_V + max(float(np.nanmax(Vo_n)), float(np.nanmax(Vo_est_n)))
    for st, end in _find_events(a_locs_bool):
        ax.fill_between(
            [st / fs / 60, end / fs / 60], offset_V, yu_bar,
            color="k", alpha=0.1, ec="none",
        )

    # ── Panel 7: CO2 (two traces + calibration + tau arrow) ──────────
    offset_CO2 = min_y - 12
    factor_CO2 = 3
    mask_duration = tau_sec * 2.5
    mask_steps = int(mask_duration * fs)

    if mask_steps < K:
        co2_min = np.nanmin(CO2[mask_steps:])
        co2_range = np.nanmax(CO2[mask_steps:]) - co2_min
        CO2_n = (CO2 - co2_min) / co2_range * factor_CO2 if co2_range > 0 else CO2
    else:
        CO2_n = CO2

    # Non-delayed (instantaneous) CO2 trace
    CO2_non_delayed = np.copy(CO2_n)
    CO2_non_delayed[:mask_steps] = np.nan
    ax.plot(time_min, CO2_non_delayed + offset_CO2, c="k", lw=1, linestyle="dashed", alpha=0.1)

    # Delayed CO2 trace (shifted by tau)
    CO2_delayed = np.roll(CO2_n, delay_steps)
    CO2_delayed[: delay_steps + mask_steps] = np.nan
    ax.plot(time_min, CO2_delayed + offset_CO2, c="b", lw=1.5, linestyle="dashed", alpha=0.75)

    ax.text(time_min[0] - 0.3, offset_CO2 + factor_CO2 / 2, r"Modeled CO$_2$ (x)", **label_txt_dic)

    # Tau arrow annotation
    finite_idx = np.where(np.isfinite(CO2_delayed))[0]
    if len(finite_idx) > 0:
        first_idx = finite_idx[0]
        arrow_right = first_idx / fs / 60
        arrow_left = (first_idx - delay_steps) / fs / 60
        arrow_y = CO2_delayed[first_idx] + 0.25 + offset_CO2
        ax.annotate(
            "", xy=(arrow_right, arrow_y), xytext=(arrow_left, arrow_y),
            arrowprops=dict(arrowstyle="<->", color="k", lw=1),
        )
        mid_x = (arrow_right + arrow_left) / 2
        mid_y = arrow_y + 0.25
        va_arr = "bottom"
        if mid_y > 4:
            mid_y = arrow_y - 0.25
            va_arr = "top"
        ax.text(mid_x, mid_y, r"$\tau$", fontsize=fz - 2, ha="center", va=va_arr)

        # Calibration shading
        ax.fill_between(
            [t0, arrow_left], offset_CO2, offset_CO2 + factor_CO2,
            color="k", alpha=0.1, ec="none",
        )
        ax.text(time_min[0] + 0.05, offset_CO2 + 0.25, "Calibrating..",
                fontsize=fz - 4, fontstyle="italic", ha="left", va="bottom")

    # ── Annotations ───────────────────────────────────────────────────

    # Demographics header
    if demographics:
        metrics = ["Sex", "Age", "AHI", "OAI", "CAI", "MAI", "HI", "SS"]
        dx_demo = len_x / 30
        y_demo = max_y + 10
        demo_with_ss = dict(demographics)
        if ss_score is not None:
            demo_with_ss["SS"] = ss_score
        for i, metric in enumerate(metrics):
            if metric in demo_with_ss:
                x = 0.1 + dx_demo * i * 2
                ax.text(x + dx_demo / 2, y_demo, metric, fontsize=fz, ha="center", va="bottom")
                val = demo_with_ss[metric]
                ax.text(x + dx_demo / 2, y_demo - 0.25, str(val), fontsize=fz - 2, ha="center", va="top")

    # Event legend (top right)
    event_types = ["RERA", "Hypopnea", "Mixed\napnea", "Central\napnea", "Obstructive\napnea"]
    event_colors = ["r", "m", "c", "g", "b"]
    dx = len_x / 25
    for i, (color, e_type) in enumerate(zip(event_colors, event_types)):
        x = len_x - 0.05 - dx * i * 2
        ax.plot([x, x - dx], [max_y + 10 - 0.5] * 2, c=color, lw=3)
        ax.text(x - dx / 2, max_y + 10, e_type, fontsize=fz - 2, ha="center", va="bottom")

    # Ventilation legend (bottom left)
    dx_leg = len_x / 22.5
    offset_leg = offset_V - 1
    for i, (line_label, c) in enumerate([("Observed", "k"), ("Modeled", "b")]):
        x = dx_leg * i * 1.5
        ax.plot([x, x + dx_leg], [offset_leg + 0.25] * 2, c=c, lw=2)
        ax.text(x + dx_leg / 2, offset_leg, line_label, fontsize=fz - 2, ha="center", va="top")

    # Full 7-parameter box: LG, gamma, tau, V_max, L, alpha, RMSE
    scores = [r"$\bf{LG}$", "$\\gamma$", "$\\tau$", "$v_{max}$", "$L$", "$\\alpha$", "RMSE"]
    values = [
        r"$\bf{" + f"{LG:.2f}" + "}$",
        f"{gamma:.1f}", f"{tau_sec:.1f} sec", f"{Vmax}", f"{L}", f"{alpha:.1f}", f"{rmse:.2f}",
    ]
    for i, (tag_label, val) in enumerate(zip(scores, values)):
        minus = len(scores) // 2
        x = len_x / 2 + ((i - minus) * 2) * dx_leg
        ax.text(x, offset_leg - 0.5, tag_label, fontsize=fz, ha="center", va="bottom")
        ax.text(x, offset_leg - 0.75, val, fontsize=fz - 2, ha="center", va="top")

    # Duration scale bar (bottom right)
    sec = 30
    ax.plot([len_x - sec / 60, len_x], [offset_leg + 0.25] * 2, color="k", lw=1.5)
    ax.plot([len_x - sec / 60] * 2, [offset_leg, offset_leg + 0.5], color="k", lw=1.5)
    ax.plot([len_x] * 2, [offset_leg, offset_leg + 0.5], color="k", lw=1.5)
    ax.text(len_x - sec / 120, offset_leg, f"{sec} sec\n({stage})", color="k", fontsize=fz - 2, ha="center", va="top")

    ax.set_xlim([-1, len_x + 0.5])
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _run_em_on_seg(T, Fs, seg_idx, nrem_starts, nrem_ends):
    """Extract a single NREM segment and run EM on it."""
    start = int(nrem_starts[seg_idx])
    end = int(nrem_ends[seg_idx]) - 1
    if start == 0:
        start = 1
        end += 1
    end = min(end, len(T) - 1)
    seg = T.iloc[start : end + 1].copy().reset_index(drop=True)

    t0 = time.time()
    upAlpha, upgamma, uptau, V_o_est, h, u_min = run_em_on_segment(
        seg, w=5 * Fs, L=0.05, gamma_init=0.5, tau_init=15 * Fs, version="non-smooth"
    )
    elapsed = time.time() - t0
    LG = compute_loop_gain(0.05, float(upgamma[-1]), u_min)
    return seg, upAlpha, upgamma, uptau, V_o_est, h, u_min, LG, elapsed


def generate_figure1(study_info: dict, n_segments: int = 3) -> None:
    """Run EM live on the first N NREM segments and generate Figure 1 panels."""
    csv_path = os.path.join(DATA_DIR, study_info["csv"])
    T = pd.read_csv(csv_path)
    Fs = int(T["Fs"].iloc[0])
    patient_id = str(T["patient_tag"].iloc[0])[:8]
    group = study_info["group"]
    study_num = study_info["study"]

    nrem_starts = T["nrem_starts"].dropna().values.astype(int)
    nrem_ends = T["nrem_ends"].dropna().values.astype(int)
    ss_scores = T["nrem_SS_score"].dropna().values
    n = min(n_segments, len(nrem_starts))

    demographics = _load_demographics().get(study_num)

    print(f"\n  [{group}] Patient {patient_id} — {len(nrem_starts)} NREM segments, processing {n}")

    for seg_idx in range(n):
        seg, upAlpha, upgamma, uptau, V_o_est, h, u_min, LG, elapsed = (
            _run_em_on_seg(T, Fs, seg_idx, nrem_starts, nrem_ends)
        )
        ss = round(float(ss_scores[seg_idx]), 2) if seg_idx < len(ss_scores) else None
        print(f"    Seg {seg_idx + 1}: LG={LG:.2f}, γ={upgamma[-1]:.2f}, τ={uptau[-1] / Fs:.0f}s ({elapsed:.1f}s)")

        out_path = os.path.join(FIG_DIR, "figure1", f"fig1_{group.replace(' ', '_')}_{patient_id}_seg{seg_idx + 1}.png")
        plot_segment_panel(
            seg, upAlpha, upgamma, uptau, V_o_est, h, u_min, Fs, "NREM", out_path,
            demographics=demographics, ss_score=ss,
        )


def generate_paper_panels() -> None:
    """Generate exactly the 6 panels (A-F) shown in the paper's Figure 1.

    Uses the PAPER_PANELS mapping to select specific NREM segments
    that match the published figure.
    """
    demographics = _load_demographics()
    study_to_csv = {s["study"]: s for s in STUDIES}

    for panel, (study_num, seg_idx) in sorted(PAPER_PANELS.items()):
        info = study_to_csv[study_num]
        csv_path = os.path.join(DATA_DIR, info["csv"])
        T = pd.read_csv(csv_path)
        Fs = int(T["Fs"].iloc[0])
        patient_id = str(T["patient_tag"].iloc[0])[:8]
        group = info["group"]

        nrem_starts = T["nrem_starts"].dropna().values.astype(int)
        nrem_ends = T["nrem_ends"].dropna().values.astype(int)
        ss_scores = T["nrem_SS_score"].dropna().values

        seg, upAlpha, upgamma, uptau, V_o_est, h, u_min, LG, elapsed = (
            _run_em_on_seg(T, Fs, seg_idx, nrem_starts, nrem_ends)
        )
        ss = round(float(ss_scores[seg_idx]), 2) if seg_idx < len(ss_scores) else None
        print(f"    Panel {panel}: Study {study_num} ({group}) seg {seg_idx + 1} → "
              f"LG={LG:.2f}, γ={upgamma[-1]:.2f}, τ={uptau[-1] / Fs:.0f}s ({elapsed:.1f}s)")

        out_path = os.path.join(FIG_DIR, "figure1", f"fig1_panel_{panel}.png")
        plot_segment_panel(
            seg, upAlpha, upgamma, uptau, V_o_est, h, u_min, Fs, "NREM", out_path,
            demographics=demographics.get(study_num), ss_score=ss,
        )


# ─────────────────────────────────────────────────────────────────────
# Figure 2: Full-night overview with LG hooks
# ─────────────────────────────────────────────────────────────────────

def generate_full_night_from_csv(study_info: dict) -> None:
    """Generate full-night overview with LG hooks directly from CSV data."""
    csv_path = os.path.join(DATA_DIR, study_info["csv"])
    T = pd.read_csv(csv_path)
    Fs = int(T["Fs"].iloc[0])
    patient_id = str(T["patient_tag"].iloc[0])[:8]
    group = study_info["group"]

    print(f"\n  [{group}] Patient {patient_id} — full-night with LG hooks")

    signal = T["ABD"].values.astype(float)
    stages = T["Stage"].values.astype(float)
    apnea = T["Apnea"].values.astype(float)
    spo2 = T["SpO2"].values.astype(float)

    # LG per-segment data
    nrem_starts = T["nrem_starts"].dropna().values.astype(int)
    nrem_ends = T["nrem_ends"].dropna().values.astype(int)
    lg_nrem = T["LG_nrem"].dropna().values
    g_nrem = T["G_nrem"].dropna().values
    d_nrem = T["D_nrem"].dropna().values

    # Row layout: each row = 1 hour
    block = 60 * 60 * Fs
    n_samples = len(signal)
    n_rows = (n_samples + block - 1) // block
    row_ids = [np.arange(i * block, min((i + 1) * block, n_samples)) for i in range(n_rows)]
    row_ids.reverse()
    nrow = len(row_ids)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111)
    row_height = 16

    # Stage-separated signal
    sleep = signal.copy()
    sleep[np.isnan(stages)] = np.nan
    sleep[stages == 5] = np.nan

    wake = np.full_like(signal, np.nan)
    wake_mask = np.isnan(stages) | (stages == 5)
    wake[wake_mask] = signal[wake_mask]

    rem = np.full_like(signal, np.nan)
    rem[stages == 4] = signal[stages == 4]

    for ri in range(nrow):
        ax.plot(sleep[row_ids[ri]] + ri * row_height, c="k", lw=0.3, alpha=0.75)
        ax.plot(wake[row_ids[ri]] + ri * row_height, c="r", lw=0.3, alpha=0.5)
        ax.plot(rem[row_ids[ri]] + ri * row_height, c="b", lw=0.3, alpha=0.5)

    # Respiratory event bars
    label_color = [None, "b", "b", "b", "m"]
    for ri in range(nrow):
        loc = 0
        for val, grp in groupby(apnea[row_ids[ri]]):
            len_j = len(list(grp))
            if np.isfinite(val) and int(val) < len(label_color) and label_color[int(val)] is not None:
                shift = 3.5 if val == 1 else 4
                ax.plot([loc, loc + len_j], [ri * row_height - shift] * 2, c=label_color[int(val)], lw=1.5)
            loc += len_j

    # LG hooks
    for seg_idx in range(min(len(nrem_starts), len(lg_nrem))):
        s = nrem_starts[seg_idx]
        e = nrem_ends[seg_idx]
        lg_val = lg_nrem[seg_idx]

        if not np.isfinite(lg_val) or lg_val <= 0:
            continue

        mid = (s + e) // 2
        row_idx = nrow - 1 - mid // block
        if row_idx < 0 or row_idx >= nrow:
            continue

        x_pos = mid % block
        hook_height = min(lg_val * 3, 12)
        hook_y = row_idx * row_height + 5

        ax.plot([x_pos, x_pos], [hook_y, hook_y + hook_height], c="k", lw=0.8)
        ax.plot([x_pos - 50, x_pos + 50], [hook_y] * 2, c="k", lw=0.8)

    # Layout
    max_row_len = max(len(x) for x in row_ids)
    ax.set_xlim([0, max_row_len])
    ax.axis("off")

    # Hour labels
    for ri in range(nrow):
        hour = nrow - 1 - ri
        ax.text(-200, ri * row_height, f"{hour}h", fontsize=9, ha="right", va="center")

    # Title
    ax.text(
        max_row_len / 2, (nrow - 1) * row_height + 14,
        f"{group} — Patient {patient_id}  ({len(nrem_starts)} NREM segments)",
        fontsize=12, ha="center", va="bottom", fontweight="bold",
    )

    # Legend
    y_legend = -10
    fz = 10
    line_types = ["NREM", "REM", "Wake"]
    line_colors = ["k", "b", "r"]
    for i, (color, e_type) in enumerate(zip(line_colors, line_types)):
        x = 60 * Fs + 200 * Fs * i
        ax.plot([x, x + 50 * Fs], [y_legend] * 2, c=color, lw=0.8)
        ax.text(x + 25 * Fs, y_legend - 3, e_type, fontsize=fz, c=color, ha="center", va="top")

    event_types = ["Apnea", "Hypopnea"]
    ev_colors = ["b", "m"]
    for i, (color, e_type) in enumerate(zip(ev_colors, event_types)):
        x = 200 * Fs * (len(line_types) + 0.5) + 300 * Fs * (i + 1)
        ax.plot([x, x + 100 * Fs], [y_legend] * 2, c=color, lw=2)
        ax.text(x + 50 * Fs, y_legend - 3, e_type, fontsize=fz, ha="center", va="top")

    # LG hook legend
    lg_x = max_row_len - 4 * 60 * Fs
    ax.plot([lg_x, lg_x], [y_legend, y_legend + 4], c="k", lw=0.8)
    ax.plot([lg_x - 50, lg_x + 50], [y_legend] * 2, c="k", lw=0.8)
    ax.text(lg_x + 100, y_legend + 2, "Estimated LG", fontsize=fz - 1, ha="left", va="center")

    # Duration scale bar
    dur_min = 5
    dur_samples = dur_min * 60 * Fs
    ax.plot([max_row_len - dur_samples, max_row_len], [y_legend] * 2, color="k", lw=1)
    ax.plot([max_row_len - dur_samples] * 2, [y_legend - 0.5, y_legend + 0.5], color="k", lw=1)
    ax.plot([max_row_len] * 2, [y_legend - 0.5, y_legend + 0.5], color="k", lw=1)
    ax.text(max_row_len - dur_samples / 2, y_legend + 1, f"{dur_min} min", fontsize=fz, ha="center", va="bottom")

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "full_night", f"full_night_{patient_id}_{group.replace(' ', '_')}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"    Saved: {out_path}")
    plt.close()


def generate_full_night(study_info: dict) -> None:
    """Generate full-night overview with LG hooks from CSV data."""
    generate_full_night_from_csv(study_info)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures for all 4 example patients")
    parser.add_argument("--figure1-only", action="store_true", help="Only generate Figure 1 panels")
    parser.add_argument("--fullnight-only", action="store_true", help="Only generate full-night figures")
    parser.add_argument("--paper-panels", action="store_true", help="Generate exactly the 6 panels (A-F) from the paper")
    parser.add_argument("--segments", type=int, default=2, help="Number of NREM segments per patient for Figure 1 (default: 2)")
    args = parser.parse_args()

    do_fig1 = not args.fullnight_only and not args.paper_panels
    do_fullnight = not args.figure1_only and not args.paper_panels
    do_paper = args.paper_panels

    os.makedirs(os.path.join(FIG_DIR, "figure1"), exist_ok=True)
    os.makedirs(os.path.join(FIG_DIR, "full_night"), exist_ok=True)

    metadata = pd.read_csv(os.path.join(DATA_DIR, "figure1_studies.csv"))
    print("=" * 70)
    print("  Paper Figure Generation — 4 Example Patients")
    print("=" * 70)
    print(f"\n  Patients:")
    for _, row in metadata.iterrows():
        print(f"    Study {row['study']:>2d}  {row['group']:<12s}  {row['Sex']}, age {row['age']}, AHI={row['AHI_3pct']:.1f}")

    t_total = time.time()

    if do_paper:
        print(f"\n{'─' * 70}")
        print("  FIGURE 1: Exact paper panels A–F")
        print(f"{'─' * 70}")
        generate_paper_panels()
    elif do_fig1:
        print(f"\n{'─' * 70}")
        print(f"  FIGURE 1: Per-segment EM panels ({args.segments} segments each)")
        print(f"{'─' * 70}")
        for study in STUDIES:
            generate_figure1(study, n_segments=args.segments)

    if do_fullnight:
        print(f"\n{'─' * 70}")
        print("  FIGURE 2: Full-night overviews with LG hooks")
        print(f"{'─' * 70}")
        for study in STUDIES:
            generate_full_night(study)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"  Complete! Total time: {elapsed:.0f}s")
    print(f"  Output: {FIG_DIR}/")
    print("=" * 70)

    for subdir in ["figure1", "full_night"]:
        d = os.path.join(FIG_DIR, subdir)
        if os.path.isdir(d):
            files = sorted(os.listdir(d))
            if files:
                print(f"\n  {subdir}/")
                for f in files:
                    size = os.path.getsize(os.path.join(d, f))
                    print(f"    {f:60s} {size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
