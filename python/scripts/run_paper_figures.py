#!/usr/bin/env python
"""
Reproduce the paper figures from the 4 example patients.

Generates:
  - Figure 1: Per-segment EM fit panels (re-runs EM live) for all 4 patients
  - Figure 2: Full-night overview with LG hooks for all 4 patients

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
) -> None:
    """Generate Figure 1 style multi-panel segment plot."""
    K = len(seg)
    fs = Fs
    fz = 13

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

    ABD = seg["ABD"].values.astype(np.float64)
    di_abd = seg["d_i_ABD"].values.astype(np.float64)
    Disturbance = di_abd + alpha * (1.0 - di_abd)
    spo2 = seg["SpO2"].values.astype(np.float64)
    y_tech = seg["Apnea"].values.astype(np.float64)
    a_locs = (Arousal > 0).astype(float)
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

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    label_txt_dic = {"fontsize": fz, "ha": "right", "va": "center"}

    factor = 4
    maxi = np.nanmax(ABD) - np.nanmin(ABD)
    ABD_n = ABD / maxi * factor * 2 if maxi > 0 else ABD
    peaks_up = find_peaks(ABD_n, distance=fs)[0]
    peaks_dn = find_peaks(-ABD_n, distance=fs)[0]
    max_y = np.median(ABD_n[peaks_up]) if len(peaks_up) > 0 else 2
    min_y = -np.median(ABD_n[peaks_dn]) if len(peaks_dn) > 0 else -2

    ax.plot(time_min, ABD_n, c="k", lw=0.5, alpha=0.75)
    ax.text(time_min[0] - 0.3, 0, "Abdominal effort", **label_txt_dic)

    offset = max_y + 8.5
    ax.plot([t0, t1], [offset, offset], c="k", lw=0.5, linestyle="dashed")
    ax.plot(time_min, a_locs * offset, c="k", lw=4)
    ax.text(time_min[0] - 0.3, offset, "Estimated arousals", **label_txt_dic)

    label_color = [None, "b", "g", "c", "m", "r", None, "b"]
    offset_events = max_y + 6.5
    ax.plot([t0, t1], [offset_events, offset_events], c="k", lw=0.5, linestyle="dashed")
    ax.text(time_min[0] - 0.3, offset_events, "Resp. events", **label_txt_dic)
    loc = 0
    for val, group in groupby(y_tech):
        len_j = len(list(group))
        if np.isfinite(val) and int(val) < len(label_color) and label_color[int(val)] is not None:
            t_start = loc / fs / 60
            t_end = (loc + len_j) / fs / 60
            ax.plot([t_start, t_end], [offset_events] * 2, c=label_color[int(val)], lw=3)
        loc += len_j

    offset_u = max_y + 3.75
    factor_u = 2
    ax.plot(time_min, di_abd * factor_u + offset_u, c="k", lw=1, alpha=0.25)
    ax.plot(time_min, Disturbance * factor_u + offset_u, c="k", lw=2, alpha=0.5)
    ax.text(time_min[0] - 0.3, offset_u + 1, "Disturbance ($U$)", **label_txt_dic)
    ax.fill_between(time_min, offset_u, offset_u + factor_u, fc="k", alpha=0.1)

    offset_spo2 = min_y - 8.5
    factor_spo2 = 5
    spo2_clean = spo2.copy()
    spo2_clean[spo2_clean < 80] = np.nan
    if np.any(np.isfinite(spo2_clean)):
        spo2_n = (spo2_clean - np.nanmin(spo2_clean)) / (np.nanmax(spo2_clean) - np.nanmin(spo2_clean)) * factor_spo2
        ax.plot(time_min, spo2_n + offset_spo2, c="y", lw=1)
    ax.text(time_min[0] - 0.3, offset_spo2 + factor_spo2 / 2, "SpO$_{2}$", **label_txt_dic)

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

    offset_CO2 = min_y - 12
    factor_CO2 = 3
    mask_steps = int(tau_sec * 2.5 * fs)
    CO2_n = (
        (CO2 - np.nanmin(CO2[mask_steps:])) / (np.nanmax(CO2[mask_steps:]) - np.nanmin(CO2[mask_steps:])) * factor_CO2
        if mask_steps < K
        else CO2
    )
    CO2_delayed = np.roll(CO2_n, delay_steps)
    CO2_delayed[: delay_steps + mask_steps] = np.nan
    ax.plot(time_min, CO2_delayed + offset_CO2, c="b", lw=1.5, linestyle="dashed", alpha=0.75)
    ax.text(time_min[0] - 0.3, offset_CO2 + factor_CO2 / 2, "CO$_{2}$ (model)", **label_txt_dic)

    # Annotations
    len_x = time_min[-1]
    scores = [r"$\bf{LG}$", "$\\gamma$", "$\\tau$", "RMSE"]
    values = [f"{LG:.2f}", f"{gamma:.2f}", f"{tau_sec:.0f}s", f"{rmse:.2f}"]
    for i, (tag_label, val) in enumerate(zip(scores, values)):
        x = len_x / 2 + (i - 1.5) * len_x / 8
        ax.text(x, offset_V - 1, tag_label, fontsize=fz, ha="center", va="bottom")
        ax.text(x, offset_V - 1.25, val, fontsize=fz - 2, ha="center", va="top")

    line_w = len_x / 15
    y_label = offset_V - 1
    y_value = offset_V - 1.25
    obs_x0 = 0
    ax.plot([obs_x0, obs_x0 + line_w], [y_label] * 2, c="k", lw=2)
    ax.text(obs_x0 + line_w / 2, y_value, "Observed", fontsize=fz - 2, ha="center", va="top")
    mod_x0 = len_x / 6
    ax.plot([mod_x0, mod_x0 + line_w], [y_label] * 2, c="b", lw=2)
    ax.text(mod_x0 + line_w / 2, y_value, "Modeled", fontsize=fz - 2, ha="center", va="top")

    event_types = ["RERA", "Hypopnea", "Mixed", "Central", "Obstructive"]
    event_colors = ["r", "m", "c", "g", "b"]
    dx = len_x / 25
    for i, (color, e_type) in enumerate(zip(event_colors, event_types)):
        x = len_x - 0.5 - dx * i * 2
        ax.plot([x, x - dx], [max_y + 10 - 0.5] * 2, c=color, lw=3)
        ax.text(x - dx / 2, max_y + 10, e_type, fontsize=fz - 3, ha="center", va="bottom")

    sec = 30
    ax.plot([len_x - sec / 60, len_x], [offset_V - 0.5] * 2, color="k", lw=1.5)
    ax.plot([len_x - sec / 60] * 2, [offset_V - 0.75, offset_V - 0.25], color="k", lw=1.5)
    ax.plot([len_x] * 2, [offset_V - 0.75, offset_V - 0.25], color="k", lw=1.5)
    ax.text(len_x - sec / 120, offset_V - 1, f"{sec} sec\n({stage})", color="k", fontsize=fz - 2, ha="center", va="top")

    ax.set_xlim([-1, len_x + 0.5])
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_figure1(study_info: dict, n_segments: int = 3) -> None:
    """Run EM live on the first N NREM segments and generate Figure 1 panels."""
    csv_path = os.path.join(DATA_DIR, study_info["csv"])
    T = pd.read_csv(csv_path)
    Fs = int(T["Fs"].iloc[0])
    patient_id = str(T["patient_tag"].iloc[0])[:8]
    group = study_info["group"]

    nrem_starts = T["nrem_starts"].dropna().values.astype(int)
    nrem_ends = T["nrem_ends"].dropna().values.astype(int)
    n = min(n_segments, len(nrem_starts))

    print(f"\n  [{group}] Patient {patient_id} — {len(nrem_starts)} NREM segments, processing {n}")

    for seg_idx in range(n):
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
        print(f"    Seg {seg_idx + 1}: LG={LG:.2f}, γ={upgamma[-1]:.2f}, τ={uptau[-1] / Fs:.0f}s ({elapsed:.1f}s)")

        out_path = os.path.join(FIG_DIR, "figure1", f"fig1_{group.replace(' ', '_')}_{patient_id}_seg{seg_idx + 1}.png")
        plot_segment_panel(seg, upAlpha, upgamma, uptau, V_o_est, h, u_min, Fs, "NREM", out_path)


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
    parser.add_argument("--segments", type=int, default=2, help="Number of NREM segments per patient for Figure 1 (default: 2)")
    args = parser.parse_args()

    do_fig1 = not args.fullnight_only
    do_fullnight = not args.figure1_only

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

    if do_fig1:
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
