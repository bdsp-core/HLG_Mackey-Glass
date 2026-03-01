#!/usr/bin/env python
"""
Generate Figure 1 style panels: per-segment EM fit with CO2 model.

Runs the Python EM on selected segments and produces the multi-panel
"audit trail" figure from the paper, showing:

  Panel 1: Abdominal effort trace
  Panel 2: Estimated arousal locations
  Panel 3: Scored respiratory events (apneas/hypopneas)
  Panel 4: Disturbance signal U(t) = d_i + alpha*(1-d_i)
  Panel 5: SpO2
  Panel 6: Observed vs Modeled ventilation (with arousals highlighted)
  Panel 7: Modeled CO2 (Euler integration of state equation)

This reproduces the key diagnostic figure from the paper without
requiring pre-computed EM output CSVs.

Usage:
    python -m scripts.run_figure1_demo
"""

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from hlg.config import config
from hlg.em.arousal import heaviside
from hlg.em.em_algorithm import run_em_on_segment
from hlg.em.loop_gain_calc import compute_loop_gain

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "demo")


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
    seg_idx: int,
    out_path: str,
) -> None:
    """Generate the Figure 1 style multi-panel segment plot."""

    K = len(seg)
    fs = Fs
    fz = 13

    # Final parameters
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

    # Compute RMSE
    rmse = float(np.sqrt(np.mean((V_o - Vo_est_scaled) ** 2)))

    # Signals
    ABD = seg["ABD"].values.astype(np.float64)
    di_abd = seg["d_i_ABD"].values.astype(np.float64)
    Disturbance = di_abd + alpha * (1.0 - di_abd)
    spo2 = seg["SpO2"].values.astype(np.float64)
    y_tech = seg["Apnea"].values.astype(np.float64)
    a_locs = (Arousal > 0).astype(float)
    a_locs[a_locs != 1] = np.nan

    # CO2 model via Euler integration
    dt = 1.0 / fs
    delay_steps = tau_samples
    CO2 = np.zeros(K)
    CO2[0] = 1.0
    for i in range(1, K):
        v_delayed = Vo_est_scaled[i - delay_steps] if i >= delay_steps else Vo_est_scaled[0]
        dCO2 = L - v_delayed * CO2[i - 1]
        CO2[i] = CO2[i - 1] + dCO2 * dt

    time_min = np.arange(K) / fs / 60

    # ── Build figure ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    label_txt_dic = {"fontsize": fz, "ha": "right", "va": "center"}

    # Normalize ABD for display
    factor = 4
    maxi = np.nanmax(ABD) - np.nanmin(ABD)
    ABD_n = ABD / maxi * factor * 2 if maxi > 0 else ABD
    max_y = np.median(ABD_n[find_peaks(ABD_n, distance=fs)[0]]) if len(find_peaks(ABD_n, distance=fs)[0]) > 0 else 2
    min_y = -np.median(ABD_n[find_peaks(-ABD_n, distance=fs)[0]]) if len(find_peaks(-ABD_n, distance=fs)[0]) > 0 else -2

    # Panel 1: Abdominal effort
    ax.plot(time_min, ABD_n, c="k", lw=0.5, alpha=0.75)
    ax.text(time_min[0] - 0.3, 0, "Abdominal effort", **label_txt_dic)

    # Data x-range for clipped dashed lines (no overshoot beyond signal)
    t0, t1 = time_min[0], time_min[-1]

    # Panel 2: Estimated arousals
    offset = max_y + 8.5
    ax.plot([t0, t1], [offset, offset], c="k", lw=0.5, linestyle="dashed")
    ax.plot(time_min, a_locs * offset, c="k", lw=4)
    ax.text(time_min[0] - 0.3, offset, "Estimated arousals", **label_txt_dic)

    # Panel 3: Respiratory events
    label_color = [None, "b", "g", "c", "m", "r", None, "b"]
    offset_events = max_y + 6.5
    ax.plot([t0, t1], [offset_events, offset_events], c="k", lw=0.5, linestyle="dashed")
    ax.text(time_min[0] - 0.3, offset_events, "Resp. events", **label_txt_dic)
    from itertools import groupby

    loc = 0
    for val, group in groupby(y_tech):
        len_j = len(list(group))
        if np.isfinite(val) and int(val) < len(label_color) and label_color[int(val)] is not None:
            t_start = loc / fs / 60
            t_end = (loc + len_j) / fs / 60
            ax.plot([t_start, t_end], [offset_events] * 2, c=label_color[int(val)], lw=3)
        loc += len_j

    # Panel 4: Disturbance U(t)
    offset_u = max_y + 3.75
    factor_u = 2
    ax.plot(time_min, di_abd * factor_u + offset_u, c="k", lw=1, alpha=0.25)
    ax.plot(time_min, Disturbance * factor_u + offset_u, c="k", lw=2, alpha=0.5)
    ax.text(time_min[0] - 0.3, offset_u + 1, "Disturbance ($U$)", **label_txt_dic)
    ax.fill_between(time_min, offset_u, offset_u + factor_u, fc="k", alpha=0.1)

    # Panel 5: SpO2
    offset_spo2 = min_y - 8.5
    factor_spo2 = 5
    spo2_clean = spo2.copy()
    spo2_clean[spo2_clean < 80] = np.nan
    if np.any(np.isfinite(spo2_clean)):
        spo2_n = (spo2_clean - np.nanmin(spo2_clean)) / (np.nanmax(spo2_clean) - np.nanmin(spo2_clean)) * factor_spo2
        ax.plot(time_min, spo2_n + offset_spo2, c="y", lw=1)
    ax.text(time_min[0] - 0.3, offset_spo2 + factor_spo2 / 2, "SpO$_{2}$", **label_txt_dic)

    # Panel 6: Ventilation (observed vs modeled)
    offset_V = min_y - 17
    factor_V = 5
    maxi_v = max(np.nanmax(V_o), np.nanmax(Vo_est_scaled))
    Vo_n = V_o / maxi_v * factor_V if maxi_v > 0 else V_o
    Vo_est_n = Vo_est_scaled / maxi_v * factor_V if maxi_v > 0 else Vo_est_scaled
    _Vd_est_n = Vd_est_scaled / maxi_v * factor_V if maxi_v > 0 else Vd_est_scaled

    ax.plot(time_min, Vo_n + offset_V, c="k", lw=2, alpha=1)
    ax.plot(time_min, Vo_est_n + offset_V, c="b", lw=2, alpha=0.7)
    ax.text(time_min[0] - 0.3, offset_V + factor_V / 2, "Ventilation", **label_txt_dic)
    ax.plot([t0, t1], [offset_V, offset_V], c="k", lw=0.5, linestyle="dashed")
    ax.plot([t0, t1], [offset_V + factor_V, offset_V + factor_V], c="k", lw=0.5, linestyle="dashed")

    # Panel 7: CO2
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

    # ── Annotations ───────────────────────────────────────────────────
    len_x = time_min[-1]

    # Parameter box
    scores = [r"$\bf{LG}$", "$\\gamma$", "$\\tau$", "RMSE"]
    values = [f"{LG:.2f}", f"{gamma:.2f}", f"{tau_sec:.0f}s", f"{rmse:.2f}"]
    for i, (tag_label, val) in enumerate(zip(scores, values)):
        x = len_x / 2 + (i - 1.5) * len_x / 8
        ax.text(x, offset_V - 1, tag_label, fontsize=fz, ha="center", va="bottom")
        ax.text(x, offset_V - 1.25, val, fontsize=fz - 2, ha="center", va="top")

    # Ventilation legend — aligned with parameter labels/values
    # Lines at the same height as "LG", "gamma", etc. (offset_V - 1)
    # Text at the same height as the parameter values (offset_V - 1.25)
    line_w = len_x / 15
    y_label = offset_V - 1      # same as parameter label row
    y_value = offset_V - 1.25   # same as parameter value row

    obs_x0 = 0
    ax.plot([obs_x0, obs_x0 + line_w], [y_label] * 2, c="k", lw=2)
    ax.text(obs_x0 + line_w / 2, y_value, "Observed", fontsize=fz - 2, ha="center", va="top")

    mod_x0 = len_x / 6
    ax.plot([mod_x0, mod_x0 + line_w], [y_label] * 2, c="b", lw=2)
    ax.text(mod_x0 + line_w / 2, y_value, "Modeled", fontsize=fz - 2, ha="center", va="top")

    # Event legend
    event_types = ["RERA", "Hypopnea", "Mixed", "Central", "Obstructive"]
    event_colors = ["r", "m", "c", "g", "b"]
    dx = len_x / 25
    for i, (color, e_type) in enumerate(zip(event_colors, event_types)):
        x = len_x - 0.5 - dx * i * 2
        ax.plot([x, x - dx], [max_y + 10 - 0.5] * 2, c=color, lw=3)
        ax.text(x - dx / 2, max_y + 10, e_type, fontsize=fz - 3, ha="center", va="bottom")

    # Duration scale bar
    sec = 30
    ax.plot([len_x - sec / 60, len_x], [offset_V - 0.5] * 2, color="k", lw=1.5)
    ax.plot([len_x - sec / 60] * 2, [offset_V - 0.75, offset_V - 0.25], color="k", lw=1.5)
    ax.plot([len_x] * 2, [offset_V - 0.75, offset_V - 0.25], color="k", lw=1.5)
    ax.text(len_x - sec / 120, offset_V - 1, f"{sec} sec\n({stage})", color="k", fontsize=fz - 2, ha="center", va="top")

    ax.set_xlim([-1, len_x + 0.5])
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load study data
    csv_path = os.path.join(config.hf5_dir, "..", "em_example", "Study_example.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "em_example", "Study_example.csv")
    T = pd.read_csv(csv_path)
    Fs = int(T["Fs"].iloc[0])
    patient_id = str(T["patient_tag"].iloc[0])[:8]

    nrem_starts = T["nrem_starts"].dropna().values.astype(int)
    nrem_ends = T["nrem_ends"].dropna().values.astype(int)

    print("=" * 60)
    print(f"  Figure 1 Demo — Patient {patient_id}")
    print("=" * 60)

    # Process first 3 NREM segments
    n_segments = min(3, len(nrem_starts))
    for seg_idx in range(n_segments):
        start = int(nrem_starts[seg_idx])
        end = int(nrem_ends[seg_idx]) - 1
        if start == 0:
            start = 1
            end += 1
        end = min(end, len(T) - 1)
        seg = T.iloc[start : end + 1].copy().reset_index(drop=True)

        print(f"\n  Segment {seg_idx + 1}/{n_segments} [{start}:{end}] ({len(seg) / Fs / 60:.1f} min)...")
        t0 = time.time()
        upAlpha, upgamma, uptau, V_o_est, h, u_min = run_em_on_segment(
            seg, w=5 * Fs, L=0.05, gamma_init=0.5, tau_init=15 * Fs, version="non-smooth"
        )
        elapsed = time.time() - t0
        LG = compute_loop_gain(0.05, float(upgamma[-1]), u_min)
        print(f"    EM: gamma={upgamma[-1]:.2f}, tau={uptau[-1] / Fs:.1f}s, LG={LG:.2f} ({elapsed:.1f}s)")

        out_path = os.path.join(FIGURES_DIR, f"figure1_seg{seg_idx + 1}_{patient_id}.png")
        plot_segment_panel(seg, upAlpha, upgamma, uptau, V_o_est, h, u_min, Fs, "nrem", seg_idx, out_path)

    print(f"\n{'=' * 60}")
    print("  All Figure 1 panels saved to figures/demo/")
    print("=" * 60)


if __name__ == "__main__":
    main()
