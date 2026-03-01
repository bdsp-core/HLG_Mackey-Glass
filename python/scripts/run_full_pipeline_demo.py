#!/usr/bin/env python
"""
End-to-end pipeline demo: from raw HF5 recording to EM estimation and figures.

This script demonstrates the complete HLG analysis pipeline on a single
example patient, producing all intermediate outputs and figures along the
way. It serves as both a tutorial and a smoke test for the entire codebase.

Pipeline stages:
    1. Load raw HF5 recording (SS pipeline output)
    2. Create ventilation trace + segment into 8-min NREM/REM blocks
    3. Run the Python EM algorithm (Mackey-Glass parameter estimation)
    4. Compute loop gain per segment
    5. Generate full-night overview figure (with LG hooks if EM data available)

All outputs are saved under figures/demo/ with clear filenames.

Usage:
    python -m scripts.run_full_pipeline_demo
"""

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hlg.config import config
from hlg.em.em_algorithm import run_em_on_segment
from hlg.em.loop_gain_calc import compute_loop_gain
from hlg.io.readers import load_sim_output
from hlg.ss.pipeline import segment_and_export_recording

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "demo")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Pick the example patient (7b1e2d31 — matches Study_example.csv)
    hf5_file = "7b1e2d318c35e3c7a0e9fd2c134abd10ee1c491aa236519c57cbe5a290cb88a7_20140804_230625000.hf5"
    hf5_path = os.path.join(config.hf5_dir, hf5_file)
    patient_id = hf5_file.split("_")[0][:8]

    print("=" * 70)
    print(f"  HLG Full Pipeline Demo — Patient {patient_id}")
    print("=" * 70)

    # ── Stage 1: Load raw recording ──────────────────────────────────
    print("\n[Stage 1] Loading HF5 recording...")
    t0 = time.time()
    data, hdr = load_sim_output(hf5_path)
    print(f"  Loaded: {len(data)} samples ({len(data) / hdr['newFs'] / 3600:.1f} hours)")
    print(f"  AHI={hdr.get('AHI', '?')}, CAI={hdr.get('CAI', '?')}, RDI={hdr.get('RDI', '?')}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── Stage 2: Segment into 8-min blocks ───────────────────────────
    print("\n[Stage 2] Creating ventilation trace and segmenting...")
    t0 = time.time()
    em_input_csv = os.path.join(DATA_DIR, "em_example", f"Study_{patient_id}.csv")
    if not os.path.exists(em_input_csv):
        # Use existing Study_example.csv if it matches
        existing = os.path.join(DATA_DIR, "em_example", "Study_example.csv")
        if os.path.exists(existing):
            tag = pd.read_csv(existing, nrows=1)["patient_tag"].iloc[0]
            if patient_id in tag:
                em_input_csv = existing
                print(f"  Using existing: {os.path.basename(em_input_csv)}")
            else:
                print("  Generating from HF5...")
                segment_and_export_recording(hf5_path, em_input_csv)
        else:
            print("  Generating from HF5...")
            segment_and_export_recording(hf5_path, em_input_csv)
    else:
        print(f"  Already exists: {os.path.basename(em_input_csv)}")

    T = pd.read_csv(em_input_csv)
    Fs = int(T["Fs"].iloc[0])
    nrem_starts = T["nrem_starts"].dropna().values.astype(int)
    rem_starts = T["rem_starts"].dropna().values.astype(int)
    print(f"  Segments: {len(nrem_starts)} NREM + {len(rem_starts)} REM = {len(nrem_starts) + len(rem_starts)} total")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── Stage 3: Run EM on first 3 NREM segments ────────────────────
    print("\n[Stage 3] Running Python EM algorithm (first 3 NREM segments)...")
    nrem_ends = T["nrem_ends"].dropna().values.astype(int)
    n_segments = min(3, len(nrem_starts))
    results = []

    for i in range(n_segments):
        start = int(nrem_starts[i])
        end = int(nrem_ends[i]) - 1
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

        # Compute loop gain
        LG = compute_loop_gain(0.05, float(upgamma[-1]), u_min)

        results.append(
            {
                "segment": i + 1,
                "start": start,
                "end": end,
                "gamma": float(upgamma[-1]),
                "tau_sec": float(uptau[-1]) / Fs,
                "alpha": float(upAlpha[-1]),
                "LG": LG,
                "n_arousals": len(h),
                "time_sec": elapsed,
            }
        )
        print(
            f"  Segment {i + 1}: gamma={upgamma[-1]:.2f}, "
            f"tau={uptau[-1] / Fs:.1f}s, alpha={upAlpha[-1]:.2f}, "
            f"LG={LG:.2f} ({elapsed:.1f}s)"
        )

    # Save results table
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(FIGURES_DIR, "em_results_summary.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\n  Results saved: {results_csv}")

    # ── Stage 4: Generate per-segment figure ─────────────────────────
    print("\n[Stage 4] Generating per-segment ventilation comparison plot...")
    # Re-run on first segment to get V_o_est for plotting
    start = int(nrem_starts[0])
    end = int(nrem_ends[0]) - 1
    if start == 0:
        start = 1
        end += 1
    end = min(end, len(T) - 1)
    seg = T.iloc[start : end + 1].copy().reset_index(drop=True)
    upAlpha, upgamma, uptau, V_o_est, h, u_min = run_em_on_segment(
        seg, w=5 * Fs, L=0.05, gamma_init=0.5, tau_init=15 * Fs, version="non-smooth"
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    time_min = np.arange(len(seg)) / Fs / 60

    # Panel 1: Observed vs Estimated Ventilation
    axes[0].plot(time_min, seg["Ventilation_ABD"].values, "k", lw=0.8, label="Observed (V_o)")
    axes[0].plot(time_min, V_o_est, "b", lw=0.8, alpha=0.7, label="Estimated (MG model)")
    axes[0].set_ylabel("Ventilation")
    axes[0].legend(fontsize=9)
    axes[0].set_title(
        f"NREM Segment 1 — gamma={upgamma[-1]:.2f}, "
        f"tau={uptau[-1] / Fs:.1f}s, LG={compute_loop_gain(0.05, float(upgamma[-1]), u_min):.2f}"
    )

    # Panel 2: Drive signal (d_i)
    axes[1].plot(time_min, seg["d_i_ABD"].values, "k", lw=0.8)
    axes[1].axhline(1.0, color="r", linestyle="--", lw=0.5, alpha=0.5)
    axes[1].set_ylabel("Drive (d_i)")
    axes[1].set_ylim(-0.1, 1.2)

    # Panel 3: Apnea events
    axes[2].plot(time_min, seg["Apnea"].values, "b", lw=0.8, label="Scored apnea")
    axes[2].set_ylabel("Apnea label")
    axes[2].set_xlabel("Time (minutes)")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    # Remove all spines from the top two panels (ventilation + drive)
    for ax in axes[:2]:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False)

    plt.tight_layout()
    seg_fig = os.path.join(FIGURES_DIR, f"segment_comparison_{patient_id}.png")
    plt.savefig(seg_fig, dpi=300, bbox_inches="tight")
    print(f"  Saved: {seg_fig}")
    plt.close()

    # ── Stage 5: Generate full-night figure ──────────────────────────
    print("\n[Stage 5] Generating full-night overview...")
    from scripts.run_full_night_example import plot_full_night_from_hf5

    fn_fig = os.path.join(FIGURES_DIR, f"full_night_{patient_id}.png")
    plot_full_night_from_hf5(hf5_path, fn_fig)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Pipeline complete! All outputs in figures/demo/")
    print("=" * 70)
    print("\n  Outputs:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        path = os.path.join(FIGURES_DIR, f)
        size = os.path.getsize(path)
        print(f"    {f:50s} {size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
