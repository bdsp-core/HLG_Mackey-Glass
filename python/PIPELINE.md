# HLG Analysis Pipeline

Complete guide to running the HLG analysis from raw recordings to publication figures.

---

## Pipeline Overview

```
Raw HF5 recording
        │
        ▼
[Stage 1] Load & preprocess ─── hlg.io.readers.load_sim_output()
        │
        ▼
[Stage 2] Ventilation + Segment ─── hlg.ss.pipeline.segment_and_export_recording()
        │                           Creates 8-min NREM/REM blocks → EM input CSV
        ▼
[Stage 3] EM Algorithm ──────────── hlg.em.em_algorithm.run_em_on_segment()
        │                           Estimates gamma, tau, alpha per segment
        ▼
[Stage 4] Loop Gain ─────────────── hlg.em.loop_gain_calc.compute_loop_gain()
        │                           LG = ventilatory overshoot / deficit
        ▼
[Stage 5] Post-process + Figures ── hlg.em.postprocessing + hlg.visualization
        │                           Smooth LG, create hypnograms, generate figures
        ▼
    Publication-ready outputs
```

---

## Data Directory

```
data/
├── hf5_examples/           Raw HF5 recordings (SS pipeline output)
│   ├── 39fc3416...hf5      Patient 1 (~8h, AHI=47)
│   ├── 4e504ee4...hf5      Patient 2 (~7h, AHI=45)
│   ├── 7b1e2d31...hf5      Patient 3 (~7h, AHI=40)  ← demo patient
│   └── d1c690da...hf5      Patient 4 (~8h, AHI=52)
│
├── em_example/             EM input CSV (from Stage 2)
│   └── Study_example.csv   Segmented data for patient 7b1e2d31
│
├── csv_files/              Cohort metadata (patient selection tables)
│   ├── mgh_table1 100_*.csv
│   ├── bdsp_table1 200_*.csv
│   └── ...
│
├── bars/                   Pre-computed LG histogram bars (CPAP analysis)
│   ├── bdsp_CPAP_success/
│   └── bdsp_CPAP_failure/
│
└── interm_Results/         Aggregated per-cohort results (from batch EM runs)
    └── non-smooth/
        ├── mgh_REM_OSA/
        ├── mgh_NREM_OSA/
        ├── mgh_high_CAI/
        ├── mgh_SS_OSA/
        ├── mgh_SS_range/
        ├── bdsp_CPAP_success/
        ├── bdsp_CPAP_failure/
        ├── redeker_Heart_Failure/
        └── rt_Altitude/
```

---

## Figures Directory

```
figures/
├── demo/                        End-to-end pipeline demo outputs
│   ├── em_results_summary.csv   Per-segment EM parameter table
│   ├── segment_comparison_*.png Observed vs estimated ventilation
│   └── full_night_*.png         Full-night overview
│
├── cohort_boxplots/             LG / gamma / tau across 5 clinical cohorts
│   └── LG_gamma_tau_boxplots.png
│
├── ss_relationship/             SS-vs-LG analysis
│   ├── SS_vs_LG_scatter.png    2nd-order polynomial regression
│   ├── NREM_vs_REM_LG_boxplot.png
│   └── LG_swimmer_plot.png     Multi-cohort overnight LG profiles
│
├── cpap_prediction/             CPAP treatment outcome prediction
│   └── CPAP_ROC_PR_calibration.png
│
├── altitude/                    Altitude study
│   └── altitude_spaghetti_histograms.png
│
├── full_night/                  Full-night overviews (one per patient)
│   ├── full_night_39fc3416.pdf
│   ├── full_night_4e504ee4.pdf
│   ├── full_night_7b1e2d31.pdf
│   └── full_night_d1c690da.pdf
│
├── em_segments/                 (Reserved for per-segment EM figures)
└── stable_ss/                   (Requires full cohort HF5 files)
```

---

## Quick Start

### Run the full demo pipeline on one patient

```bash
cd python
uv run python -m scripts.run_full_pipeline_demo
```

This runs all 5 stages on patient `7b1e2d31` and saves outputs to `figures/demo/`.
Takes ~70 seconds (3 NREM segments × ~16s each for EM).

### Generate all available publication figures

```bash
uv run python -m scripts.generate_all_figures
```

### Run the EM algorithm on a study CSV

```bash
# Single study
uv run python -m scripts.run_em_python --input data/em_example/Study_example.csv

# All studies in a folder
uv run python -m scripts.run_em_python --input-dir data/em_input/ --output-dir data/em_output/
```

### Generate full-night figures

```bash
uv run python -m scripts.run_full_night_example
```

---

## Scripts Reference

| Script | What it does | Input | Output |
|--------|-------------|-------|--------|
| `run_full_pipeline_demo.py` | End-to-end demo on 1 patient | HF5 file | `figures/demo/` |
| `run_em_python.py` | Run EM on study CSVs | EM input CSV | EM output CSV |
| `run_full_night_example.py` | Full-night overview figures | HF5 files | `figures/full_night/` |
| `run_group_analysis.py` | Cohort boxplots (LG/gamma/tau) | `interm_Results/` | `figures/cohort_boxplots/` |
| `run_ss_relationship.py` | SS-vs-LG scatter + swimmer plot | `interm_Results/` | `figures/ss_relationship/` |
| `run_cpap_analysis.py` | CPAP ROC/PR curves | `interm_Results/` + `bars/` | `figures/cpap_prediction/` |
| `run_altitude_analysis.py` | Altitude spaghetti plots | `interm_Results/` | `figures/altitude/` |
| `run_stable_ss.py` | Stable SS detection + histograms | Full HF5 cohort | `figures/stable_ss/` |
| `update_mgh_info.py` | Enrich MGH clinical table | Full HF5 cohort | Updated CSV |
| `generate_all_figures.py` | Run all figure scripts | All data | All `figures/` |

---

## Module Map

| Pipeline Stage | Module | Key Function |
|---------------|--------|-------------|
| Load HF5 | `hlg.io.readers` | `load_sim_output()` |
| Preprocess | `hlg.core.preprocessing` | `do_initial_preprocessing()` |
| Ventilation | `hlg.core.ventilation` | `create_ventilation_trace()` |
| Segment | `hlg.ss.segmentation` | `segment_data_based_on_nrem()` |
| SS Score | `hlg.ss.scoring` | `convert_ss_seg_scores_into_arrays()` |
| Export CSV | `hlg.ss.pipeline` | `segment_and_export_recording()` |
| EM Algorithm | `hlg.em.em_algorithm` | `run_em_on_segment()` |
| Mackey-Glass | `hlg.em.mackey_glass` | `state_space_loop()` |
| Arousal | `hlg.em.arousal` | `estimate_arousals()` |
| Loop Gain | `hlg.em.loop_gain_calc` | `compute_loop_gain()` |
| Post-process | `hlg.em.postprocessing` | `post_process_EM_output()` |
| LG Array | `hlg.em.loop_gain` | `create_total_LG_array()` |
| Study Runner | `hlg.em.run_em` | `process_study()` |
| Report | `hlg.reporting` | `create_report()` |
| Full Night Fig | `hlg.visualization.full_night` | `plot_full_night()` |
| Segment Fig | `hlg.visualization.segments` | `plot_EM_output_per_segment()` |

---

## Performance

Per 8-minute segment (4,800 samples at 10 Hz):
- **Pure Python**: ~16 seconds
- **With Numba JIT** (future): ~0.5 seconds (estimated 30x speedup)
- **MATLAB original**: ~3 seconds

Full study (44 segments): ~12 minutes in pure Python.
