# HLG: Loop Gain Estimation for Sleep-Disordered Breathing

Automated estimation of ventilatory loop gain (LG) and its control parameters from respiratory inductance plethysmography (RIP) signals during sleep.

This repository contains the complete codebase for the method described in:

> Nassi T, Amidi Y, Oppersma E, Donker DW, Redeker NS, Westover MB, Thomas RJ.
> **Unraveling sleep apnea dynamics: quantifying loop gain using dynamical modeling of ventilatory control.**
> *SLEEP*, 2026, 49(2), zsaf213. [doi:10.1093/sleep/zsaf213](https://doi.org/10.1093/sleep/zsaf213)

The published manuscript is included in [`docs/LG_Manuscript.pdf`](docs/LG_Manuscript.pdf).

## Repository Structure

```
HLG_Mackey-Glass/
  docs/                  Published paper
  python/                Complete Python package (EM algorithm + analysis + figures)
  _original/             Untouched backup copies of all original files
```

### `python/` -- Complete Analysis Pipeline

A self-contained Python package (`hlg`) that implements the **entire pipeline** from raw polysomnography recordings to publication figures:

- **EM parameter estimation** -- Mackey-Glass ventilatory control model fitting via grid search, with Numba JIT acceleration (~22x speedup, identical results to original MATLAB)
- **Signal preprocessing** -- notch filtering, band-pass filtering, resampling, normalization
- **Self-Similarity (SS) analysis** -- oscillation detection, change-point segmentation, stable SS regions
- **Loop gain computation** -- steady-state analysis of the fitted model
- **Statistical analysis** -- cohort comparisons, CPAP prediction, altitude effects
- **Publication figures** -- all 8 paper figures reproducible from code

No MATLAB installation required.

See [`python/README.md`](python/README.md) for setup and [`python/PIPELINE.md`](python/PIPELINE.md) for the end-to-end workflow.

### `_original/` -- Untouched Backups

Preserved copies of the original code before refactoring:

- `_original/matlab/` -- 17 original MATLAB `.m` files (camelCase naming, flat structure)
- `_original/hlg_v1/` -- 19 original Python scripts (flat structure, no package)

## Quick Start

```bash
cd python
uv sync --extra dev        # Install dependencies
uv run python -m pytest    # Run tests (19 pass)

# Reproduce exact paper Figure 1 panels A–F (~20s)
uv run python -m scripts.run_paper_figures --paper-panels

# Reproduce paper Figure 1 + 2 for all 4 example patients (~12s)
uv run python -m scripts.run_paper_figures

# Generate all publication figures
uv run python -m scripts.generate_all_figures
```

## Paper Figure Mapping

Every figure in the published paper can be reproduced from this codebase:

| Figure | Description | Script |
|--------|-------------|--------|
| Fig. 1 | Per-segment EM fits with CO2 model | `scripts/run_paper_figures.py` |
| Fig. 2 | Full-night overview with LG hooks | `scripts/run_paper_figures.py` |
| Fig. 3 | LG / gamma / tau boxplots across cohorts | `scripts/run_group_analysis.py` |
| Fig. 4 | Swimmer plots and LG bar graphs | `scripts/run_ss_relationship.py` |
| Fig. 5 | SS vs LG scatter with polynomial regression | `scripts/run_ss_relationship.py` |
| Fig. 6 | NREM vs REM LG boxplots | `scripts/run_ss_relationship.py` |
| Fig. 7 | Altitude LG histograms and spaghetti plots | `scripts/run_altitude_analysis.py` |
| Fig. 8 | CPAP failure prediction (ROC/PR/calibration) | `scripts/run_cpap_analysis.py` |

## Pipeline Overview

```
RIP Signal (from PSG)
        |
        v
  [Self-Similarity Detection]  -->  SS scores, stable regions
        |
        v
  [Preprocessing & Segmentation]  -->  8-min NREM/REM windows
        |
        v
  [EM Algorithm (Python + Numba)]  -->  gamma, tau, alpha, LG per segment
        |
        v
  [Post-Processing]  -->  outlier smoothing, quality filtering
        |
        v
  [Analysis & Figures]  -->  cohort comparisons, CPAP prediction,
                              full-night overviews, strip charts
```

## Validation

The Python EM implementation produces **identical results** to the original MATLAB code. This was verified by:

1. **Mathematical property tests** -- steady-state convergence, loop gain monotonicity, arousal pulse shape, RMSE self-consistency, parameter recovery on synthetic data
2. **Direct comparison** -- GNU Octave cross-validation on real patient data: gamma, tau, alpha, LG, and arousal count all match exactly across all 5 EM iterations

See `python/tests/validate_matlab_vs_python.m` and `python/tests/matlab_em_results.csv`.

## License

This code is licensed under CC BY-NC 4.0 (Attribution-NonCommercial 4.0). See [LICENSE](LICENSE) for details.

## Data Availability

Data and code are available at [bdsp.io/content/hlg-data-code/2.0/](https://bdsp.io/content/hlg-data-code/2.0/). Preprocessed CSV files for the 4 example patients used in the paper figures are included in [`python/data/paper_examples/`](python/data/paper_examples/).
