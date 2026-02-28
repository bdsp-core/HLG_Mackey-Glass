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
  matlab/                MATLAB EM algorithm (parameter estimation)
  python/                Python analysis and figure generation package
  _original/             Untouched backup copies of all original files
```

### `matlab/` -- EM Parameter Estimation (MATLAB)

The core Expectation-Maximization algorithm that fits an augmented Mackey-Glass ventilatory control model to 8-minute segments of RIP data. Estimates loop gain (LG), controller gain (gamma), circulation delay (tau), and arousal parameters.

See [`matlab/README.md`](matlab/README.md) for setup and usage.

### `python/` -- Analysis, Visualization, and Figure Generation (Python)

A complete Python package (`hlg`) for post-processing EM output, statistical analysis, and generating all publication figures. Includes the self-similarity (SS) pipeline integration, cohort comparisons, CPAP prediction, altitude analysis, and full-night "eye test" visualizations.

See [`python/README.md`](python/README.md) for setup and usage.

## Paper Figure Mapping

Every figure in the published paper can be reproduced from this codebase:

| Figure | Description | Code |
|--------|-------------|------|
| Fig. 1 | Six 8-min segment examples with EM fits | `python/src/hlg/visualization/segments.py` |
| Fig. 2 | Full-night "eye test" overview with LG hooks | `python/src/hlg/visualization/full_night.py` |
| Fig. 3 | LG / gamma / tau boxplots across cohorts | `python/scripts/run_group_analysis.py` |
| Fig. 4 | Swimmer plots and LG bar graphs | `python/scripts/run_ss_relationship.py` |
| Fig. 5 | SS vs LG scatter with polynomial regression | `python/scripts/run_ss_relationship.py` |
| Fig. 6 | NREM vs REM LG boxplots | `python/scripts/run_ss_relationship.py` |
| Fig. 7 | Altitude LG histograms and spaghetti plots | `python/scripts/run_altitude_analysis.py` |
| Fig. 8 | CPAP failure prediction (ROC/PR/calibration) | `python/scripts/run_cpap_analysis.py` |

To generate all figures at once:

```bash
cd python
uv sync --extra dev
uv run python -m scripts.generate_all_figures
```

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
  [EM Algorithm (MATLAB)]  -->  gamma, tau, alpha, Vmax, LG per segment
        |
        v
  [Post-Processing (Python)]  -->  outlier smoothing, quality filtering
        |
        v
  [Analysis & Figures (Python)]  -->  cohort comparisons, scatter plots,
                                      full-night overviews, CPAP prediction
```

## License

This code is licensed under CC BY-NC 4.0 (Attribution-NonCommercial 4.0). See [LICENSE](LICENSE) for details.

## Data Availability

Data are available at [bdsp.io](https://bdsp.io/) upon request to the corresponding author.
