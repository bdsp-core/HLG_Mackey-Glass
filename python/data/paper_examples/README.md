# Paper Example Data — Figure 1 Patients

This directory should contain the 4 preprocessed CSV files used to reproduce
the paper's Figure 1 (per-segment EM fits) and Figure 2 (full-night overviews).

## Required Files

| File | Study | Group | Patient (short) | Description |
|------|-------|-------|-----------------|-------------|
| `figure1_studies.csv` | — | — | — | Metadata mapping (demographics, clinical indices) |
| `high_CAI_Study_99.csv` | 99 | High CAI | `4e504ee4` | Cheyne-Stokes, centrally driven events (Fig. 1 A–B) |
| `HLG_OSA_Study_97.csv` | 97 | HLG OSA | `39fc3416` | Obstructive apneas with varying LG (Fig. 1 C–D) |
| `NREM_OSA_Study_5.csv` | 5 | NREM OSA | `d1c690da` | Hypopneas with low LG (Fig. 1 E) |
| `HLG_OSA_Study_7.csv` | 7 | HLG OSA | `7b1e2d31` | Hypopneas with moderate LG (Fig. 1 F) |

## Figure 1 Panel Mapping

| Panel | Study | NREM Segment | Row Range | LG | γ | τ | Description |
|-------|-------|-------------|-----------|-----|------|-----|-------------|
| A | 99 (High CAI) | seg 1 | 9000–13800 | 1.87 | 0.80 | 25s | CSR with high LG |
| B | 99 (High CAI) | seg 14 | 82200–87000 | 3.85 | 1.46 | 31s | CSR with very high LG |
| C | 97 (HLG OSA) | seg 8 | 64800–69600 | 0.97 | 0.62 | 20s | Obstructive, moderate LG |
| D | 97 (HLG OSA) | seg 20 | 158400–163200 | 1.10 | 0.57 | 18s | Obstructive, high LG |
| E | 5 (NREM OSA) | seg 6 | 54600–59400 | 0.11 | 0.10 | 38s | Hypopneas, low LG |
| F | 7 (HLG OSA) | seg 14 | 75300–80100 | 0.39 | 0.33 | 27s | Hypopneas, moderate LG |

## Data Source

Data and code are available at [bdsp.io/content/hlg-data-code/2.0/](https://bdsp.io/content/hlg-data-code/2.0/). The preprocessed CSV files are included in this directory.

## Optional: Raw H5 Files

For the end-to-end pipeline (`run_end_to_end.py`), place raw BDSP H5 recordings
in `data/raw_h5/`:

| File | Study | BDSP Patient ID |
|------|-------|-----------------|
| `sub-S0001111922082_ses-1.h5` | 5 | 111922082 |
| `sub-S0001111985952_ses-1.h5` | 7 | 111985952 |
| `sub-S0001114591660_ses-1.h5` | 97 | 114591660 |
| `sub-S0001116587855_ses-1.h5` | 99 | 116587855 |
