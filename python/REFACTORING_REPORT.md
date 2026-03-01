# HLG Codebase Refactoring Report

Complete record of the refactoring process applied to the HLG (High Loop Gain) research codebase. Use this as a template/instruction set for refactoring other research codebases in the same style.

---

## Starting Point

- **19 flat Python scripts** in a single directory, no package structure
- No `pyproject.toml`, `requirements.txt`, `.gitignore`, or README
- No tests
- Duplicated functions across files (e.g. `load_sim_output` copied 3 times)
- Hardcoded file paths (Dropbox, `/media/cdac/`, Windows paths)
- Mixed concerns: data-processing functions inside plotting files
- `__main__` blocks with significant pipeline logic mixed with configuration
- No type hints, minimal docstrings, sparse comments
- `import pdb; pdb.set_trace()` scattered throughout

---

## Phase 1: Investigation

Before writing any code, we did a **thorough codebase investigation**:

1. **Listed all files** and their sizes (~4,900 lines across 19 files)
2. **Read every file** to understand:
   - All function signatures, parameters, and return types
   - What each function does (1-2 sentence summary)
   - All imports (third-party and local)
   - Inter-file dependencies (which file imports from which)
   - Module-level constants and configuration values
3. **Built a dependency graph** showing the call hierarchy
4. **Identified problems**:
   - Code duplication (3 copies of `load_sim_output`, 2 copies of `predband`, `func`, `sort_dic_keys`, `add_statistical_significance`)
   - Misplaced functions (`add_arousals`, `match_EM_with_SS_output` living in a "Figures" file but used as data-processing utilities)
   - Hardcoded paths to specific machines
   - No separation between library code, scripts, and configuration

---

## Phase 2: Architecture Design

Designed the new package structure following these principles:

### Principles

1. **Layered dependency flow** — lower layers have no local deps; higher layers import downward only
2. **Separation of concerns** — data processing, I/O, analysis, and visualization in separate sub-packages
3. **Single source of truth** — deduplicate all copied functions into one canonical location
4. **Configuration centralisation** — all magic constants and file paths in one `config.py` dataclass
5. **`src/` layout** — uses the modern `src/pkg/` convention to avoid import shadowing during development
6. **Scripts separate from library** — `__main__` block logic extracted into `scripts/` directory

### Package Structure

```
hlg_v2/
├── pyproject.toml              # Build system, dependencies, tool config
├── README.md                   # Comprehensive documentation
├── .gitignore                  # Python + data file exclusions
├── pyrightconfig.json          # Type checker configuration
├── .vscode/settings.json       # IDE: format-on-save, test discovery
├── src/hlg/
│   ├── __init__.py             # Package docstring + version
│   ├── config.py               # HLGConfig dataclass (all paths + constants)
│   ├── core/                   # Layer 0: no local deps
│   │   ├── events.py           # Event detection (find_events, etc.)
│   │   ├── preprocessing.py    # Filtering, resampling, normalisation
│   │   ├── ventilation.py      # Ventilation envelope + trace computation
│   │   └── sleep_metrics.py    # RDI, AHI, CAI computation
│   ├── io/                     # Layer 1: depends on core
│   │   ├── readers.py          # HDF5 loading (deduplicated)
│   │   └── writers.py          # HDF5 + MATLAB export
│   ├── ss/                     # Layer 2: depends on core, io
│   │   ├── scoring.py          # SS score arrays
│   │   ├── segmentation.py     # NREM/REM block segmentation
│   │   ├── stable.py           # Oscillation chains, change-point detection
│   │   └── pipeline.py         # Patient selection, sorting, export
│   ├── em/                     # Layer 2: depends on core, io, ss
│   │   ├── extraction.py       # Parallel EM output extraction
│   │   ├── postprocessing.py   # LG smoothing, arousal matching
│   │   ├── histograms.py       # LG histogram bars for CPAP prediction
│   │   └── loop_gain.py        # Full-night LG array reconstruction
│   ├── analysis/               # Layer 3: depends on everything below
│   │   ├── statistics.py       # Shared: prediction bands, significance tests
│   │   ├── cpap.py             # Logistic regression for CPAP outcome
│   │   ├── group.py            # Cohort-level comparisons
│   │   ├── ss_relationship.py  # SS-vs-LG polynomial regression
│   │   └── altitude.py         # Altitude study analysis
│   ├── visualization/          # Layer 3: pure plotting, no data processing
│   │   ├── segments.py         # Per-segment multi-panel figures
│   │   ├── full_night.py       # Full-night overview plots
│   │   ├── histograms.py       # LG histogram bar charts
│   │   └── stable_ss.py        # SS detection plots + length histograms
│   └── reporting.py            # Clinical summary reports
├── scripts/                    # Entrypoint scripts (from __main__ blocks)
│   ├── generate_all_figures.py # Master orchestrator
│   ├── run_em_extraction.py
│   ├── run_group_analysis.py
│   ├── run_cpap_analysis.py
│   ├── run_ss_relationship.py
│   ├── run_altitude_analysis.py
│   ├── run_stable_ss.py
│   └── update_mgh_info.py
├── tests/
│   └── test_core_events.py     # 19 tests for the most-used utility module
├── data/                       # All input data (gitignored except CSV)
│   ├── csv_files/              # Patient metadata tables
│   ├── hf5_examples/           # Example HDF5 recordings
│   ├── bars/                   # Pre-computed LG histogram bars
│   └── interm_Results/         # Intermediate pipeline outputs
└── figures/                    # Generated figures (gitignored)
    ├── cohort_boxplots/
    ├── cpap_prediction/
    ├── ss_relationship/
    ├── altitude/
    └── stable_ss/
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `src/hlg/` layout | Prevents accidental imports from the working directory |
| `config.py` with `@dataclass` | Single place for all paths and constants; overridable via env vars |
| Deduplication into canonical modules | 3 copies of `load_sim_output` → 1 in `io/readers.py` |
| Plotting separated from processing | `em/postprocessing.py` has the data functions; `visualization/segments.py` has the plots |
| `scripts/` with `def main()` | Clean entrypoints, no side effects on import, orchestratable |
| `_PROJECT_ROOT` in config | Paths resolve correctly regardless of working directory |

---

## Phase 3: Implementation

### 3.1 — Scaffold

Created all directories, `__init__.py` files, and foundational files:
- `pyproject.toml` with hatchling build, all dependencies, ruff + pytest config
- `.gitignore` for Python, data files, IDE artifacts
- `config.py` with `HLGConfig` dataclass

### 3.2 — Core Modules (no local deps)

Refactored the 4 leaf modules that everything else depends on:
- `core/events.py` ← `Event_array_modifiers.py`
- `core/preprocessing.py` ← `Preprocessing.py`
- `core/ventilation.py` ← `Create_Ventilation.py` + `Ventilation_envelope.py` (merged)
- `core/sleep_metrics.py` ← `Compute_sleep_metrics.py`

### 3.3 — I/O Modules

- `io/readers.py` ← deduplicated `load_sim_output` from 3 files + `load_SS_percentage`
- `io/writers.py` ← `Data_writers.py`

### 3.4 — SS Modules

- `ss/scoring.py` ← `Convert_SS_seg_scores.py`
- `ss/segmentation.py` ← `segment_data_based_on_nrem` + `compute_SS_score_per_segement` from `SS_output_to_EM_input.py`
- `ss/stable.py` ← `compute_osc_chains` + `compute_change_points_ruptures` from `Stable_SS_analysis.py`
- `ss/pipeline.py` ← `patient_selection`, `sort_input_files`, `sort_altitude_files`, export functions from `SS_output_to_EM_input.py`

### 3.5 — EM Modules

- `em/loop_gain.py` ← `Recreate_LG_array.py`
- `em/postprocessing.py` ← data-processing functions extracted from `EM_output_to_Figures.py` (`match_EM_with_SS_output`, `add_arousals`, `post_process_EM_output`, etc.)
- `em/histograms.py` ← `EM_output_histograms.py`
- `em/extraction.py` ← `EM_output_extraction.py` (parallel pipeline)

### 3.6 — Analysis Modules

- `analysis/statistics.py` ← deduplicated `predband`, `func`, `sort_dic_keys`, `add_statistical_significance`
- `analysis/cpap.py` ← `EM_output_to_CPAP_Analysis.py`
- `analysis/group.py` ← `EM_output_to_Group_Analysis.py`
- `analysis/ss_relationship.py` ← `EM_output_to_SS_Relationship.py`
- `analysis/altitude.py` ← `EM_output_to_Alitude_Relationship.py`

### 3.7 — Visualization Modules

Plotting functions extracted from their original mixed-concern files:
- `visualization/segments.py` ← `plot_EM_output_per_segment`
- `visualization/full_night.py` ← `plot_full_night`, `add_LG_hooks`, `find_row_location`
- `visualization/histograms.py` ← `total_histogram_plot`
- `visualization/stable_ss.py` ← `plot_SS`, `create_length_histogram`

### 3.8 — Reporting

- `reporting.py` ← `Save_and_Report.py`

### 3.9 — Scripts

Extracted `__main__` blocks into proper entrypoint scripts:
- `scripts/run_em_extraction.py`
- `scripts/run_group_analysis.py`
- `scripts/run_cpap_analysis.py`
- `scripts/run_ss_relationship.py`
- `scripts/run_altitude_analysis.py`
- `scripts/run_stable_ss.py`
- `scripts/update_mgh_info.py`
- `scripts/generate_all_figures.py` (master orchestrator)

All scripts:
- Wrap logic in `def main()` + `if __name__ == "__main__": main()`
- Use `config` for paths instead of hardcoded strings
- Save figures to `figures/` directory instead of `plt.show()`
- Remove all `import pdb; pdb.set_trace()`

### 3.10 — Tests

Created `tests/test_core_events.py` with 19 tests covering:
- `find_events` (7 tests: empty, single, multiple, boundary, full-signal, different labels)
- `events_to_array` (3 tests: roundtrip, custom labels, empty)
- `window_correction` (2 tests: extension, boundary clipping)
- `connect_events` (5 tests: merge, no-merge, empty, labels, edge cases)

### 3.11 — README

Comprehensive README.md with:
- Project overview (what loop gain is, what the package does)
- Full package tree diagram
- Installation instructions
- Quick-start usage examples
- Mermaid architecture flowchart
- Module reference table (old file → new location)
- Configuration documentation
- Data format documentation

---

## Phase 4: Code Quality (Linting & Formatting)

### 4.1 — Ruff Configuration

Added to `pyproject.toml`:
```toml
[tool.ruff.lint]
select = ["E", "F", "B", "C4", "UP"]
ignore = [
    "E501",  # line length (handled by formatter)
    "B905",  # zip-without-strict (numpy parallel arrays)
    "E712",  # == True/False (numpy element-wise, not Python truthiness)
]
```

### 4.2 — Auto-fix Pass

```bash
ruff check src/ scripts/ tests/ --fix   # Fixed 306 issues automatically
ruff format src/ scripts/ tests/        # Reformatted 26 files
```

Auto-fixes included:
- **Type annotation modernisation**: `List[int]` → `list[int]`, `Tuple` → `tuple`, `Optional` → `| None`
- **Unused import removal**: 28 unused imports removed
- **Import sorting**: all imports organised consistently

### 4.3 — Manual Fixes (24 remaining issues)

| Category | Count | Fix |
|----------|-------|-----|
| `UP031` % format strings | 3 | Converted to f-strings |
| `B006` mutable defaults | 3 | `labels=[]` → `labels=None` + guard clause |
| `B007` unused loop vars | 3 | Prefixed with `_` |
| `F841` unused variables | 10 | Prefixed with `_` |
| `C417` unnecessary map | 1 | Replaced with list comprehension |
| `C419` unnecessary list in any() | 1 | Changed to generator expression |
| `E721` type comparison | 1 | `type(x) == str` → `isinstance(x, str)` |
| `C408` dict() call | 1 | Replaced with `{}` literal |

### 4.4 — IDE Configuration

`.vscode/settings.json`:
- Format-on-save with Ruff
- Auto-fix (remove unused imports) on save
- Auto-organise imports on save
- Pytest test discovery
- Python interpreter pointed at `.venv`
- `python.analysis.extraPaths` for `src/` layout

`pyrightconfig.json`:
- Includes `src/`, `scripts/`, `tests/`
- Points at `.venv`
- `typeCheckingMode: "basic"`

---

## Phase 5: Bug Discovery & Fix

During testing, discovered a **latent bug in `connect_events`**:

**The bug**: After the while-loop that merges adjacent events, the function unconditionally appends `events[-1]`. When the last two events were already merged (causing the loop counter to jump past the end), this duplicates the last event.

**The fix**: Added a guard — only append when `cnt == len(events) - 1` (last event wasn't consumed by a merge).

**Impact**: The function wasn't actively called in any pipeline, so the bug was latent. But it would have produced incorrect results for any future caller.

**Tests added**: 3 additional edge-case tests (first-pair merge, last-pair merge, single event).

---

## Phase 6: Data Organisation

Moved all data into a `data/` directory:

```
data/
├── csv_files/           9 patient metadata CSVs
├── hf5_examples/        4 example HDF5 recordings
├── bars/                Per-patient LG histogram bars (CPAP cohorts)
└── interm_Results/      Pre-computed intermediate pipeline outputs
    └── non-smooth/
        ├── bdsp_CPAP_failure/
        ├── bdsp_CPAP_success/
        ├── mgh_REM_OSA/
        ├── mgh_NREM_OSA/
        ├── mgh_high_CAI/
        ├── mgh_SS_OSA/
        ├── mgh_SS_range/
        ├── redeker_Heart_Failure/
        └── rt_Altitude/
```

Updated `config.py` to resolve all paths relative to `_PROJECT_ROOT = Path(__file__).resolve().parents[2]`, with environment variable overrides for each path.

---

## Phase 7: Figure Generation

### 7.1 — Figures Directory

Created `figures/` with subdirectories for each analysis type. All scripts save to this directory instead of calling `plt.show()`.

### 7.2 — Master Script

`scripts/generate_all_figures.py`:
- Imports and runs each figure script via `importlib.import_module`
- Gracefully handles missing data (reports SKIPPED)
- Supports `--only cohort cpap` for running a subset
- Prints summary table at the end

### 7.3 — Path Fixes

Updated all scripts to use `config.interm_dir`, `config.csv_dir`, `config.bars_dir` instead of hardcoded relative paths. Mapped folder names correctly (e.g. `"REM_OSA"` → `"mgh_REM_OSA"` in the actual directory structure).

### 7.4 — Runtime Bug Fixes

- `altitude.py`: `LG_all[~valids] = 0` failed on read-only numpy array from CSV. Fixed with `np.array(LG_all, copy=True)`.
- Altitude script: `base_folder` was a relative path `./interm_Results/...` instead of using `config.interm_dir`.

### 7.5 — Figure Styling

Refined the cohort boxplot figure to match publication reference:
- Removed all spines except custom-drawn y-axis lines
- Set exact y-axis tick values matching the reference (LG: 0.0–1.5 by 0.3, γ: 0.1–1.1 by 0.2, τ: 10–50 by 10)
- Extended y-axis lines beyond tick range
- Removed mean markers (green triangles)
- Hardcoded legend position with manual text placement
- Narrowed figure aspect ratio (5.5 × 10)
- Added cohort labels with sample sizes on bottom panel only

---

## Phase 8: Original Code Preservation

Moved all 19 original scripts + `csv_files/` into `hlg_v1/` so the repo root is clean:

```
GitHub - HLG/
├── hlg_v1/     ← Original flat scripts (preserved for reference)
└── hlg_v2/     ← Refactored package
```

---

## Summary of Rules Followed

1. **Preserve all algorithm logic exactly** — no changes to thresholds, formulas, signal processing parameters, or mathematical operations
2. **Add extensive comments** explaining domain-specific rationale (why 60 Hz notch, what loop gain means, why error_threshold = 1.8, etc.)
3. **Google-style docstrings** with Args/Returns sections on every function
4. **Type hints** on all function signatures
5. **Package-style imports** (`from hlg.core.events import find_events`)
6. **Centralised configuration** via `config.py` dataclass with env var overrides
7. **Deduplication** of all copied functions into single canonical locations
8. **Separation of concerns** — data processing and plotting in different modules
9. **`__main__` blocks** extracted into `scripts/` as `def main()` functions
10. **Remove all debug artifacts** (`pdb.set_trace()`, commented-out code blocks)
11. **Ruff linting + formatting** with auto-fix on save
12. **Tests** for the most critical/widely-used module

---

## Final Metrics

| Metric | Before | After |
|--------|--------|-------|
| Files | 19 flat scripts | 45+ organised modules |
| Package structure | None | 7 sub-packages + scripts + tests |
| Dependencies declared | None | `pyproject.toml` with 11 deps |
| `.gitignore` | None | Python + data + figures |
| Type hints | ~5% of functions | 100% of functions |
| Docstrings | ~20% of functions | 100% of functions |
| Tests | 0 | 19 passing |
| Lint errors | N/A | 0 |
| Duplicated functions | 8+ | 0 |
| Hardcoded paths | 15+ | 0 (all in config) |
| `pdb.set_trace()` | 5 | 0 |
| Documentation | None | README + module docstrings |
