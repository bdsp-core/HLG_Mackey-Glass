"""
Centralized configuration for the HLG pipeline.

All hardcoded paths and magic constants that were previously scattered
across individual scripts are consolidated here. Values can be overridden
via environment variables or by mutating the module-level `Config` instance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Resolve the project root (python/) relative to this file's location,
# regardless of the working directory the script is launched from.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"


@dataclass
class HLGConfig:
    """
    Central configuration object for the HLG analysis pipeline.

    Attributes are grouped by subsystem and documented with their
    physiological or algorithmic rationale.
    """

    # ── Sampling ──────────────────────────────────────────────────────────
    # All recordings are downsampled from the original PSG rate (typically
    # 200 Hz) to 10 Hz. This is sufficient for respiratory signals (cycle
    # period ~3-6 s) while dramatically reducing memory and compute cost.
    default_fs: int = 10

    # ── Signal Filtering ──────────────────────────────────────────────────
    # US power-line noise is at 60 Hz; a notch filter removes it from EEG
    # and respiratory channels before any downstream analysis.
    notch_freq_us: float = 60.0

    # EEG bandpass: 0.1-20 Hz captures delta through beta activity while
    # rejecting DC drift and high-frequency EMG/noise.
    bandpass_freq_eeg: tuple[float, float] = (0.1, 20.0)

    # Respiratory bandpass: the default 0-10 Hz is intentionally wide so
    # that ventilation envelope computation is not distorted. The "extreme"
    # variant (0.1-2 Hz) can be used for noisy abdominal signals.
    bandpass_freq_breathing: tuple[float, float] = (0.0, 10.0)
    bandpass_freq_breathing_extreme: tuple[float, float] = (0.1, 2.0)

    # ECG high-pass at 0.3 Hz removes baseline wander while preserving QRS.
    bandpass_freq_ecg: tuple[float | None, float | None] = (0.3, None)

    # ── EM Estimation ─────────────────────────────────────────────────────
    # Segments whose root-mean-square error (RMSE) between observed and
    # modeled ventilation exceeds this threshold are excluded from group
    # analyses. The value 1.8 was empirically determined to reject poor
    # fits without being overly conservative.
    error_threshold: float = 1.8

    # ── Self-Similarity ───────────────────────────────────────────────────
    # The SS convolution score threshold above which a breath is considered
    # "self-similar" (i.e., part of a periodic breathing oscillation).
    ss_threshold: float = 0.5

    # ── Segmentation ──────────────────────────────────────────────────────
    # EM input segments are 8-minute blocks, chosen to contain enough
    # breathing cycles (~100-160 at 12-20 breaths/min) for reliable LG
    # estimation, while being short enough to capture within-night changes.
    segment_block_size_min: int = 8

    # ── File Paths ────────────────────────────────────────────────────────
    # All default paths resolve relative to python/data/.
    # Override via environment variables for different machines / layouts.
    csv_dir: str = field(default_factory=lambda: os.environ.get("HLG_CSV_DIR", str(_DATA_DIR / "csv_files")))
    hf5_dir: str = field(default_factory=lambda: os.environ.get("HLG_HF5_DIR", str(_DATA_DIR / "hf5_examples")))
    bars_dir: str = field(default_factory=lambda: os.environ.get("HLG_BARS_DIR", str(_DATA_DIR / "bars")))
    interm_dir: str = field(default_factory=lambda: os.environ.get("HLG_INTERM_DIR", str(_DATA_DIR / "interm_Results")))
    output_dir: str = field(default_factory=lambda: os.environ.get("HLG_OUTPUT_DIR", str(_PROJECT_ROOT / "figures")))

    # ── Known Bad Recordings ──────────────────────────────────────────────
    # These recording IDs were manually identified as having poor signal
    # quality (e.g., disconnected belts, severe artifact) and are excluded
    # from cohort-level analyses.
    bad_recording_ids: tuple[str, ...] = (
        "b346da6",
        "dd20181",
        "b551d67",
        "0be8481",
        "fe5cdc8",
        "97a9256",
        "a58f9df",
        "76f60a8",
        "9b91a5f",
        "ad0ee71",
        "f3a9d3e",
        "f1487f4",
        "d300222",
        "2fbff9f",
        "8ed09d7",
        "e7de9ac",
        "e0e70ec",
        "55a330f",
        "d886538",
        "204b19a",
        "15a8620",
        "8c4d264",
    )


# Module-level singleton -- import and use directly, or replace for testing.
config = HLGConfig()
