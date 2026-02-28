"""
HLG -- Loop Gain estimation and analysis for sleep-disordered breathing.

This package processes polysomnography (PSG) data to estimate respiratory
loop gain (LG), a physiological parameter that quantifies the instability
of the ventilatory control system. High loop gain is a key endotype of
obstructive sleep apnea and central sleep apnea.

Package structure:
    hlg.core           - Low-level signal processing primitives (no local deps)
    hlg.io             - File I/O for HDF5, MATLAB, and CSV formats
    hlg.ss             - Self-Similarity (SS) analysis and segmentation
    hlg.em             - Estimation Model (EM) output processing
    hlg.analysis       - Statistical analyses (CPAP prediction, group comparison)
    hlg.visualization  - Plotting and figure generation
    hlg.reporting      - Clinical summary report generation
"""

__version__ = "0.1.0"
