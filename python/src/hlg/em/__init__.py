"""
Estimation Model (EM) output processing.

The EM fits a physiological model of ventilatory control to each
8-minute sleep segment, estimating the loop gain (LG), controller
gain (gamma), and circulation delay (tau). This sub-package handles
extraction, post-processing, and histogram summarization of EM output.

Modules:
    loop_gain       - Sample-level LG array reconstruction.
    postprocessing  - LG outlier smoothing, arousal disentanglement, metadata linkage.
    histograms      - LG histogram computation and CPAP-response prediction.
    extraction      - Parallel per-patient EM extraction and aggregation.
"""
