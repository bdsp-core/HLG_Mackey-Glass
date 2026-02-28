"""
Signal preprocessing for polysomnography (PSG) channels.

This module handles the two main preprocessing stages applied to raw PSG
signals before any analysis:

1. **Initial preprocessing** -- powerline-noise removal (60 Hz notch filter),
   band-pass filtering appropriate to each signal modality, and resampling to
   a common target rate.
2. **Clipping and normalisation** -- outlier suppression via IQR-based clipping
   for EEG/EMG/ECG, percentile-based clipping for respiratory signals, and
   robust z-score normalisation so that all channels share a comparable scale.

Every threshold and filter parameter in this module was empirically tuned on
clinical PSG datasets and must be preserved exactly.

No local package dependencies -- only numpy, pandas, scipy, sklearn, and mne.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from scipy.signal import resample_poly
from mne.filter import filter_data, notch_filter


# ---------------------------------------------------------------------------
# Channel groupings used for modality-specific processing
# ---------------------------------------------------------------------------
_EEG_EMG_CHANNELS = [
    "f3-m2",
    "f4-m1",
    "c3-m2",
    "c4-m1",
    "o1-m2",
    "o2-m1",
    "e1-m2",
    "chin1-chin2",
]
_RESPIRATORY_CHANNELS = [
    "abd",
    "chest",
    "airflow",
    "ptaf",
    "cflow",
    "breathing_trace",
]
_ALL_ANALOG_CHANNELS = _EEG_EMG_CHANNELS + _RESPIRATORY_CHANNELS + ["ecg"]


def do_initial_preprocessing(
    signals: pd.DataFrame,
    new_Fs: int,
    original_Fs: int,
    extreme_resp_filtering: bool = False,
) -> pd.DataFrame:
    """Apply powerline-notch, band-pass filtering, and resampling.

    Each signal modality (EEG/EMG, respiratory, ECG) receives a tailored
    filter chain:

    * **60 Hz notch** -- removes US powerline interference.  Only applied when
      the original sample rate is at least 120 Hz (Nyquist criterion).
    * **EEG/EMG band-pass 0.1-20 Hz** -- retains sleep-relevant brain rhythms
      (delta through sigma) and chin EMG while discarding DC drift and
      high-frequency muscle artefact.
    * **Respiratory band-pass** -- ``[0.1, 2] Hz`` in extreme mode (isolates
      the fundamental breathing frequency ~0.2 Hz and its first harmonic)
      or ``[0, 10] Hz`` in normal mode (preserves the full waveform shape
      needed for flow-limitation analysis).
    * **ECG high-pass 0.3 Hz** -- removes baseline wander while preserving
      the QRS complex; no low-pass because R-peak detection benefits from
      sharp edges.

    After filtering, each channel is resampled to ``new_Fs``.  Non-analog
    channels (e.g. sleep stage labels, SpO2) are resampled by simple
    nearest-neighbour repeat-and-decimate to avoid introducing fractional
    label values.

    Args:
        signals: DataFrame where each column is a named PSG channel.
        new_Fs: Target sampling frequency in Hz.
        original_Fs: Original sampling frequency of the raw data in Hz.
        extreme_resp_filtering: If ``True``, use a narrow 0.1-2 Hz band-pass
            for respiratory channels (useful for very noisy CPAP signals).

    Returns:
        A new DataFrame at the target sample rate with all channels filtered
        and stored as float32 to save memory.
    """
    # US powerline frequency -- 60 Hz (change to 50 for European data).
    notch_freq_us = 60.0

    # Band-pass corners per modality.
    bandpass_freq_eeg = [0.1, 20]
    bandpass_freq_breathing = [0.1, 2] if extreme_resp_filtering else [0.0, 10]
    # ECG: high-pass only -- low-pass is intentionally omitted.
    bandpass_freq_ecg = [0.3, None]

    new_df = pd.DataFrame([], columns=signals.columns)

    for sig in signals.columns:
        image = signals[sig].values
        not_zero = not np.all(image == 0)

        # --- Notch filter (powerline removal) ---
        # Applied to all analog channels.  Zero-valued channels are skipped
        # to avoid processing empty/disconnected leads.
        if not_zero and sig in _ALL_ANALOG_CHANNELS:
            # Only apply if the Nyquist frequency is above the notch
            # (otherwise the notch frequency cannot exist in the signal).
            if original_Fs >= 2 * notch_freq_us:
                image = notch_filter(image.astype(float), original_Fs, notch_freq_us, verbose=False)

        # --- Band-pass filtering per modality ---
        if not_zero and sig in _EEG_EMG_CHANNELS:
            image = filter_data(
                image,
                original_Fs,
                bandpass_freq_eeg[0],
                bandpass_freq_eeg[1],
                verbose=False,
            )
        elif not_zero and sig in _RESPIRATORY_CHANNELS:
            image = filter_data(
                image,
                original_Fs,
                bandpass_freq_breathing[0],
                bandpass_freq_breathing[1],
                verbose=False,
            )
        elif not_zero and sig == "ecg":
            image = filter_data(
                image,
                original_Fs,
                bandpass_freq_ecg[0],
                bandpass_freq_ecg[1],
                verbose=False,
            )

        # --- Resampling ---
        if new_Fs != original_Fs:
            if not_zero and sig in _ALL_ANALOG_CHANNELS:
                # Polyphase resampling preserves spectral fidelity for analog
                # channels (anti-alias filter built in).
                image = resample_poly(image, new_Fs, original_Fs)
            else:
                # Repeat-and-decimate for label/categorical channels -- avoids
                # interpolation artefacts (e.g. fractional sleep stages).
                image = np.repeat(image, new_Fs)[::original_Fs]

        new_df[sig] = image.astype(np.float32)

    return new_df


def clip_normalize_signals(
    signals: pd.DataFrame,
    sample_rate: int,
    br_trace: list[str],
    split_loc: int = None,
    min_max_times_global_iqr: int = 20,
) -> pd.DataFrame:
    """Clip outliers and normalise each channel to a comparable scale.

    Two normalisation strategies are used depending on modality:

    * **EEG / EMG / ECG** -- Symmetric IQR-based clipping followed by
      ``RobustScaler`` (median-centred, IQR-scaled).  The clipping threshold
      is set at ``min_max_times_global_iqr x IQR`` (default 20x) to
      suppress high-amplitude artefacts (movements, electrode pops) while
      retaining physiological extremes like K-complexes.

    * **Respiratory channels** -- Percentile clipping (5th-95th) followed by
      mean/std normalisation.  An additional asymmetric threshold (based on
      the 20th and 80th quantiles) suppresses residual artefacts.  The
      ``split_loc`` mechanism allows for a mid-study CPAP titration where
      different breathing traces are active before and after the split.

    * **SpO2** -- Clamped to the physiologically plausible range [60, 100] %
      and rounded to integer (pulse oximeters report whole percentages).

    Channels in ``skip_cols`` (annotations, metadata) and all-zero channels
    (disconnected leads) are passed through unchanged.

    Args:
        signals: DataFrame of preprocessed PSG channels.
        sample_rate: Sampling frequency in Hz.
        br_trace: Two-element list of the primary breathing-trace channel
            names ``[pre_split_channel, post_split_channel]``.  Used when
            ``split_loc`` is not ``None`` to decide which trace is active
            in each half of the recording.
        split_loc: Sample index where the recording transitions from one
            breathing setup to another (e.g. diagnostic -> CPAP).  ``None``
            means no split.
        min_max_times_global_iqr: Multiplier for the IQR clipping threshold
            on EEG/EMG/ECG channels.

    Returns:
        The input DataFrame with normalised channel values in-place.
    """
    # Columns that are annotations or metadata -- never normalise these.
    skip_cols = [
        "stage",
        "arousal",
        "resp",
        "spo2_desat",
        "cpap_pressure",
        "cpap_on",
        "spo2_artifact",
        "effort_artifact",
        "start",
        "end",
    ]

    for chan in signals.columns:
        if any(skip in chan for skip in skip_cols) or np.all(signals[chan] == 0):
            continue

        signal = signals[chan].values

        # --- SpO2: simple range clamping ---
        if chan == "spo2":
            # Values below 60 % are almost certainly artefact; values above
            # 100 % are impossible.
            signals[chan] = np.clip(signal.astype(float).round(), 60, 100)
            continue

        # --- EEG / EMG / ECG: IQR clipping + RobustScaler ---
        if chan in _EEG_EMG_CHANNELS + ["ecg"]:
            iqr = np.subtract(*np.percentile(signal, [75, 25]))
            # Wide clipping window (20x IQR) keeps K-complexes and spindles
            # while removing gross artefacts.
            threshold = iqr * min_max_times_global_iqr
            signal_clipped = np.clip(signal, -threshold, threshold)

            # RobustScaler centres on median and scales by IQR -- robust to
            # the residual outliers that survive clipping.
            transformer = RobustScaler().fit(np.atleast_2d(signal_clipped).T)
            signal_normalized = np.squeeze(transformer.transform(np.atleast_2d(signal_clipped).T))

        # --- Respiratory channels: percentile clipping + z-score ---
        elif chan in _RESPIRATORY_CHANNELS:
            region = np.arange(len(signal))

            # If the study has a diagnostic/CPAP split, only the active
            # breathing trace for each half of the night is processed.
            if split_loc is not None:
                replacement_signal = np.empty(len(signals)) * np.nan
                if chan in [br_trace[0], "airflow"]:
                    region = region[:split_loc]
                elif chan == br_trace[1]:
                    region = region[split_loc:]
                replacement_signal[region] = signal[region]
                signal = replacement_signal

            if np.all(signal[region] == 0):
                continue

            # Clip to 5th-95th percentile to remove extreme breaths /
            # movement artefact while keeping the bulk of the distribution.
            signal_clipped = np.clip(
                signal,
                np.nanpercentile(signal[region], 5),
                np.nanpercentile(signal[region], 95),
            )

            if np.all(signal_clipped[region] == 0):
                continue

            # Standard z-score using clipped statistics.
            signal_normalized = (signal - np.nanmean(signal_clipped)) / np.nanstd(signal_clipped)

            # Secondary artefact suppression: a tighter, signal-adaptive
            # clipping threshold derived from the inter-quantile range of
            # the normalised signal.
            #   - Effort channels (abd, chest): moderate threshold (10x)
            #     and 1 % quantile for clipping -- they are intrinsically noisier.
            #   - Flow channels: tighter quantile (0.1 %) but wider factor (20x)
            #     to preserve flow-limitation detail.
            _clp = 0.01 if chan in ["abd", "chest"] else 0.001
            factor = 10 if chan in ["abd", "chest"] else 20
            quan = 0.2
            thresh = factor * np.mean(
                [
                    np.abs(np.nanquantile(signal_normalized[region], quan)),
                    np.abs(np.nanquantile(signal_normalized[region], 1 - quan)),
                ]
            )
            signal_normalized = np.clip(signal_normalized, -thresh, thresh)

        signals[chan] = signal_normalized

    return signals
