"""
Ventilation envelope computation and breath-by-breath ventilation trace.

This module computes the **ventilation signal** -- a smooth, breath-by-breath
estimate of tidal ventilation amplitude -- from raw respiratory effort or flow
channels.  It is central to the loop-gain estimation pipeline because the
ventilation trace is what gets decomposed into the drive and response
components of the ventilatory control system.

The processing chain is:

1. **Envelope extraction** -- positive and negative peak envelopes are
   interpolated from the raw breathing waveform, giving a smooth upper
   and lower boundary of breath amplitude.
2. **Baseline estimation** -- a slowly varying baseline captures postural
   shifts and sensor drift so that ventilation changes are measured relative
   to the patient's own "eupneic" (normal breathing) level.
3. **Ventilation trace construction** -- the peak-to-trough amplitude
   (positive envelope minus negative envelope) after detrending and
   normalisation yields the instantaneous ventilation signal.
4. **Eupnea and fractional ventilation (d_i)** -- a long-window median of
   ventilation defines the eupneic level; the ratio ``ventilation / eupnea``
   gives the fractional ventilation ``d_i`` used by the loop-gain model.
5. **Arousal-location detection** -- abrupt ventilatory overshoots after
   periods of reduced ventilation are flagged as candidate arousal sites.

This module merges the original ``Ventilation_envelope.py`` and
``Create_Ventilation.py`` into one coherent file.

Dependencies: numpy, pandas, scipy, matplotlib (optional plotting).
Local dependency: ``hlg.core.events`` for ``find_events`` and
``window_correction``.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import groupby

from scipy.signal import find_peaks, savgol_filter, detrend

from hlg.core.events import find_events, window_correction


# ===================================================================
# Envelope computation (from Ventilation_envelope.py)
# ===================================================================


def compute_ventilation_envelopes(
    data: pd.DataFrame,
    Fs: int,
    channels: list[str] | None = None,
) -> pd.DataFrame:
    """Compute positive/negative envelopes and baselines for ventilation.

    Wraps ``compute_envelope`` and writes the results back into the input
    DataFrame under standardised column names.  If multiple channels are
    provided they are summed first (e.g. combining thoracic and abdominal
    effort into a single surrogate).

    Args:
        data: DataFrame containing at least the columns listed in
            ``channels``.
        Fs: Sampling frequency in Hz.
        channels: Column name(s) to use as the source ventilation signal.
            If more than one is given, their sample-wise sum is used.

    Returns:
        The input DataFrame with five new columns appended:
        ``Ventilation_pos_envelope``, ``Ventilation_neg_envelope``,
        ``Ventilation_default_baseline``, ``Ventilation_baseline``,
        and ``baseline2``.
    """
    if channels is None:
        channels = ["Ventilation_combined"]
    if len(channels) == 1:
        new_df = compute_envelope(data[channels[0]].values, Fs)
    else:
        new_df = compute_envelope(data[channels].sum(axis=1).values, Fs)

    data["Ventilation_pos_envelope"] = new_df["pos_envelope"].values
    data["Ventilation_neg_envelope"] = new_df["neg_envelope"].values
    data["Ventilation_default_baseline"] = new_df["baseline"].values
    data["Ventilation_baseline"] = new_df["correction_baseline"].values
    data["baseline2"] = new_df["baseline2"].values
    return data


def compute_envelope(
    signal: np.ndarray,
    Fs: int,
    base_win: int = 30,
    env_smooth: int = 5,
) -> pd.DataFrame:
    """Extract peak envelopes and baselines from a breathing waveform.

    Peak detection parameters are tuned for adult breathing at rest:
    * ``distance = 1.5 s`` -- enforces a minimum breathing period of 1.5 s
      (i.e. max ~40 breaths/min), filtering out cardiac-frequency artefact.
    * ``width = 0.4 s`` -- rejects narrow spikes that are unlikely to be
      genuine breath peaks.

    The envelopes are then median-smoothed over ``env_smooth`` seconds to
    remove breath-to-breath variability, yielding a clean amplitude trend.

    Args:
        signal: 1-D breathing waveform (e.g. nasal flow or thoracic effort).
        Fs: Sampling frequency in Hz.
        base_win: Baseline averaging window in seconds (default 30 s covers
            roughly 5-8 breaths, enough to track slow trends without
            following individual apneas).
        env_smooth: Envelope smoothing window in seconds.

    Returns:
        A DataFrame with columns ``x`` (input signal), ``pos_envelope``,
        ``neg_envelope``, ``baseline``, ``baseline2``, and
        ``correction_baseline``.
    """
    new_df = pd.DataFrame()
    new_df["x"] = np.squeeze(signal)

    # Detect inspiratory peaks (positive) and expiratory troughs (negative).
    # The distance and width constraints are critical for rejecting cardiac
    # artefact that leaks into thoracic effort channels.
    pos_peaks, _ = find_peaks(new_df["x"], distance=int(Fs * 1.5), width=int(0.4 * Fs), rel_height=1)
    neg_peaks, _ = find_peaks(-new_df["x"], distance=int(Fs * 1.5), width=int(0.4 * Fs), rel_height=1)

    # Cubic interpolation between detected peaks produces a smooth envelope.
    # ``limit_area='inside'`` avoids extrapolation beyond the first/last peak.
    new_df["pos_envelope"] = (
        new_df["x"].iloc[pos_peaks].reindex_like(new_df).interpolate(method="cubic", limit_area="inside")
    )
    new_df["neg_envelope"] = (
        new_df["x"].iloc[neg_peaks].reindex_like(new_df).interpolate(method="cubic", limit_area="inside")
    )

    # Rolling median removes breath-to-breath jitter while preserving the
    # slow amplitude trend (hypopnea crescendo/decrescendo patterns).
    new_df["pos_envelope"] = new_df["pos_envelope"].rolling(env_smooth * Fs, center=True).median()
    new_df["neg_envelope"] = new_df["neg_envelope"].rolling(env_smooth * Fs, center=True).median()

    # Sanity check: the positive envelope must always be above the negative.
    # Violations indicate regions where peak detection failed (e.g. flat
    # signal during a central apnea); zero them out.
    check_invalids = new_df["pos_envelope"] < new_df["neg_envelope"]
    new_df.loc[check_invalids, ["pos_envelope", "neg_envelope"]] = 0

    new_df["baseline"], new_df["baseline2"], new_df["correction_baseline"] = compute_baseline(new_df, Fs, base_win)
    return new_df


def compute_baseline(
    new_df: pd.DataFrame,
    Fs: int,
    base_win: int,
    correction_ratio: int = 2,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute a robust, slowly varying ventilatory baseline.

    Three baselines are produced:

    * **baseline1** -- rolling median then mean of the raw signal. Tracks the
      DC offset (sensor drift, posture changes) without being pulled by
      individual breaths.
    * **baseline2** -- rolling mean of the midpoint between the positive and
      negative envelopes.  Captures the long-term average breath amplitude
      midline.
    * **correction_baseline** -- a weighted blend
      ``(2 * baseline1 + baseline2) / 3`` that combines the stability of the
      signal-level estimate with the amplitude-aware envelope midpoint.  The
      2 : 1 weighting favours the median-based estimate because it is more
      resistant to artefact.

    All windows are ``base_win`` seconds long and centred, so the baseline
    does not lag behind ventilatory events.

    Args:
        new_df: DataFrame with ``x``, ``pos_envelope``, and ``neg_envelope``
            columns.
        Fs: Sampling frequency in Hz.
        base_win: Window length in seconds for all rolling operations.
        correction_ratio: Weighting of baseline1 vs baseline2 in the
            correction blend (default 2 -> baseline1 has twice the weight).

    Returns:
        A tuple ``(baseline1, baseline2, correction_baseline)`` as pandas
        Series.
    """
    pos = new_df["pos_envelope"].rolling(base_win * Fs, center=True).mean()
    neg = new_df["neg_envelope"].rolling(base_win * Fs, center=True).mean()
    base = (pos + neg) / 2

    # baseline1: double-smoothed signal median -- very stable.
    base1 = new_df["x"].rolling(base_win * Fs, center=True).median().rolling(base_win * Fs, center=True).mean()
    # baseline2: smoothed envelope midpoint -- amplitude-aware.
    base2 = base.rolling(base_win * Fs, center=True).mean()

    # Weighted blend: the correction_ratio (default 2) gives the signal-level
    # median twice the weight of the envelope midpoint.
    base_corr = (correction_ratio * base1 + base2) / (1 + correction_ratio)
    base_corr = base_corr.rolling(base_win * Fs, center=True).mean()

    return base1, base2, base_corr


def compute_smooth_envelope(
    data: pd.DataFrame,
    region: list[int],
) -> pd.DataFrame:
    """Apply Savitzky-Golay smoothing to ventilation envelopes.

    A first-order Savitzky-Golay filter with a 51-sample window is used to
    produce a very smooth version of the envelopes for downstream
    calculations that need a monotone trend (e.g. the self-similarity
    segmentation).

    Args:
        data: DataFrame containing ``Ventilation_pos_envelope`` and
            ``Ventilation_neg_envelope`` columns.
        region: Index array (sample indices) over which to apply smoothing.

    Returns:
        The input DataFrame with ``Smooth_pos_envelope`` and
        ``Smooth_neg_envelope`` columns written for the specified region.
    """
    for env_tag in ["Smooth_pos_envelope", "Smooth_neg_envelope"]:
        original_env = env_tag.replace("Smooth", "Ventilation")
        data.loc[region, env_tag] = savgol_filter(data.loc[region, original_env], 51, 1)
    return data


# ===================================================================
# Ventilation trace creation (from Create_Ventilation.py)
# ===================================================================


def create_ventilation_trace(
    data: pd.DataFrame,
    Fs: int,
    plot: bool = False,
) -> pd.DataFrame:
    """Build the breath-by-breath ventilation trace and fractional ventilation.

    For each available breathing channel (``breathing_trace``, ``ABD``) this
    function:

    1. Computes the channel's peak-to-trough amplitude via envelope extraction.
    2. Detrends and normalises the channel by its average envelope amplitude
       (``Q``), making the ventilation trace unit-free.
    3. Constructs a running envelope of the normalised signal using rolling
       quantiles (95th for positive, 5th for negative) smoothed with a
       Savitzky-Golay filter.
    4. Derives the ventilation amplitude as ``pos_env - neg_env``, down-
       sampled by 10x and then repeated back up to save computation.
    5. Computes the **eupneic ventilation** as a 30-minute centred median --
       representing the patient's "normal" breathing level across long
       stretches of stable sleep.
    6. Computes the **fractional ventilation** ``d_i = ventilation / eupnea``,
       capped at 1.0 (i.e. above-eupnea breaths are treated as 100 %).
    7. Identifies periods of **below-eupnea ventilation** and marks candidate
       **arousal locations** as the ventilatory peak following each such
       period.

    Args:
        data: DataFrame with preprocessed respiratory and staging channels.
            Must contain ``Stage`` and at least one of ``breathing_trace``
            or ``ABD``.
        Fs: Sampling frequency in Hz (must be an integer).
        plot: If ``True``, display a diagnostic matplotlib figure showing
            the ventilation trace, eupnea, and detected events.

    Returns:
        The input DataFrame augmented with per-channel columns:
        ``Ventilation_<chan>``, ``Eupnea_<chan>``, ``d_i_<chan>``,
        ``d_i_<chan>_smooth``, and ``arousal_locs``.

    Raises:
        AssertionError: If ``Fs`` is not an integer value.
    """
    assert Fs == int(Fs), 'Provided "Fs" is a double'
    Fs = int(Fs)

    for col in ["breathing_trace", "ABD"]:
        if col not in data.columns:
            continue

        # --- Step 1: Envelope-based amplitude normalisation ---
        # Compute the average peak-to-trough range (Q) so we can express
        # ventilation in units of "fraction of normal breath amplitude".
        df = compute_ventilation_envelopes(data.copy(), Fs, channels=[col])
        Q = df.Ventilation_pos_envelope.mean() - df.Ventilation_neg_envelope.mean()
        sig = detrend(data[col].fillna(0)) / Q

        # --- Step 2: Rolling quantile envelopes of the normalised signal ---
        # The 95th/5th percentile over a 20-sample window (~2 s at 10 Hz)
        # tracks the instantaneous breath amplitude without being sensitive
        # to the exact peak location.
        df["pos_env"] = np.array(sig)
        df["neg_env"] = np.array(sig)
        df["pos_env"] = df["pos_env"].rolling(20, center=True, min_periods=1).quantile(0.95)
        df["neg_env"] = df["neg_env"].rolling(20, center=True, min_periods=1).quantile(0.05)

        # Savitzky-Golay smoothing (51-point, 1st order) produces a clean
        # monotone envelope suitable for amplitude measurement.
        df["pos_env"] = savgol_filter(df["pos_env"], 51, 1)
        df["neg_env"] = savgol_filter(df["neg_env"], 51, 1)

        # --- Step 3: Ventilation amplitude ---
        # Down-sample by 10x before differencing to reduce noise, then
        # repeat back up to the original rate.
        Ventilation = df["pos_env"].values[::10] - df["neg_env"].values[::10]
        # Negative ventilation is physically impossible -- clamp to zero.
        Ventilation[Ventilation < 0] = 0
        df[f"Ventilation_{col}"] = np.repeat(Ventilation, 10)[: len(df)]

        # --- Step 4: Eupneic ventilation (30-minute centred median) ---
        # A 30-minute window captures the patient's stable baseline
        # breathing across multiple sleep cycles while smoothing over
        # individual apnea/hypopnea clusters.
        df[f"average_Ventilation_{col}"] = (
            df[f"Ventilation_{col}"].rolling(Fs * 60 * 30, center=True, min_periods=1).median()
        )

        # --- Step 5: Fractional ventilation d_i ---
        # d_i = current ventilation / eupnea.  Values above 1 are capped
        # because above-eupnea breaths carry no additional clinical
        # information for loop-gain estimation.
        df[f"d_i_{col}"] = df[f"Ventilation_{col}"] / df[f"average_Ventilation_{col}"]
        data[f"Ventilation_{col}"] = df[f"Ventilation_{col}"].values
        data[f"Eupnea_{col}"] = df[f"average_Ventilation_{col}"].values
        data[f"d_i_{col}"] = df[f"d_i_{col}"].values
        data.loc[np.where(data[f"d_i_{col}"] > 1)[0], f"d_i_{col}"] = 1

        # --- Step 6: Below-eupnea detection ---
        # A 6-second centred rolling mean equal to 1 means the ventilation
        # has been continuously below eupnea for 6 s -- long enough to be a
        # genuine respiratory event, not just a single shallow breath.
        below_eupnea = df[f"Ventilation_{col}"] < df[f"average_Ventilation_{col}"]
        df[f"below_Eupnea_{col}"] = below_eupnea.rolling(6 * Fs, center=True).mean() == 1
        df[f"below_Eupnea_{col}"] = window_correction(df[f"below_Eupnea_{col}"], window_size=6 * Fs)

        # --- Step 7: Detect large ventilatory overshoots (arousal candidates) ---
        # A trailing 3-second max-minus-min that exceeds 0.5 (half of eupnea)
        # and is itself the local maximum over 10 s flags an abrupt amplitude
        # surge -- characteristic of arousal-driven hyperventilation.
        trailing_min = df[f"Ventilation_{col}"].rolling(3 * Fs, min_periods=Fs).min()
        trailing_max = df[f"Ventilation_{col}"].rolling(3 * Fs, min_periods=Fs).max()
        biggest_arousal = trailing_max - trailing_min
        biggest_arousal = biggest_arousal.rolling(10 * Fs, min_periods=Fs).max()

        df["possible_large_locs"] = 0
        for loc, _ in find_events(biggest_arousal > 0.5):
            # Only accept if the ventilation at this point is a local
            # minimum (not already rising) and is non-zero.
            if any(df.loc[loc - 2 * Fs : loc, f"Ventilation_{col}"] > df.loc[loc, f"Ventilation_{col}"]):
                continue
            if df.loc[loc, f"Ventilation_{col}"] <= 0:
                continue
            df.loc[loc, "possible_large_locs"] = 1

        # --- Step 8: Assign arousal locations from below-eupnea events ---
        # For each sustained below-eupnea period with mean d_i <= 0.85
        # (at least 15 % ventilatory reduction), find the ventilatory peak
        # in the 2 s before to 10 s after the event end -- this is the
        # post-event overshoot that marks the arousal.
        data[f"d_i_{col}_smooth"] = 1
        data[f"d_i_{col}_smooth"] = data[f"d_i_{col}_smooth"].astype(float)
        data["arousal_locs"] = 0

        for st, end in find_events(df[f"below_Eupnea_{col}"] > 0):
            avg_decrease = np.mean(data.loc[st:end, f"d_i_{col}"])
            if avg_decrease <= 0.85:
                data.loc[st:end, f"d_i_{col}_smooth"] = avg_decrease
                left, right = end - 2 * Fs, end + 10 * Fs
                plus = np.argmax(df.loc[left:right, f"Ventilation_{col}"])
                data.loc[left + plus, "arousal_locs"] = 1
                # Suppress large-loc candidates inside this event to avoid
                # double-counting.
                df.loc[left : left + plus, "possible_large_locs"] = 0

        # --- Step 9: Reconcile large-overshoot candidates with arousal locs ---
        # If a large overshoot coincides with an already-detected arousal
        # (within +/- 3 s), keep only the dominant one; otherwise add it.
        for _st, end in find_events(df["possible_large_locs"] > 0):
            left, right = end - 2 * Fs, end + 10 * Fs
            plus = np.argmax(df.loc[left:right, f"Ventilation_{col}"])
            loc = left + plus
            close_double = np.where(data.loc[loc - 3 * Fs : loc + 3 * Fs, "arousal_locs"] == 1)[0]
            if len(close_double) > 0:
                if close_double[0] != 30:
                    data.loc[loc - 3 * Fs : loc + 3 * Fs, "arousal_locs"] = 0
            df.loc[left:loc, "possible_large_locs"] = 0
            df.loc[loc, "possible_large_locs"] = 1

        data["arousal_locs"] += df["possible_large_locs"]

        # --- Optional diagnostic plot ---
        if plot:
            _plot_ventilation_diagnostics(data, df, col, Fs)

    del df
    return data


def _plot_ventilation_diagnostics(
    data: pd.DataFrame,
    df: pd.DataFrame,
    col: str,
    Fs: int,
) -> None:
    """Render a diagnostic figure of the ventilation trace and events.

    Internal helper used by ``create_ventilation_trace`` when ``plot=True``.
    Displays the raw breathing signal (coloured by sleep/wake), the
    ventilation envelope, eupnea, fractional ventilation, scored events,
    and detected arousal locations.

    Args:
        data: The main analysis DataFrame.
        df: Working DataFrame containing intermediate ventilation columns.
        col: Name of the breathing channel being plotted.
        Fs: Sampling frequency in Hz.
    """
    plt.plot(df[f"{col}"].mask(data.Stage < 5) - 1, "y", alpha=0.4)
    plt.plot(df[f"{col}"].mask(data.Stage == 5) - 1, "k", alpha=0.4)

    vent_plot = df[f"Ventilation_{col}"] + df["pos_env"].mean()
    plt.plot(vent_plot, "b")
    plt.plot([df["pos_env"].mean()] * len(df), "r")
    plt.plot(
        df[f"average_Ventilation_{col}"] + df["pos_env"].mean(),
        "g",
        alpha=0.4,
    )
    plt.plot(
        data[f"d_i_{col}"] * df[f"average_Ventilation_{col}"] + df["pos_env"].mean(),
        "g",
    )
    plt.plot(
        data[f"d_i_{col}_smooth"] * df[f"average_Ventilation_{col}"] + df["pos_env"].mean(),
        "r",
    )

    label_color = [None, "b", "g", "c", "m", "r", None, "g"]
    for li, labels in enumerate([data.Apnea, data.Apnea_algo, df[f"below_Eupnea_{col}"]]):
        loc = 0
        for i, j in groupby(labels):
            len_j = len(list(j))
            if not np.isnan(i) and label_color[int(i)] is not None:
                if li == 0:
                    plt.plot([loc, loc + len_j], [0] * 2, c=label_color[int(i)], lw=2)
                    if int(i) == 7:
                        plt.plot([loc, loc + len_j], [-0.1] * 2, c="m", lw=1)
                if li == 1:
                    plt.plot([loc, loc + len_j], [-0.5] * 2, c=label_color[int(i)], lw=2)
                if li == 2:
                    plt.plot([loc, loc + len_j], [-0.25] * 2, c="k", lw=2)
            loc += len_j

    for loc in np.where(data["arousal_locs"] == 1)[0] + 1:
        if loc >= len(df) - 1:
            loc = len(df) - 1
        y = (data.loc[loc, f"Ventilation_{col}"] + df.loc[loc, "pos_env"].mean()) + 0.2
        plt.text(loc, y, "*", c="k", ha="center")

    for loc in np.where(df["possible_large_locs"] == 1)[0] + 1:
        if loc >= len(df) - 1:
            loc = len(df) - 1
        y = (data.loc[loc, f"Ventilation_{col}"] + df.loc[loc, "pos_env"].mean()) + 0.2
        plt.text(loc, y, "*", c="r", ha="center")

    plt.show()
