"""
Mackey-Glass state-space ventilation model and RMSE objective function.

The Mackey-Glass model is a discrete-time dynamical system that simulates
the respiratory control loop:

  **State equation:**
    x(k) = x(k-1) * (1 - V_o(k - tau)) + L + noise

  **Output equation (Hill function):**
    V_o(k) = V_max * x(k)^gamma / (1 + x(k)^gamma) * u(k) + Arousal(k) + noise

Where:
  - ``x`` is the chemosensory drive (e.g. CO2 level)
  - ``V_o`` is the estimated ventilation
  - ``tau`` is the circulation delay (chemo-receptor feedback lag)
  - ``gamma`` controls the steepness of the Hill nonlinearity
  - ``u`` is the inspiratory drive (1 = normal, < 1 during obstruction)
  - ``L`` is the constant CO2 production rate (typically 0.05)
  - ``Arousal`` is an additive arousal perturbation

This module ports:
  - ``fcn_state_space_loop.m``  ->  ``state_space_loop``
  - ``fcn_apply_mg.m``          ->  ``compute_rmse``
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ── Numba-accelerated inner loop ──────────────────────────────────────
# The state-space recurrence is inherently sequential (each sample
# depends on the previous), so it cannot be vectorized with NumPy.
# Numba's @njit compiles this to native machine code, giving a ~30-50x
# speedup over the pure-Python for-loop.
#
# The noise vectors (nss2, nss3) are pre-generated outside the loop
# and passed in as arrays — this avoids Numba having to deal with
# NumPy's random Generator API.


def _state_space_loop_python(
    K: int,
    L: float,
    V_max: float,
    gamma: float,
    tau: int,
    u: np.ndarray,
    Arousal: np.ndarray,
    nss2: np.ndarray,
    nss3: np.ndarray,
) -> np.ndarray:
    """Pure-Python fallback for the state-space loop."""
    x = np.zeros(K, dtype=np.float64)
    V_o_es = np.zeros(K, dtype=np.float64)
    eps = 1e-2

    for k in range(K):
        if k < tau:
            if k == 0:
                x[k] = 0.0
            else:
                x[k] = L + nss2[k]
        else:
            x[k] = x[k - 1] * (1.0 - V_o_es[k - tau]) + L + nss2[k]
            if x[k] < 0:
                x[k] = eps

        xg = x[k] ** gamma
        V_o_es[k] = (V_max * xg / (1.0 + xg)) * u[k] + Arousal[k] + nss3[k]

        if V_o_es[k] < 0:
            V_o_es[k] = 0.0

    return V_o_es


if _HAS_NUMBA:
    _state_space_loop_fast = njit(cache=True)(_state_space_loop_python)
else:
    _state_space_loop_fast = _state_space_loop_python


def state_space_loop(
    K: int,
    L: float,
    V_max: float,
    gamma: float,
    tau: int,
    s: float,
    u: np.ndarray,
    Arousal: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run the Mackey-Glass state-space forward simulation.

    Direct port of MATLAB ``fcn_state_space_loop``.  Uses a sequential
    for-loop because each sample depends on the previous one (the
    recurrence x(k) depends on x(k-1) and V_o(k-tau)).

    When Numba is installed, the inner loop is JIT-compiled to native
    code (~30-50x faster). Falls back to pure Python otherwise.

    Args:
        K: Number of samples.
        L: CO2 production rate / loop gain parameter (typically 0.05).
        V_max: Maximum ventilation scaling (typically 1.0).
        gamma: Hill function nonlinearity exponent.
        tau: Feedback delay in samples (integer).
        s: Noise scale (typically 1e-8, making noise negligible).
        u: Drive signal array (length K). Values in [0, 1].
        Arousal: Arousal contribution array (length K).
        rng: NumPy random generator. If None, creates one with seed 0
             to match MATLAB's ``rng('default')``.

    Returns:
        V_o_es: Estimated ventilation signal (length K).
    """
    if rng is None:
        rng = np.random.default_rng(seed=0)

    tau = int(tau)

    # Pre-generate noise vectors outside the loop (so Numba doesn't
    # need to handle the NumPy Generator API).
    nss2 = s * rng.standard_normal(K) * np.sqrt(s)
    nss3 = s * rng.standard_normal(K) * np.sqrt(s)

    return _state_space_loop_fast(K, L, V_max, gamma, tau, u, Arousal, nss2, nss3)


def compute_rmse(
    V_o: np.ndarray,
    V_max: float,
    gamma: float,
    tau: int,
    s: float,
    D: np.ndarray,
    Arousal: np.ndarray,
    u: np.ndarray,
) -> float:
    """Run the Mackey-Glass model and compute scaled RMSE against observed ventilation.

    Direct port of MATLAB ``fcn_apply_mg``.  The steps are:
    1. Run the state-space loop with candidate parameters.
    2. Compute a scaling factor from the ratio V_o / V_o_est (filtering
       out NaN, Inf, and outlier ratios > 5).
    3. Scale the estimate at non-arousal locations.
    4. Compute RMSE only at locations where u == 1 (normal breathing,
       not during obstructed breaths).

    Note: L is hardcoded to 0.05 inside this function, matching the
    MATLAB implementation.

    Args:
        V_o: Observed ventilation (length K).
        V_max: Maximum ventilation parameter.
        gamma: Hill exponent candidate.
        tau: Delay candidate (samples).
        s: Noise scale.
        D: Alpha-adjusted drive signal.
        Arousal: Arousal contribution.
        u: Original (un-adjusted) drive signal.

    Returns:
        RMSE: Root-mean-square error (scalar).
    """
    K = len(V_o)
    # L is hardcoded to 0.05 in the MATLAB fcn_apply_mg
    V_o_es = state_space_loop(K, 0.05, V_max, gamma, tau, s, D, Arousal)

    # Compute scaling factor: ratio of observed to estimated, excluding
    # NaN/Inf and extreme outliers (ratio > 5)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = V_o / V_o_es
    valid = np.isfinite(ratio) & (ratio <= 5.0)
    if np.sum(valid) == 0:
        return np.inf
    Scale = np.mean(ratio[valid])

    # Scale estimate at non-arousal locations only
    V_o_es_scaled = V_o_es.copy()
    non_arousal = Arousal == 0
    V_o_es_scaled[non_arousal] = V_o_es[non_arousal] * Scale

    # RMSE computed only where u == 1 (normal breathing)
    mask = u == 1.0
    if np.sum(mask) == 0:
        return np.inf
    return float(np.linalg.norm(V_o_es_scaled[mask] - V_o[mask]))
