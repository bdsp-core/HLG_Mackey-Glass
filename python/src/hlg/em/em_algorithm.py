"""
Expectation-Maximization algorithm for Mackey-Glass parameter estimation.

The EM algorithm estimates three ventilatory control parameters from an
8-minute segment of respiratory data:

  - **alpha**: arousal mixing factor (0 to 1 in steps of 0.25)
  - **gamma**: Hill function nonlinearity exponent (0.1 to 2.0, step 0.01)
  - **tau**: chemosensory feedback delay in samples (50 to 500, step 10)

The algorithm alternates between:
  1. **E-step**: estimating arousal event magnitudes from the observed
     ventilation signal.
  2. **M-step**: grid-searching over (alpha, gamma, tau) to minimise the
     RMSE between the Mackey-Glass model output and observed ventilation.

This is repeated for ``Iter`` iterations (typically 5).

This module ports:
  - ``fcn_em_algorithm.m``            ->  ``run_em``
  - ``fcn_em_algorithm_real_data.m``  ->  ``run_em_on_segment``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hlg.em.arousal import estimate_arousals, heaviside
from hlg.em.mackey_glass import compute_rmse, state_space_loop


def run_em(
    K: int,
    L: float,
    V_max: float,
    gamma: float,
    tau: int,
    s: float,
    V_o: np.ndarray,
    u: np.ndarray,
    dit: np.ndarray,
    Iter: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Core EM parameter estimation via grid search.

    Direct port of MATLAB ``fcn_em_algorithm``.

    Args:
        K: Number of samples in the segment.
        L: CO2 production rate (typically 0.05).
        V_max: Maximum ventilation scaling (typically 1.0).
        gamma: Initial gamma estimate.
        tau: Initial tau estimate (samples).
        s: Noise scale (typically 1e-8).
        V_o: Observed ventilation signal (length K).
        u: Inspiratory drive signal (length K).
        dit: Arousal location indicators (length K, binary).
        Iter: Number of EM iterations (typically 5).
        w: Arousal pulse width in samples.

    Returns:
        A tuple ``(upAlpha, upgamma, uptau, h)`` where each ``up*``
        array has length ``Iter`` (one value per iteration) and ``h``
        is the refined arousal magnitude array.
    """
    np.random.seed(0)  # Match MATLAB rng('default')

    upAlpha = np.zeros(Iter)
    upgamma = np.zeros(Iter)
    uptau = np.zeros(Iter)

    # ── E-step: estimate arousal events ──────────────────────────────
    h, Arousal = estimate_arousals(dit, K, V_o, w)
    V_o_es = state_space_loop(K, L, V_max, gamma, tau, s, u, Arousal)

    # Refine arousals by subtracting model prediction error at arousal
    # locations.  This corrects for the initial over/under-estimation
    # of arousal heights.
    Arousal_err = np.zeros(K)
    arousal_mask = Arousal != 0
    Arousal_err[arousal_mask] = V_o_es[arousal_mask] - V_o[arousal_mask]
    h_diff, Arousal_dif = estimate_arousals(dit, K, Arousal_err, w)
    Arousal = Arousal - Arousal_dif

    # Update arousal heights, clamping negatives to zero
    temp = h - h_diff
    temp[temp < 0] = 0.0
    h = temp

    # ── Parameter grid definitions ───────────────────────────────────
    Fs = 10
    tau_range = np.arange(5 * Fs, 50 * Fs + Fs, Fs, dtype=int)  # 50 to 500, step 10
    gamma_range = np.arange(0.1, 2.0 + 0.01, 0.01)  # 0.1 to 2.0, step 0.01
    alpha_range = np.arange(0, 1.0 + 0.25, 0.25)  # 0, 0.25, 0.5, 0.75, 1.0

    n_alpha = len(alpha_range)
    n_gamma = len(gamma_range)
    n_tau = len(tau_range)

    # ── M-step: iterative grid search ────────────────────────────────
    for iteration in range(Iter):
        RMSE = np.zeros(n_alpha)
        tau_temp = np.zeros(n_alpha, dtype=int)
        gamma_temp = np.zeros(n_alpha)

        for a_idx, alpha_val in enumerate(alpha_range):
            # Adjust drive signal based on alpha
            D = u + alpha_val * (1.0 - u)

            # Grid search over gamma (with current tau held fixed)
            gamma_RMSE = np.zeros(n_gamma)
            for g_idx in range(n_gamma):
                gamma_RMSE[g_idx] = compute_rmse(V_o, V_max, gamma_range[g_idx], tau, s, D, Arousal, u)
            gamma_temp[a_idx] = gamma_range[np.argmin(gamma_RMSE)]

            # Grid search over tau (with current gamma held fixed)
            tau_RMSE = np.zeros(n_tau)
            for t_idx in range(n_tau):
                tau_RMSE[t_idx] = compute_rmse(V_o, V_max, gamma, tau_range[t_idx], s, D, Arousal, u)
            tau_temp[a_idx] = tau_range[np.argmin(tau_RMSE)]

            # Evaluate RMSE at the best (gamma, tau) for this alpha
            RMSE[a_idx] = compute_rmse(V_o, V_max, gamma_temp[a_idx], tau_temp[a_idx], s, D, Arousal, u)

        # Select the alpha with the lowest RMSE
        best_idx = np.argmin(RMSE)
        alpha = alpha_range[best_idx]
        gamma = gamma_temp[best_idx]
        tau = int(tau_temp[best_idx])

        upAlpha[iteration] = alpha
        uptau[iteration] = tau
        upgamma[iteration] = gamma

    return upAlpha, upgamma, uptau, h


def run_em_on_segment(
    df_segment: pd.DataFrame,
    w: int,
    L: float,
    gamma_init: float,
    tau_init: int,
    version: str = "non-smooth",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Run the EM algorithm on a single 8-minute sleep segment.

    Direct port of MATLAB ``fcn_em_algorithm_real_data``.

    Extracts the relevant columns from the segment DataFrame, runs the
    EM parameter estimation, reconstructs the arousal signal from the
    estimated magnitudes, and produces the final model output.

    Args:
        df_segment: DataFrame for one 8-minute segment. Must contain
            columns: ``Ventilation_ABD``, ``d_i_ABD`` (or ``d_i_ABD_smooth``),
            ``arousal_locs``.
        w: Arousal pulse width in samples.
        L: CO2 production rate (typically 0.05).
        gamma_init: Initial gamma for the grid search.
        tau_init: Initial tau in samples (e.g. 15 * Fs = 150).
        version: ``"smooth"`` uses ``d_i_ABD_smooth``; ``"non-smooth"``
            uses ``d_i_ABD``.

    Returns:
        A tuple ``(upAlpha, upgamma, uptau, V_o_est, h, u_min)`` where:
          - ``upAlpha/upgamma/uptau``: per-iteration parameter estimates
          - ``V_o_est``: final estimated ventilation signal
          - ``h``: arousal magnitudes
          - ``u_min``: minimum drive value (used for LG computation)
    """
    np.random.seed(0)  # Match MATLAB rng('default')

    V_max = 1.0
    K = len(df_segment)
    s = 1e-8

    V_o = df_segment["Ventilation_ABD"].values.astype(np.float64)
    if version == "smooth":
        u = df_segment["d_i_ABD_smooth"].values.astype(np.float64)
    else:
        u = df_segment["d_i_ABD"].values.astype(np.float64)

    u_min = float(np.min(u))
    dit = df_segment["arousal_locs"].values.astype(np.float64)

    Iter = 5
    upAlpha, upgamma, uptau, h = run_em(K, L, V_max, gamma_init, tau_init, s, V_o, u, dit, Iter, w)

    # Reconstruct arousal signal from estimated magnitudes
    t_ar = np.where(dit != 0)[0]
    t = np.arange(1, K + 1, dtype=np.float64)
    Arousal_est = np.zeros(K, dtype=np.float64)
    for idx in range(len(t_ar)):
        centre = t_ar[idx] + 1  # 1-based to match MATLAB
        square = heaviside(t, centre - w / 2) - heaviside(t - w / 2, centre)
        Arousal_est += h[idx] * square

    # Final forward model with best-fit parameters
    Alpha = upAlpha[-1]
    gamma = upgamma[-1]
    tau = int(uptau[-1])
    D = u + Alpha * (1.0 - u)
    V_o_est = state_space_loop(K, L, V_max, gamma, tau, s, D, Arousal_est)

    return upAlpha, upgamma, uptau, V_o_est, h, u_min
