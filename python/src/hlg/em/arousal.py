"""
Arousal event estimation for the Mackey-Glass EM algorithm.

Arousal events are modelled as rectangular pulses of width ``w`` centred
at each detected arousal location.  The height of each pulse is determined
by the maximum of a reference signal (typically observed ventilation or
model error) within the pulse window.

This module ports the MATLAB functions:
  - ``fcn_get_unit_function.m``  ->  ``heaviside``
  - ``fcn_arousal_event.m``      ->  ``estimate_arousals``

The Heaviside step function is the building block: a rectangular pulse
from ``t0 - w/2`` to ``t0 + w/2`` is constructed as:
    ``u(t, t0 - w/2) - u(t - w/2, t0)``
where ``u(t, t0) = 1 if t >= t0, else 0``.
"""

from __future__ import annotations

import numpy as np


def heaviside(t: np.ndarray, t0: float) -> np.ndarray:
    """Unit step function: returns 1 where t >= t0, 0 elsewhere.

    Direct port of MATLAB ``fcn_get_unit_function(t, t0)``.

    Args:
        t: Time/sample index array.
        t0: Step onset position.

    Returns:
        Boolean-like float array (0.0 or 1.0).
    """
    return (t >= t0).astype(np.float64)


def estimate_arousals(
    dit: np.ndarray,
    K: int,
    Vref: np.ndarray,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate arousal event magnitudes and construct the arousal signal.

    For each arousal location (non-zero entry in ``dit``), constructs a
    rectangular pulse of width ``w`` samples and sets its height to the
    maximum of ``Vref`` within the pulse window.  The individual pulses
    are summed into a single cumulative arousal signal.

    Direct port of MATLAB ``fcn_arousal_event(dit, K, Vref, w)``.

    Args:
        dit: Binary array of arousal location indicators (length K).
             Non-zero entries mark arousal positions.
        K: Total number of samples.
        Vref: Reference signal for determining pulse heights (length K).
              Typically the observed ventilation or the model prediction
              error at arousal locations.
        w: Arousal pulse width in samples (e.g. 5 seconds * 10 Hz = 50).

    Returns:
        A tuple ``(h, Arousal)`` where:
          - ``h`` is a 1-D array of per-arousal magnitudes (length = number
            of arousals).
          - ``Arousal`` is the cumulative arousal signal (length K),
            formed by summing all rectangular pulses.
    """
    # Find indices where arousals are located (MATLAB: find(dit))
    # MATLAB uses 1-based indexing; Python uses 0-based.
    t_ar = np.where(dit != 0)[0]

    # Time index vector (0-based in Python, 1-based in MATLAB).
    # We use 1-based to match MATLAB's arithmetic exactly.
    t = np.arange(1, K + 1, dtype=np.float64)

    h = np.zeros(len(t_ar), dtype=np.float64)
    Arousal = np.zeros(K, dtype=np.float64)

    for idx in range(len(t_ar)):
        # MATLAB t_ar is 1-based; our t_ar is 0-based, so add 1 for
        # the pulse centre to match MATLAB's arithmetic.
        centre = t_ar[idx] + 1  # convert to 1-based

        # Rectangular pulse: step-up at (centre - w/2), step-down at centre
        square = heaviside(t, centre - w / 2) - heaviside(t - w / 2, centre)

        # Pulse height = max of Vref within the pulse window
        h[idx] = np.max(square * Vref)

        # Accumulate into the total arousal signal
        Arousal += h[idx] * square

    return h, Arousal
