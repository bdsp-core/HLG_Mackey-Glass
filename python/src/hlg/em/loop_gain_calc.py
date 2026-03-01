"""
Loop gain computation from Mackey-Glass steady-state analysis.

Loop gain (LG) quantifies the stability of the respiratory control
system.  It is defined as the ratio of the ventilatory response to
the ventilatory deficit during an obstructive event:

    LG = |vss - vr| / |vss - vd|

Where:
  - ``vss`` = steady-state ventilation under normal breathing (drive=1)
  - ``vd``  = steady-state ventilation during partial obstruction (drive=d)
  - ``vr``  = ventilation immediately after obstruction release (the
              chemosensory drive built up during obstruction is now
              unobstructed)

A higher LG means the system overshoots more after obstruction release,
leading to oscillatory/unstable breathing — the hallmark of central and
obstructive sleep apnea driven by high loop gain.

This module ports:
  - ``fcn_get_xss_a.m``       ->  ``find_steady_state_x``
  - ``fcn_get_loop_gain.m``   ->  ``compute_loop_gain``
"""

from __future__ import annotations

import numpy as np


def find_steady_state_x(L: float, g: float, d: float) -> float:
    """Find the steady-state value of x in the Mackey-Glass system.

    At steady state, the Mackey-Glass recurrence simplifies to the
    implicit equation:

        d * x^(g+1) / (1 + x^g) = L

    This is solved by brute-force evaluation on a dense grid of 10,000
    points from 0 to 50, selecting the x that minimises the squared
    residual.

    Direct port of MATLAB ``fcn_get_xss_a(L, g, d)``.

    Args:
        L: CO2 production rate (typically 0.05).
        g: Gamma (Hill exponent).
        d: Drive factor (1 = normal, < 1 = obstructed).

    Returns:
        xss: Steady-state value of x.
    """
    x = np.linspace(0, 50, 10000)
    f = d * x ** (g + 1) / (1 + x**g)
    e = (f - L) ** 2
    return float(x[np.argmin(e)])


def compute_loop_gain(L: float, g: float, d: float) -> float:
    """Compute loop gain from the Mackey-Glass steady-state analysis.

    Direct port of MATLAB ``fcn_get_loop_gain(L, g, d)``.

    The computation proceeds in three steps:
    1. Find steady-state ventilation ``vss`` under normal breathing (d=1).
    2. Find steady-state ventilation ``vd`` during obstruction (drive=d).
    3. Find ventilation ``vr`` immediately after release — the drive built
       up during obstruction now acts on an unobstructed airway.

    Loop gain is the ratio of overshoot-on-release to deficit-during-
    obstruction: ``LG = |vss - vr| / |vss - vd|``.

    Args:
        L: CO2 production rate (typically 0.05).
        g: Gamma (Hill exponent) from EM estimation.
        d: Minimum drive value during obstructive events (from the
           observed d_i signal; values in (0, 1]).

    Returns:
        LG: Loop gain (dimensionless, typically 0.1 to 2+).
    """
    # Steady-state under normal breathing (drive = 1)
    xss = find_steady_state_x(L, g, 1.0)
    vss = xss**g / (1 + xss**g)

    # Steady-state during obstruction (drive = d < 1)
    xd = find_steady_state_x(L, g, d)
    vd = d * xd**g / (1 + xd**g)
    dvd = abs(vss - vd)

    # Ventilation immediately after releasing the obstruction:
    # the accumulated chemoreflex drive xd is now unobstructed (drive = 1)
    vr = xd**g / (1 + xd**g)
    dvr = abs(vss - vr)

    if dvd == 0:
        return np.inf

    return float(dvr / dvd)
