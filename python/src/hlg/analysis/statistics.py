"""
Shared statistical utility functions used across analysis modules.

This module consolidates helper functions that were previously duplicated
in ``EM_output_to_SS_Relationship.py`` and
``EM_output_to_Alitude_Relationship.py``.  Three categories of utilities
are provided:

1. **Dictionary sorting** -- ``sort_dic_keys`` ensures that per-group
   result dictionaries are iterated in alphabetical order, which makes
   downstream CSV output reproducible regardless of insertion order.

2. **Curve fitting helpers** -- ``quadratic_model`` and
   ``prediction_band`` support the second-order polynomial regression
   of SS-vs-LG (self-similarity score vs. loop gain) with frequentist
   prediction intervals based on the Student *t* distribution.

3. **Statistical significance annotation** --
   ``add_statistical_significance`` performs the Mann-Whitney *U* test
   and draws significance brackets on matplotlib axes, following the
   standard ``*`` / ``**`` / ``***`` convention.

All original logic and numerical constants are preserved exactly.

Source files
------------
- ``EM_output_to_SS_Relationship.py``  (``sort_dic_keys``, ``func``, ``predband``)
- ``EM_output_to_Alitude_Relationship.py``  (``sort_dic_keys``, ``func``, ``predband``)
- ``EM_output_to_Group_Analysis.py``  (``add_statistical_significance``)
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Dictionary helpers
# ---------------------------------------------------------------------------


def sort_dic_keys(dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort the keys of each dictionary in a list alphabetically.

    This is used after populating per-group result dictionaries (e.g.
    ``LG_data``, ``G_data``, …) to guarantee that when the dictionaries
    are iterated together (to build a combined CSV), the column order is
    deterministic and reproducible across runs -- even though Python >=3.7
    dicts are insertion-ordered, the insertion order depends on the glob
    ordering of input files which may vary across file systems.

    Args:
        dicts: A list of dictionaries whose keys should be sorted.

    Returns:
        A new list of dictionaries with identical content but with keys
        in sorted (ascending lexicographic) order.

    Example:
        >>> sort_dic_keys([{"b": 2, "a": 1}])
        [{'a': 1, 'b': 2}]
    """
    sorted_dics: list[dict[str, Any]] = []
    for dic in dicts:
        sorted_dic: dict[str, Any] = {}
        # Alphabetical sort ensures reproducible iteration order.
        keys = list(dic.keys())
        keys.sort()
        for key in keys:
            sorted_dic[key] = dic[key]
        sorted_dics.append(sorted_dic)

    return sorted_dics


# ---------------------------------------------------------------------------
# Curve-fitting helpers
# ---------------------------------------------------------------------------


def quadratic_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Second-order polynomial: *a*x^2 + b*x + c*.

    This is the functional form used by ``scipy.optimize.curve_fit`` to
    fit the relationship between self-similarity (SS) score and loop gain
    (LG).  A quadratic was chosen because the SS-vs-LG relationship
    exhibits a concave-upward shape: low SS corresponds to low LG, while
    high SS is associated with disproportionately higher LG, reflecting
    the nonlinear dynamics of ventilatory control instability.

    Args:
        x: Independent variable (SS scores).
        a: Coefficient of the quadratic term.
        b: Coefficient of the linear term.
        c: Intercept (constant term).

    Returns:
        Predicted values *y_hat = a*x^2 + b*x + c* with the same shape as
        ``x``.
    """
    return a * x * x + b * x + c


def prediction_band(
    x: np.ndarray,
    xd: np.ndarray,
    yd: np.ndarray,
    p: tuple[float, ...],
    func: Callable[..., np.ndarray],
    conf: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a prediction band for a fitted curve.

    The prediction band gives the interval within which a *new*
    observation is expected to fall with probability ``conf``, accounting
    for both parameter uncertainty and residual scatter.  It is wider
    than a confidence band (which covers the *mean* response) because it
    includes the irreducible variance of individual measurements.

    The derivation follows the classical formula for prediction intervals
    in nonlinear least-squares regression using the Student *t*
    distribution:

        *y_hat +/- t(1-alpha/2, N-p) * s_e * sqrt(1 + 1/N + (x - x_mean)^2 / sum(x_i - x_mean)^2)*

    where:
        - *N* is the sample size
        - *p* is the number of fitted parameters
        - *s_e* is the residual standard error
        - *t(…)* is the *t*-quantile at the desired confidence level

    The term ``(x - x_mean)^2 / sum(x_i - x_mean)^2`` inflates the band at the
    extremes of the data range where extrapolation uncertainty is
    greater -- a well-known property of regression prediction intervals.

    Args:
        x: Points at which the prediction band is evaluated.
        xd: Observed *x*-data used for the fit (SS scores).
        yd: Observed *y*-data used for the fit (LG values).
        p: Best-fit parameter tuple returned by ``curve_fit``.
        func: The model function (e.g. ``quadratic_model``).
        conf: Confidence level (default 0.95 -> 95 % prediction band).

    Returns:
        A tuple ``(lower_bound, upper_bound)`` arrays, each the same
        shape as ``x``.
    """
    # Significance level (two-tailed).
    alpha: float = 1.0 - conf

    # Sample size and number of free parameters.
    N: int = xd.size
    var_n: int = len(p)

    # Quantile of Student's t-distribution for the desired confidence.
    # Degrees of freedom = N - var_n (residual d.f.).
    q: float = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)

    # Residual standard error: root-mean-square of the residuals,
    # divided by the residual degrees of freedom.
    se: float = np.sqrt(1.0 / (N - var_n) * np.sum((yd - func(xd, *p)) ** 2))

    # Leverage correction: points far from the centroid of the data have
    # higher prediction uncertainty.
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)

    # Best-fit model prediction.
    yp: np.ndarray = func(x, *p)

    # Full prediction interval width: combines residual variance (1.0),
    # mean estimation uncertainty (1/N), and leverage (sx/sxd).
    dy: np.ndarray = q * se * np.sqrt(1.0 + (1.0 / N) + (sx / sxd))

    # Lower and upper prediction bounds.
    lpb: np.ndarray = yp - dy
    upb: np.ndarray = yp + dy

    return lpb, upb


# ---------------------------------------------------------------------------
# Statistical significance annotation
# ---------------------------------------------------------------------------


def add_statistical_significance(
    data: np.ndarray,
    ref_data: np.ndarray,
    pos: int,
    ax: plt.Axes,
    i: int = 0,
) -> None:
    """Compute Mann-Whitney *U* test and draw significance brackets.

    Compares ``data`` against ``ref_data`` using the two-sided
    Mann-Whitney *U* test (a non-parametric test appropriate for
    comparing distributions that may not be normal -- which is the case
    for loop-gain and controller-gain estimates that are bounded and
    often skewed).

    If the p-value is below 0.05, a significance bracket is drawn on
    the matplotlib axis connecting position 1 (the reference group) to
    ``pos`` (the comparison group), annotated with star symbols:

        =========  ===========
        Symbol     Threshold
        =========  ===========
        ``*``      *p* < 0.05
        ``**``     *p* < 0.01
        ``***``    *p* < 0.001
        =========  ===========

    If *p* >= 0.05, nothing is drawn (no "ns" annotation is added).

    The bracket layout (vertical level, offset, hook height) changes
    depending on the parameter ``i``, which controls whether the bracket
    is placed above or below the box-plot region:

    * ``i == 0`` -- bracket is drawn *below* the axes (negative y offset),
      used for the LG parameter panel.
    * ``i != 0`` -- bracket is drawn *above* the axes (positive y offset),
      used for the controller-gain panel.

    This stacking convention prevents brackets from different parameter
    panels from colliding.

    Args:
        data: Array of values for the comparison group.
        ref_data: Array of values for the reference (baseline) group
            (always plotted at x-position 1).
        pos: x-position of the comparison group on the axis.
        ax: Matplotlib ``Axes`` on which to draw the bracket.
        i: Panel index controlling bracket placement direction.
            0 = below axis, non-zero = above axis.
    """
    # Two-sided Mann-Whitney U test (non-parametric).
    U, p = stats.mannwhitneyu(ref_data, data, alternative="two-sided")

    # Map p-value to conventional star notation.
    if p < 0.001:
        sig_symbol = "***"
    elif p < 0.01:
        sig_symbol = "**"
    elif p < 0.05:
        sig_symbol = "*"
    else:
        # Not significant -- skip drawing entirely.
        return

    # --- Bracket geometry ---
    # ``level`` is the baseline y-coordinate; ``offset`` shifts successive
    # brackets so they don't overlap when multiple comparisons are drawn;
    # ``hook`` is the height of the vertical end-caps.
    if i == 0:
        # Below-axis placement (LG panel).
        level = -0.05
        offset = -0.1 * (pos - 1)
        hook = 0.0275
    else:
        # Above-axis placement (gain/delay panels).
        level = 0.05
        offset = -0.06 * (pos - 1)
        hook = 0.015

    # Horizontal line connecting the reference group (pos=1) to the
    # comparison group (pos).
    ax.plot([1, pos], [level + offset] * 2, color="k", lw=1)
    # Left vertical hook (at the reference group position).
    ax.plot([1, 1], [level + offset, level + offset + hook], color="k", lw=1)
    # Right vertical hook (at the comparison group position).
    ax.plot([pos, pos], [level + offset, level + offset + hook], color="k", lw=1)
    # Star symbol placed at the midpoint of the horizontal line, slightly
    # below it so it reads naturally.
    ax.text(
        (1 + pos) / 2,
        level + offset - 0.005,
        sig_symbol,
        va="top",
        ha="center",
    )
