"""
CPAP treatment outcome prediction using logistic regression.

This module implements the machine-learning pipeline for predicting
CPAP treatment success vs. failure from loop-gain (LG) estimates and
other polysomnography-derived features.  The approach uses
**cross-validated logistic regression** with nested hyperparameter
tuning:

1. **Outer loop** -- manual K-fold CV (default 5 folds) to generate
   held-out predictions that are never seen during training.
2. **Inner loop** -- ``LogisticRegressionCV`` with 3-fold CV inside
   each training split to select the regularisation strength (``C``).

The resulting predictions are evaluated using:
- Confusion matrix, accuracy
- ROC AUC and Precision-Recall AUC with bootstrapped 95 % CIs
- Calibration curves (reliability diagrams)

Design rationale
----------------
* **Balanced class weighting** -- CPAP success/failure cohorts may be
  imbalanced; ``class_weight='balanced'`` inversely weights samples by
  class frequency, preventing the model from simply predicting the
  majority class.
* **Bootstrap CIs** -- parametric confidence intervals assume normality
  of the AUC distribution, which is unreliable for small sample sizes.
  Non-parametric bootstrap (100 resamples) provides a distribution-free
  alternative.
* **Colour coding** -- each feature set (LG, CAI, AHI, SS, Combined)
  has a fixed colour for consistent figure styling across publications.

Source: ``EM_output_to_CPAP_Analysis.py`` (all functions except
``__main__`` block).
"""

from __future__ import annotations

import random
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from matplotlib.lines import Line2D
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Cross-validation fold construction
# ---------------------------------------------------------------------------


def set_cross_validation_folds(
    x: np.ndarray,
    y: np.ndarray,
    folds: int = 5,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Create stratified-like manual K-fold cross-validation splits.

    Shuffles the sample indices deterministically (``random_state=0``)
    and partitions them into ``folds`` contiguous blocks.  The last fold
    absorbs any remainder when ``len(y)`` is not evenly divisible by
    ``folds`` -- this avoids discarding samples at the cost of a slightly
    larger final test set.

    The split is *not* stratified by class label.  The original code
    relied on whole-dataset shuffling to achieve approximate balance,
    which is acceptable when the class ratio is close to 50/50 (as in
    the matched CPAP success/failure cohort).

    Args:
        x: Feature matrix of shape ``(n_samples, n_features)`` or
            ``(n_samples, 1)`` for univariate features.
        y: Binary label array of shape ``(n_samples,)``.
        folds: Number of cross-validation folds.

    Returns:
        A 3-tuple ``(xs, ys, inds)`` where:

        * **xs** -- dict mapping ``'tr_fold_{i}'`` and ``'te_fold_{i}'``
          (1-indexed) to training and test feature arrays.
        * **ys** -- same structure for label arrays.
        * **inds** -- the shuffled index permutation.
    """
    # Deterministic shuffle ensures reproducibility across runs.
    inds = np.array(sklearn.utils.shuffle(range(len(y)), random_state=0))

    xs: dict[str, np.ndarray] = {}
    ys: dict[str, np.ndarray] = {}
    split = len(y) // folds

    for i in range(folds):
        if i < folds - 1:
            # Standard fold: contiguous block of ``split`` samples.
            test_inds = inds[split * i : split * (i + 1)]
        else:
            # Last fold absorbs the remainder to use all data.
            test_inds = inds[split * i :]

        # Training set: complement of the test indices.
        xs[f"tr_fold_{i + 1}"] = x[[j for j in range(len(y)) if j not in test_inds]]
        ys[f"tr_fold_{i + 1}"] = y[[j for j in range(len(y)) if j not in test_inds]]
        # Test set: the held-out block.
        xs[f"te_fold_{i + 1}"] = x[test_inds]
        ys[f"te_fold_{i + 1}"] = y[test_inds]

    return xs, ys, inds


# ---------------------------------------------------------------------------
# Logistic regression with nested cross-validation
# ---------------------------------------------------------------------------


def compute_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    tag: str,
    axs: list[plt.Axes] | None = None,
    CV_folds: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit cross-validated logistic regression and plot ROC / PR curves.

    For each outer fold, a ``LogisticRegressionCV`` is fit on the
    training split (with an inner 3-fold CV over 10 regularisation
    strengths ``Cs=10``).  Predictions from all held-out folds are
    concatenated to form the full-sample prediction vector, which is
    then used for curve plotting and metric computation.

    The ``scoring='roc_auc'`` inner objective was chosen because the
    clinical question (will CPAP therapy fail?) benefits from good
    discrimination across all operating thresholds rather than optimising
    accuracy at a single decision boundary.

    Args:
        x: Feature matrix, shape ``(n_samples, n_features)``.
        y: Binary label vector, shape ``(n_samples,)``.
        tag: Short label for the feature set (e.g. ``'LG'``, ``'CAI'``,
            ``'Combined'``).  Used in plot legends and console output.
        axs: A list of **two** matplotlib ``Axes`` -- ``axs[0]`` for the
            ROC curve and ``axs[1]`` for the PR curve.  If ``None``,
            plots are skipped.
        CV_folds: Number of outer cross-validation folds.

    Returns:
        A tuple ``(prob, y)`` where ``prob`` contains the predicted
        probabilities for the positive class and ``y`` the corresponding
        true labels -- both ordered by the concatenation of held-out
        folds.
    """
    if axs is None:
        axs = []

    # Build the CV fold dictionaries.
    xs, ys, inds = set_cross_validation_folds(x, y, folds=CV_folds)

    for i in range(1, CV_folds + 1):
        # --- Inner cross-validated logistic regression ---
        # Cs=10: test 10 logarithmically spaced regularisation strengths.
        # class_weight='balanced': up-weight the minority class.
        # scoring='roc_auc': optimise for area under the ROC curve.
        clf = LogisticRegressionCV(
            Cs=10,
            random_state=0,
            cv=3,
            class_weight="balanced",
            scoring="roc_auc",
        ).fit(xs[f"tr_fold_{i}"], ys[f"tr_fold_{i}"])

        # Concatenate held-out predictions across folds.
        x_te = xs[f"te_fold_{i}"]
        y_te = ys[f"te_fold_{i}"]

        if i == 1:
            x = x_te
            y = y_te
            pred = clf.predict(x_te)
            prob = clf.predict_proba(x_te)[:, 1]
        else:
            x = np.concatenate([x, x_te])
            y = np.concatenate([y, y_te])
            pred = np.concatenate([pred, clf.predict(x_te)])
            prob = np.concatenate([prob, clf.predict_proba(x_te)[:, 1]])

    # --- Console performance summary ---
    print(f" {tag} CMT:\n{confusion_matrix(y, pred, labels=[0, 1])}")
    print(f" {tag} Acc: {np.round(sum(y == pred) / len(y), 2)}")

    # --- Bootstrap confidence intervals for AUC metrics ---
    mean_roc, CI_roc = do_bootstrapping(y, prob, my_auc_roc)
    mean_pr, CI_pr = do_bootstrapping(y, prob, my_auc_pr)

    # --- Plot ROC and PR curves ---
    line_color = set_line_color(tag)

    if len(axs) >= 2:
        # ROC curve on axs[0].
        fpr, tpr, _ = roc_curve(y, prob)
        area = f"{mean_roc:.2f} [{CI_roc[0]:.2f}-{CI_roc[1]:.2f}]"
        axs[0].plot(fpr, tpr, line_color, label=f"{tag}${area}")

        # Precision-Recall curve on axs[1].
        precision, recall, _ = precision_recall_curve(y, prob)
        area = f"{mean_pr:.2f} [{CI_pr[0]:.2f}-{CI_pr[1]:.2f}]"
        axs[1].plot(recall, precision, line_color, label=f"{tag}${area}")

    return prob, y


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def do_bootstrapping(
    y: np.ndarray,
    proba: np.ndarray,
    my_stat: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 100,
    percentage: str = "95%",
) -> tuple[float, list[float]]:
    """Compute bootstrapped mean and confidence interval for a metric.

    Non-parametric bootstrap: draw ``n_bootstraps`` random samples (with
    replacement) of size *N* from the original sample, compute the
    metric on each resample, then report the mean and the symmetric
    ``percentage`` quantile interval.

    This is preferred over parametric CIs because the sampling
    distribution of AUC (both ROC and PR) can be heavily skewed when
    the sample size is small or class imbalance is moderate -- conditions
    that are common in CPAP outcome studies where the cohort is
    typically 100-200 patients.

    Args:
        y: True binary labels, shape ``(n_samples,)``.
        proba: Predicted probabilities for the positive class.
        my_stat: A callable ``f(y, prob) -> float`` that returns the
            metric to bootstrap (e.g. ``my_auc_roc`` or ``my_auc_pr``).
        n_bootstraps: Number of bootstrap resamples (default 100).
        percentage: Confidence level as a string like ``'95%'``.

    Returns:
        A tuple ``(mean, [lower, upper])`` where ``mean`` is the
        bootstrap mean of the metric and the list contains the lower
        and upper CI bounds, all rounded to 2 decimal places.
    """
    n_options: int = y.shape[0]
    index_original = np.arange(n_options).astype(int)
    metrics: list[float] = []

    for n in range(n_bootstraps):
        print(f"bootstrap #{n + 1} / {n_bootstraps}", end="\r")

        # Sample *with replacement* -- the defining property of the
        # non-parametric bootstrap.
        index_bootstrap = random.choices(index_original, k=n_options)

        true = y[index_bootstrap]
        prob = proba[index_bootstrap]

        metric = my_stat(true, prob)
        metrics.append(metric)

    # Convert percentage string (e.g. '95%') to a float (0.95).
    perc = int(percentage[:-1]) / 100
    metrics_arr = np.array(metrics)
    mean = np.round(np.mean(metrics_arr), 2)
    # Symmetric quantile interval: [1-perc, perc].
    lower_bound = np.round(np.quantile(metrics_arr, 1 - perc, axis=0), 2)
    upper_bound = np.round(np.quantile(metrics_arr, perc, axis=0), 2)

    return mean, [lower_bound, upper_bound]


# ---------------------------------------------------------------------------
# AUC metric functions
# ---------------------------------------------------------------------------


def my_auc_roc(y: np.ndarray, p: np.ndarray) -> float:
    """Compute the Area Under the ROC Curve (ROC AUC).

    The ROC curve plots sensitivity (true positive rate) against
    1 - specificity (false positive rate) across all classification
    thresholds.  AUC summarises overall discriminative ability: 0.5 is
    chance, 1.0 is perfect separation.

    Args:
        y: True binary labels.
        p: Predicted probabilities for the positive class.

    Returns:
        ROC AUC as a scalar float.
    """
    fpr, tpr, _ = roc_curve(y, p)
    return auc(fpr, tpr)


def my_auc_pr(y: np.ndarray, p: np.ndarray) -> float:
    """Compute the Area Under the Precision-Recall Curve (PR AUC).

    The PR curve is more informative than ROC when class imbalance is
    present (common in clinical datasets).  High PR AUC indicates that
    the model achieves both high precision (few false alarms) and high
    recall (few missed cases) simultaneously.

    Args:
        y: True binary labels.
        p: Predicted probabilities for the positive class.

    Returns:
        PR AUC as a scalar float.
    """
    precision, recall, _ = precision_recall_curve(y, p)
    return auc(recall, precision)


# ---------------------------------------------------------------------------
# Visual styling helpers
# ---------------------------------------------------------------------------


def set_line_color(tag: str) -> str:
    """Map a feature-set tag to its canonical matplotlib colour/style.

    Consistent colour coding across all CPAP analysis figures:

    ==========  ===========  =======================================
    Tag         Colour       Rationale
    ==========  ===========  =======================================
    LG          blue         Primary predictor (loop gain)
    LG range    blue         LG with 25th/median/75th percentiles
    LG bar      magenta      Histogram-based LG summary
    CAI         black solid  Central Apnea Index -- clinical baseline
    AHI         black dashed Apnea-Hypopnea Index -- standard metric
    SS          yellow       Self-Similarity percentage
    Combined    red          Multi-feature combined model
    ==========  ===========  =======================================

    Args:
        tag: Feature-set identifier string.

    Returns:
        Matplotlib colour/linestyle string (e.g. ``'b'``, ``'k--'``).
    """
    colors: dict[str, str] = {
        "LG": "b",
        "LG range": "b",
        "LG bar": "m",
        "CAI": "k",
        "AHI": "k--",
        "SS": "y",
        "Combined": "r",
    }
    return colors.get(tag, "k")


def set_AUC_curve_layout(axs: list[plt.Axes], subtitle: str) -> None:
    """Apply publication-ready layout to paired ROC / PR axes.

    The legend is constructed manually to achieve a two-column layout
    where the left column shows feature names (with matching line
    colours/styles) and the right column shows the corresponding AUC
    values (as invisible-line handles so they appear as plain text).

    This manual legend construction is necessary because matplotlib's
    default legend cannot interleave coloured feature names with their
    numeric AUC values in a two-column grid.

    The label format ``'tag$value'`` (with ``$`` as delimiter) is an
    internal convention set by ``compute_logistic_regression``.

    Args:
        axs: A list of exactly two ``Axes`` -- index 0 for ROC, index 1
            for PR.
        subtitle: Legend title text (e.g. ``'Features              AUC'``).
            Words are individually bolded using LaTeX math-mode markup.
    """
    for n in range(2):
        ax = axs[n]
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        # --- Reconstruct legend handles ---
        # Labels are in the form "tag$auc_string"; split to separate
        # feature names from AUC values.
        lines, labels = ax.get_legend_handles_labels()
        tags = [label.split("$")[0] for label in labels]
        vals = [label.split("$")[1] for label in labels]

        # Feature-name handles: inherit colour and linestyle from the
        # plotted lines.
        handles: list[Line2D] = []
        for line, tag in zip(lines, tags):
            handle = Line2D(
                [0],
                [0],
                label=tag,
                c=line.get_c(),
                linestyle=line.get_linestyle(),
            )
            handles.append(handle)

        # AUC-value handles: invisible lines (colour='none') so they
        # appear as text-only entries in the second legend column.
        for val in vals:
            handle = Line2D([0], [0], label=val, c="none")
            handles.append(handle)

        # Bold each word in the subtitle using LaTeX math-mode.
        title = "".join([r"$\bf{" + word + "}$" + " " for word in subtitle.split(" ")]) + "[95% CI]"

        ax.legend(
            handles=handles,
            loc=4,
            ncol=2,
            fontsize=9,
            title=title,
            title_fontsize=10,
            alignment="left",
            facecolor="grey",
            framealpha=0.5,
            edgecolor="k",
            columnspacing=0,
        )

        # --- Axis-specific labels and reference lines ---
        if n == 0:
            # ROC curve panel.
            if "Individual" in subtitle:
                ax.set_title("ROC", fontsize=11, weight="bold")
            ax.set_xlabel("1 - specificity", weight="bold", fontsize=10)
            ax.set_ylabel("sensitivity", weight="bold", fontsize=10)
            # Diagonal reference: ROC AUC = 0.5 (chance level).
            ax.plot([-0.1, 1.1], [-0.1, 1.1], "grey", lw=1)

        if n == 1:
            # Precision-Recall curve panel.
            if "Individual" in subtitle:
                ax.set_title("PR", fontsize=11, weight="bold")
            ax.set_xlabel("sensitivity", weight="bold", fontsize=10)
            ax.set_ylabel("precision", weight="bold", fontsize=10)
            # Anti-diagonal reference for PR space.
            ax.plot([-0.1, 1.1], [1.1, -0.1], "grey", lw=1)


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------


def compute_calibration_curve(
    prob: np.ndarray,
    y: np.ndarray,
    tag: str,
    ax: plt.Axes,
) -> None:
    """Plot a calibration (reliability) curve for predicted probabilities.

    A well-calibrated model produces predicted probabilities that match
    the observed event rates: if the model says "70 % chance of CPAP
    failure" for a group of patients, roughly 70 % of them should
    indeed fail.  The calibration curve plots observed frequency vs.
    predicted probability -- perfect calibration lies on the diagonal.

    Args:
        prob: Predicted probabilities for the positive class.
        y: True binary labels.
        tag: Feature-set label (used for colour and legend entry).
        ax: Matplotlib ``Axes`` to plot on.
    """
    # sklearn returns (fraction_of_positives, mean_predicted_value).
    yy, xx = calibration_curve(y, prob)

    line_color = set_line_color(tag)
    ax.plot(xx, yy, line_color, marker="o", label=tag, markersize=6)


def set_calibration_curve_layout(axs: list[plt.Axes]) -> None:
    """Apply publication-ready layout to calibration-curve axes.

    Adds a diagonal reference line (perfect calibration), sets axis
    limits to [0, 1], and creates a styled legend.

    Args:
        axs: List of ``Axes`` objects (typically a single-element list
            containing the calibration subplot).
    """
    for ax in axs:
        # Diagonal reference: perfect calibration.
        ax.plot([-0.1, 1.1], [-0.1, 1.1], "grey", lw=1)

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_ylabel(
            "Ratio of patients that fail CPAP",
            weight="bold",
            fontsize=10,
        )
        ax.set_title("Calibration", fontsize=11, weight="bold")

        # Bold title for the legend.
        title = "".join([r"$\bf{" + word + "}$" + " " for word in "Features".split(" ")])
        ax.legend(
            loc=4,
            ncol=1,
            fontsize=9,
            title=title,
            title_fontsize=10,
            alignment="left",
            facecolor="grey",
            framealpha=0.5,
            edgecolor="k",
        )
        ax.set_xlabel(
            "Predicted CPAP failure risk",
            weight="bold",
            fontsize=10,
        )
