"""Evaluation helpers for binary classification with class imbalance.

These wrap sklearn / scipy primitives with the conventions used in
04_evaluation.ipynb, so notebook cells stay short.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic - standard in credit scoring.

    Measures the maximum separation between the score distributions of
    defaulters and non-defaulters. Industry rule of thumb: KS >= 0.40 is
    a strong model, 0.30-0.40 is acceptable, below 0.30 is weak.
    """
    scores_pos = y_score[y_true == 1]
    scores_neg = y_score[y_true == 0]
    ks, _ = stats.ks_2samp(scores_pos, scores_neg)
    return float(ks)


def compute_headline_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Return the metrics we report in the headline table.

    ROC-AUC, PR-AUC, KS, and base rate. We deliberately exclude accuracy
    because it is misleading on imbalanced data (a 'predict zero' baseline
    gets ~92% accuracy here).
    """
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "ks": ks_statistic(y_true, y_score),
        "base_rate": float(np.mean(y_true)),
    }


def plot_roc_pr_curves(
    y_true: np.ndarray, y_scores: dict[str, np.ndarray], figsize=(12, 5)
) -> plt.Figure:
    """Plot ROC and PR curves side-by-side for one or more models.

    Parameters
    ----------
    y_scores : dict
        Mapping of model_name -> predicted probability array.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=figsize)
    base_rate = np.mean(y_true)

    for name, scores in y_scores.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = roc_auc_score(y_true, scores)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        ax_pr.plot(recall, precision, label=f"{name} (AP={pr_auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.3)

    ax_pr.axhline(base_rate, color="k", linestyle="--", alpha=0.3,
                  label=f"Base rate ({base_rate:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend(loc="upper right")
    ax_pr.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_calibration(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10, figsize=(7, 6)
) -> plt.Figure:
    """Plot a reliability diagram.

    A well-calibrated model has its curve close to the diagonal: among
    applicants the model assigns 30% default risk to, roughly 30% should
    actually default. Calibration matters for credit because the threshold
    and the implied loss provisioning are tied to the score level.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins,
                                              strategy="quantile")
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", label="Model")
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed default rate (bin)")
    ax.set_title("Calibration plot")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def threshold_analysis(
    y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    """Sweep thresholds and report operating-point metrics.

    Returns one row per threshold with the metrics a credit team cares
    about: how many applicants would be flagged, how many true defaulters
    among them, precision, recall.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05)
    rows = []
    n = len(y_true)
    n_defaults = int(np.sum(y_true))
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        flagged = tp + fp
        precision = tp / flagged if flagged > 0 else 0.0
        recall = tp / n_defaults if n_defaults > 0 else 0.0
        rows.append({
            "threshold": round(float(thr), 3),
            "flagged_pct": flagged / n,
            "captured_defaulters_pct": recall,
            "precision_at_threshold": precision,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return pd.DataFrame(rows)
