"""Phase 4 — Validation: Leave-One-Subject-Out cross-validation and metrics.

Exports:
- run_loso()
- compute_metrics()
- save_results_csv()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_FILENAME: str = "classifier_results.csv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_loso(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> list[dict]:
    """Run Leave-One-Subject-Out cross-validation.

    Args:
        pipeline: A fitted or unfitted sklearn Pipeline.
        X: Feature matrix (N, n_features).
        y: Binary labels (N,).
        groups: Subject ID per sample (N,) — one unique value per subject.

    Returns:
        List of per-fold result dicts with keys:
        fold, subject_id, accuracy, sensitivity, specificity, precision, f1, auc.
    """
    cv = LeaveOneGroupOut()
    results: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        subject_id = int(groups[test_idx[0]])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = _get_probabilities(pipeline, X_test)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["fold"] = fold_idx
        metrics["subject_id"] = subject_id
        results.append(metrics)

    return results


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Compute classification metrics for one fold.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for class 1. If None, AUC is NaN.

    Returns:
        Dict with keys: accuracy, sensitivity, specificity, precision, f1, auc.
    """
    acc = float(accuracy_score(y_true, y_pred))
    sensitivity = float(recall_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    specificity = _compute_specificity(y_true, y_pred)

    if y_prob is not None and len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
    else:
        auc = float("nan")

    return {
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc": auc,
    }


def save_results_csv(results: list[dict], output_dir: Path | str) -> Path:
    """Save per-fold LOSO results to a CSV file.

    Args:
        results: List of per-fold result dicts from run_loso().
        output_dir: Directory where classifier_results.csv will be written.

    Returns:
        Path to the saved CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    # Ensure column order
    ordered_cols = ["fold", "subject_id", "accuracy", "sensitivity",
                    "specificity", "precision", "f1", "auc"]
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[ordered_cols]
    out_path = output_dir / RESULTS_FILENAME
    df.to_csv(str(out_path), index=False)
    return out_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity (true negative rate) from binary labels."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp = cm[0, 0], cm[0, 1]
        denom = tn + fp
        return float(tn / denom) if denom > 0 else 0.0
    return 0.0


def _get_probabilities(pipeline: Pipeline, X: np.ndarray) -> np.ndarray | None:
    """Return probability scores for class 1 if the pipeline supports it."""
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)
            return proba[:, 1]
        except Exception:
            pass
    if hasattr(pipeline, "decision_function"):
        try:
            return pipeline.decision_function(X)
        except Exception:
            pass
    return None
