"""Phase 4 — Classification: rule-based CAS labelling and sklearn pipelines.

Exports:
- make_rule_based_labels()
- PIPELINES
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Constants — rule-based labelling thresholds
# ---------------------------------------------------------------------------

CAS_FLATNESS_THRESHOLD: float = 0.10   # spectral_flatness < threshold → tonal
CAS_MIN_PEAKS: int = 2                  # n_spectral_peaks_100_1000 >= threshold
CAS_MIN_DURATION_S: float = 0.15       # duration_s > threshold
CAS_MIN_PEAK_PROMINENCE: float = 0.05  # max_peak_prominence_frac >= threshold

# Feature name strings (must match LABELING_FEATURE_NAMES in phase3_features)
FEAT_SPECTRAL_FLATNESS: str = "spectral_flatness"
FEAT_N_PEAKS: str = "n_spectral_peaks_100_1000"
FEAT_DURATION: str = "duration_s"
FEAT_PEAK_PROMINENCE: str = "max_peak_prominence_frac"

# SVM / RF hyperparameters
SVM_KERNEL: str = "rbf"
RF_N_ESTIMATORS: int = 200
KNN_N_NEIGHBORS: int = 7
RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Rule-based ground-truth labels
# ---------------------------------------------------------------------------


def make_rule_based_labels(
    X: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Assign binary CAS labels using rule-based thresholds.

    A segment is labelled CAS=1 when ALL of the following hold:
    - spectral_flatness < CAS_FLATNESS_THRESHOLD
    - n_spectral_peaks_100_1000 >= CAS_MIN_PEAKS
    - duration_s > CAS_MIN_DURATION_S
    - max_peak_prominence_frac >= CAS_MIN_PEAK_PROMINENCE

    Args:
        X: Labeling feature matrix of shape (N, n_labeling_features).
           Must be built with build_labeling_feature_matrix(), NOT the ML
           feature matrix, to avoid circular feature/label dependency.
        feature_names: List of feature names matching the columns of X.
           Pass LABELING_FEATURE_NAMES from phase3_features.

    Returns:
        Binary int32 array of shape (N,) with 1=CAS, 0=normal.

    Raises:
        ValueError: If any required feature name is not found.
    """
    _check_feature(feature_names, FEAT_SPECTRAL_FLATNESS)
    _check_feature(feature_names, FEAT_N_PEAKS)
    _check_feature(feature_names, FEAT_DURATION)
    _check_feature(feature_names, FEAT_PEAK_PROMINENCE)

    i_flat = feature_names.index(FEAT_SPECTRAL_FLATNESS)
    i_peaks = feature_names.index(FEAT_N_PEAKS)
    i_dur = feature_names.index(FEAT_DURATION)
    i_prom = feature_names.index(FEAT_PEAK_PROMINENCE)

    cond_flat = X[:, i_flat] < CAS_FLATNESS_THRESHOLD
    cond_peaks = X[:, i_peaks] >= CAS_MIN_PEAKS
    cond_dur = X[:, i_dur] > CAS_MIN_DURATION_S
    cond_prom = X[:, i_prom] >= CAS_MIN_PEAK_PROMINENCE

    labels = (cond_flat & cond_peaks & cond_dur & cond_prom).astype(np.int32)
    return labels


# ---------------------------------------------------------------------------
# sklearn pipelines
# ---------------------------------------------------------------------------

PIPELINES: dict[str, Pipeline] = {
    "svm": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel=SVM_KERNEL,
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "rf": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=RF_N_ESTIMATORS,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "knn": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS)),
        ]
    ),
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_feature(feature_names: list[str], name: str) -> None:
    """Raise ValueError if feature name is not in the list."""
    if name not in feature_names:
        raise ValueError(
            f"Required feature '{name}' not found in feature_names. "
            f"Available: {feature_names}"
        )
