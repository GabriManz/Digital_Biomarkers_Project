"""Tests for src/phase4_classifier.py and src/phase4_validation.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import LeaveOneGroupOut

from src.phase4_classifier import PIPELINES, make_rule_based_labels
from src.phase4_validation import compute_metrics, run_loso, save_results_csv
from src.phase3_features import FEATURE_NAMES, LABELING_FEATURE_NAMES

# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 200
N_SUBJECTS = 4
RNG = np.random.default_rng(42)


@pytest.fixture
def synthetic_X() -> np.ndarray:
    return RNG.standard_normal((N_SAMPLES, len(FEATURE_NAMES)))


@pytest.fixture
def synthetic_y() -> np.ndarray:
    # Balanced binary labels
    y = np.zeros(N_SAMPLES, dtype=np.int32)
    y[: N_SAMPLES // 2] = 1
    rng = np.random.default_rng(1)
    rng.shuffle(y)
    return y


@pytest.fixture
def synthetic_groups() -> np.ndarray:
    # 4 subjects, 50 samples each
    return np.repeat(np.arange(1, N_SUBJECTS + 1), N_SAMPLES // N_SUBJECTS)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestPipelines:
    def test_pipeline_fits_without_error(
        self, synthetic_X: np.ndarray, synthetic_y: np.ndarray
    ) -> None:
        pipeline = PIPELINES["rf"]
        pipeline.fit(synthetic_X, synthetic_y)  # should not raise

    def test_pipeline_predict_output_shape(
        self, synthetic_X: np.ndarray, synthetic_y: np.ndarray
    ) -> None:
        pipeline = PIPELINES["rf"]
        pipeline.fit(synthetic_X, synthetic_y)
        preds = pipeline.predict(synthetic_X)
        assert preds.shape == (N_SAMPLES,)

    def test_pipeline_predict_proba_in_range(
        self, synthetic_X: np.ndarray, synthetic_y: np.ndarray
    ) -> None:
        pipeline = PIPELINES["rf"]
        pipeline.fit(synthetic_X, synthetic_y)
        proba = pipeline.predict_proba(synthetic_X)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_pipeline_predict_binary_output(
        self, synthetic_X: np.ndarray, synthetic_y: np.ndarray
    ) -> None:
        pipeline = PIPELINES["knn"]
        pipeline.fit(synthetic_X, synthetic_y)
        preds = pipeline.predict(synthetic_X)
        assert set(preds.tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# LOSO cross-validation tests
# ---------------------------------------------------------------------------


class TestLOSO:
    def test_loso_produces_correct_number_of_folds(
        self,
        synthetic_X: np.ndarray,
        synthetic_y: np.ndarray,
        synthetic_groups: np.ndarray,
    ) -> None:
        cv = LeaveOneGroupOut()
        folds = list(cv.split(synthetic_X, synthetic_y, groups=synthetic_groups))
        assert len(folds) == N_SUBJECTS

    def test_loso_no_subject_leakage_across_folds(
        self,
        synthetic_X: np.ndarray,
        synthetic_y: np.ndarray,
        synthetic_groups: np.ndarray,
    ) -> None:
        cv = LeaveOneGroupOut()
        for train_idx, test_idx in cv.split(synthetic_X, synthetic_y, groups=synthetic_groups):
            train_subjects = set(synthetic_groups[train_idx].tolist())
            test_subjects = set(synthetic_groups[test_idx].tolist())
            assert train_subjects.isdisjoint(test_subjects)


# ---------------------------------------------------------------------------
# Rule-based labelling test
# ---------------------------------------------------------------------------


class TestRuleBasedLabels:
    def test_rule_based_labels_cas_rate_higher_in_patients(self) -> None:
        """Synthetic: patients have low flatness + peaks, controls have high flatness."""
        rng = np.random.default_rng(5)
        n_patients, n_controls = 100, 50
        n_total = n_patients + n_controls

        # Use the labeling feature matrix (not the ML feature matrix)
        X_label = np.zeros((n_total, len(LABELING_FEATURE_NAMES)))

        i_flat = LABELING_FEATURE_NAMES.index("spectral_flatness")
        i_peaks = LABELING_FEATURE_NAMES.index("n_spectral_peaks_100_1000")
        i_dur = LABELING_FEATURE_NAMES.index("duration_s")
        i_prom = LABELING_FEATURE_NAMES.index("max_peak_prominence_frac")

        # Patients: low flatness, many peaks, long duration, high prominence → high CAS rate
        X_label[:n_patients, i_flat] = rng.uniform(0.01, 0.08, n_patients)
        X_label[:n_patients, i_peaks] = rng.uniform(2, 5, n_patients)
        X_label[:n_patients, i_dur] = rng.uniform(0.2, 1.0, n_patients)
        X_label[:n_patients, i_prom] = rng.uniform(0.1, 0.5, n_patients)

        # Controls: high flatness → CAS = 0
        X_label[n_patients:, i_flat] = rng.uniform(0.5, 1.0, n_controls)
        X_label[n_patients:, i_dur] = rng.uniform(0.2, 1.0, n_controls)
        X_label[n_patients:, i_prom] = rng.uniform(0.0, 0.04, n_controls)

        labels = make_rule_based_labels(X_label, LABELING_FEATURE_NAMES)
        patient_rate = float(np.mean(labels[:n_patients]))
        control_rate = float(np.mean(labels[n_patients:]))
        assert patient_rate > control_rate


# ---------------------------------------------------------------------------
# Results CSV test
# ---------------------------------------------------------------------------


class TestResultsCSV:
    def test_results_csv_has_required_columns(self, tmp_path: Path) -> None:
        fake_results = [
            {
                "fold": 0,
                "subject_id": 1,
                "accuracy": 0.9,
                "sensitivity": 0.85,
                "specificity": 0.92,
                "precision": 0.88,
                "f1": 0.86,
                "auc": 0.91,
            }
        ]
        csv_path = save_results_csv(fake_results, tmp_path)
        import pandas as pd

        df = pd.read_csv(csv_path)
        required = {"fold", "subject_id", "accuracy", "sensitivity",
                    "specificity", "precision", "f1", "auc"}
        assert required.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# Additional coverage tests for phase4_validation.py
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_compute_metrics_perfect_classifier(self) -> None:
        y = np.array([0, 0, 1, 1])
        m = compute_metrics(y, y, y_prob=y.astype(float))
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["sensitivity"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(1.0)

    def test_compute_metrics_no_proba_gives_nan_auc(self) -> None:
        y = np.array([0, 1, 0, 1])
        m = compute_metrics(y, y, y_prob=None)
        assert np.isnan(m["auc"])

    def test_compute_metrics_all_same_class_gives_nan_auc(self) -> None:
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        y_prob = np.zeros(4)
        m = compute_metrics(y_true, y_pred, y_prob=y_prob)
        assert np.isnan(m["auc"])


class TestRunLoso:
    def test_run_loso_returns_one_result_per_subject(
        self,
        synthetic_X: np.ndarray,
        synthetic_y: np.ndarray,
        synthetic_groups: np.ndarray,
    ) -> None:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        fast_pipe = Pipeline([
            ("sc", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=3)),
        ])
        results = run_loso(fast_pipe, synthetic_X, synthetic_y, synthetic_groups)
        assert len(results) == N_SUBJECTS

    def test_run_loso_result_keys(
        self,
        synthetic_X: np.ndarray,
        synthetic_y: np.ndarray,
        synthetic_groups: np.ndarray,
    ) -> None:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        fast_pipe = Pipeline([
            ("sc", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=3)),
        ])
        results = run_loso(fast_pipe, synthetic_X, synthetic_y, synthetic_groups)
        expected_keys = {"fold", "subject_id", "accuracy", "sensitivity",
                         "specificity", "precision", "f1", "auc"}
        assert expected_keys.issubset(set(results[0].keys()))
