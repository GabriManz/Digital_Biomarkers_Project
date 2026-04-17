"""Tests for src/phase5_biomarker.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.phase5_biomarker import (
    BD_POST,
    BD_PRE,
    FIG_BARPLOT,
    FIG_BOXPLOT,
    FIG_HEATMAP,
    FIG_ROC,
    cohen_d,
    compute_cas_rates,
    generate_all_figures,
    run_between_group_test,
    run_normality_test,
    run_pre_post_test,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_SUBJECTS = 10
N_SEGMENTS = 200
RNG = np.random.default_rng(42)


def _make_synthetic_inputs(n_subjects: int = N_SUBJECTS, n_segs: int = N_SEGMENTS) -> tuple:
    """Create synthetic cas_labels, v_subject, v_bd arrays."""
    rng = np.random.default_rng(7)
    v_subject = np.repeat(np.arange(1, n_subjects + 1), n_segs // n_subjects)
    v_bd = np.where(np.arange(len(v_subject)) % (n_segs // n_subjects) < (n_segs // n_subjects) // 2,
                    BD_PRE, BD_POST).astype(np.int32)
    cas_labels = rng.integers(0, 2, size=len(v_subject), dtype=np.int32)
    return cas_labels, v_subject.astype(np.int32), v_bd


@pytest.fixture
def cas_inputs() -> tuple:
    return _make_synthetic_inputs()


@pytest.fixture
def cas_df(cas_inputs: tuple) -> pd.DataFrame:
    labels, v_sub, v_bd = cas_inputs
    return compute_cas_rates(labels, v_sub, v_bd)


@pytest.fixture
def metadata_df() -> pd.DataFrame:
    return pd.DataFrame({
        "subject_num": list(range(1, N_SUBJECTS + 1)),
        "bdr_label": ["BDR+" if i < N_SUBJECTS // 2 else "BDR-"
                      for i in range(N_SUBJECTS)],
    })


# ---------------------------------------------------------------------------
# Tests — CAS rate computation
# ---------------------------------------------------------------------------


class TestCasRates:
    def test_cas_rate_values_in_valid_range(self, cas_df: pd.DataFrame) -> None:
        for col in ["cas_rate_pre", "cas_rate_post"]:
            vals = cas_df[col].dropna()
            assert np.all((vals >= 0.0) & (vals <= 1.0))

    def test_cas_rate_vectors_have_correct_length(self, cas_df: pd.DataFrame) -> None:
        assert len(cas_df) == N_SUBJECTS

    def test_delta_cas_equals_pre_minus_post(self, cas_df: pd.DataFrame) -> None:
        valid = cas_df.dropna(subset=["cas_rate_pre", "cas_rate_post", "delta_cas"])
        for _, row in valid.iterrows():
            assert row["delta_cas"] == pytest.approx(
                row["cas_rate_pre"] - row["cas_rate_post"], abs=1e-9
            )


# ---------------------------------------------------------------------------
# Tests — Statistical functions
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_shapiro_pvalue_in_unit_interval(self) -> None:
        data = RNG.standard_normal(30)
        result = run_normality_test(data)
        assert 0.0 <= result.pvalue <= 1.0

    def test_wilcoxon_identical_arrays_pvalue_is_one(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        # Wilcoxon on identical arrays → p-value = 1.0
        result = run_pre_post_test(x, x)
        assert result["pvalue"] == pytest.approx(1.0, abs=1e-9)

    def test_cohen_d_identical_groups_is_zero(self) -> None:
        x = RNG.standard_normal(30)
        assert cohen_d(x, x) == pytest.approx(0.0, abs=1e-9)

    def test_cohen_d_known_value(self) -> None:
        a = np.zeros(30, dtype=float)
        b = np.ones(30, dtype=float)
        d = cohen_d(a, b)
        assert d == pytest.approx(-1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Tests — Figures
# ---------------------------------------------------------------------------


class TestFigures:
    def test_output_figures_are_created(
        self,
        tmp_path: Path,
        cas_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
    ) -> None:
        generate_all_figures(cas_df, metadata_df, tmp_path)
        for fig_name in [FIG_BOXPLOT, FIG_ROC, FIG_BARPLOT, FIG_HEATMAP]:
            assert (tmp_path / fig_name).exists(), f"Missing figure: {fig_name}"
