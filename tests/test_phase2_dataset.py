"""Tests for src/phase2_dataset.py — build_dataset() using mock data only."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.phase2_dataset import (
    BD_POST,
    BD_PRE,
    CHANNEL_LOWER,
    CHANNEL_UPPER,
    build_dataset,
)
from src.phase2_segmentation import PHASE_EXPIRATION, PHASE_INSPIRATION


# ---------------------------------------------------------------------------
# Helpers — build a tiny synthetic dataset without real .mat files
# ---------------------------------------------------------------------------

FS = 4_000
_SEGMENT_DURATION = 0.5  # seconds
_SEGMENT_LEN = int(_SEGMENT_DURATION * FS)


def _make_synthetic_dataset(
    n_subjects: int = 2,
    n_blocks: int = 6,
    n_cycles: int = 5,
    n_channels: int = 2,
) -> dict:
    """Create a synthetic dataset dict mimicking build_dataset() output."""
    rng = np.random.default_rng(42)
    all_signals = []
    v_subject, v_bd, v_channel, v_phase = [], [], [], []

    seg_per_cycle = 2  # inspiration + expiration
    total = n_subjects * n_blocks * n_channels * n_cycles * seg_per_cycle

    subject_base = 1
    for s in range(n_subjects):
        subject_id = subject_base + s
        for b in range(n_blocks):
            bd = BD_PRE if b < 3 else BD_POST
            for ch in range(n_channels):
                ch_label = CHANNEL_LOWER if ch == 0 else CHANNEL_UPPER
                for _ in range(n_cycles):
                    for phase in [PHASE_INSPIRATION, PHASE_EXPIRATION]:
                        seg = rng.standard_normal(_SEGMENT_LEN)
                        all_signals.append(seg)
                        v_subject.append(subject_id)
                        v_bd.append(bd)
                        v_channel.append(ch_label)
                        v_phase.append(phase)

    return {
        "all_signals": all_signals,
        "v_subject": np.array(v_subject, dtype=np.int32),
        "v_bd": np.array(v_bd, dtype=np.int32),
        "v_channel": np.array(v_channel, dtype=np.int32),
        "v_phase": np.array(v_phase, dtype=np.int32),
    }


@pytest.fixture
def synthetic_dataset() -> dict:
    return _make_synthetic_dataset(n_subjects=4, n_blocks=6, n_cycles=5)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDatasetStructure:
    def test_all_metadata_vectors_have_correct_length(
        self, synthetic_dataset: dict
    ) -> None:
        ds = synthetic_dataset
        n = len(ds["all_signals"])
        assert ds["v_subject"].shape == (n,)
        assert ds["v_bd"].shape == (n,)
        assert ds["v_channel"].shape == (n,)
        assert ds["v_phase"].shape == (n,)

    def test_v_subject_valid_range(self, synthetic_dataset: dict) -> None:
        v = synthetic_dataset["v_subject"]
        assert np.all((v >= 1) & (v <= 28))

    def test_v_bd_only_valid_values(self, synthetic_dataset: dict) -> None:
        assert set(synthetic_dataset["v_bd"].tolist()) == {BD_PRE, BD_POST}

    def test_v_channel_only_valid_values(self, synthetic_dataset: dict) -> None:
        assert set(synthetic_dataset["v_channel"].tolist()) == {CHANNEL_LOWER, CHANNEL_UPPER}

    def test_v_phase_only_valid_values(self, synthetic_dataset: dict) -> None:
        assert set(synthetic_dataset["v_phase"].tolist()) == {PHASE_INSPIRATION, PHASE_EXPIRATION}

    def test_pre_post_bd_roughly_balanced(self, synthetic_dataset: dict) -> None:
        v = synthetic_dataset["v_bd"]
        n = len(v)
        imbalance = abs(np.sum(v == BD_PRE) - np.sum(v == BD_POST)) / n
        assert imbalance < 0.10

    def test_no_empty_segments(self, synthetic_dataset: dict) -> None:
        assert all(len(s) > 0 for s in synthetic_dataset["all_signals"])

    def test_total_segment_count_synthetic(self) -> None:
        """Synthetic: 2 subjects × 6 blocks × 2 channels × 3 cycles × 2 phases = 144."""
        ds = _make_synthetic_dataset(n_subjects=2, n_blocks=6, n_cycles=3, n_channels=2)
        expected = 2 * 6 * 2 * 3 * 2
        assert len(ds["all_signals"]) == expected


# ---------------------------------------------------------------------------
# Build-dataset integration test with mocked I/O
# ---------------------------------------------------------------------------


class TestBuildDatasetMocked:
    """Test build_dataset() with mocked read_signals / preprocess / markers."""

    def _make_mock_mat(self) -> dict:
        """Minimal mat dict: 2 channels × 6 blocks, 200 samples each."""
        rng = np.random.default_rng(0)
        signals = [[rng.standard_normal(200) for _ in range(6)] for _ in range(2)]
        return {
            "signals": signals,
            "nchannels": 2,
            "nblocks": 6,
            "samplerate": np.full((2, 6), 12500),
            "titles": ["RS_LL", "RS_LR"],
        }

    def _make_mock_markers(self) -> list:
        """6 manoeuvres, each with 2 cycles of 0.2 s insp + 0.2 s exp.

        Each phase is 0.2 s = 800 samples > MIN_SEGMENT_SAMPLES (200).
        """
        rows = []
        for m in range(6):
            t0 = m * 2.0
            cycle = np.array([
                [t0, t0 + 0.2, t0 + 0.2, t0 + 0.4],
                [t0 + 0.6, t0 + 0.8, t0 + 0.8, t0 + 1.0],
            ])
            rows.append(cycle)
        return rows

    @patch("src.phase2_dataset.preprocess_signal")
    @patch("src.phase2_dataset.load_markers")
    @patch("src.phase2_dataset.read_signals")
    def test_build_dataset_runs_with_mocked_io(
        self, mock_read, mock_markers, mock_preprocess, tmp_path: Path
    ) -> None:
        """build_dataset() should accumulate segments without raising."""
        mock_read.return_value = self._make_mock_mat()
        mock_markers.return_value = self._make_mock_markers()
        # preprocess returns a 4000-sample signal (1 s at FS_TARGET=4000)
        mock_preprocess.side_effect = lambda s, **kw: np.ones(4000, dtype=float)

        # Only process 1 patient to keep the test fast
        with patch("src.phase2_dataset.PATIENT_IDS", range(1, 2)), \
             patch("src.phase2_dataset.CONTROL_IDS", range(1, 1)):
            result = build_dataset(tmp_path / "fake_data")

        assert len(result["all_signals"]) > 0
        assert result["v_subject"].dtype == np.int32
        assert result["v_bd"].dtype == np.int32

    @patch("src.phase2_dataset.preprocess_signal")
    @patch("src.phase2_dataset.load_markers")
    @patch("src.phase2_dataset.read_signals")
    def test_build_dataset_saves_files(
        self, mock_read, mock_markers, mock_preprocess, tmp_path: Path
    ) -> None:
        """build_dataset() with output_dir saves dataset.npz and all_signals.pkl."""
        mock_read.return_value = self._make_mock_mat()
        mock_markers.return_value = self._make_mock_markers()
        mock_preprocess.side_effect = lambda s, **kw: np.ones(4000, dtype=float)

        with patch("src.phase2_dataset.PATIENT_IDS", range(1, 2)), \
             patch("src.phase2_dataset.CONTROL_IDS", range(1, 1)):
            build_dataset(tmp_path / "data", output_dir=tmp_path / "out")

        assert (tmp_path / "out" / "dataset.npz").exists()
        assert (tmp_path / "out" / "all_signals.pkl").exists()
