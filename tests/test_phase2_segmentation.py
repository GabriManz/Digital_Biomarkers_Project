"""Tests for src/phase2_segmentation.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.phase2_segmentation import (
    FS_TARGET,
    MIN_SEGMENT_SAMPLES,
    PHASE_EXPIRATION,
    PHASE_INSPIRATION,
    load_markers,
    segment_signal,
    time_to_sample,
)


class TestLoadMarkers:
    def test_load_markers_returns_6_manoeuvres(self, mock_markers_file: Path) -> None:
        markers = load_markers(mock_markers_file)
        assert len(markers) == 6

    def test_load_markers_each_element_has_4_columns(
        self, mock_markers_file: Path
    ) -> None:
        markers = load_markers(mock_markers_file)
        for m in markers:
            assert m.ndim == 2
            assert m.shape[1] == 4

    def test_load_markers_invalid_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_markers(Path("/does/not/exist.mat"))


class TestTimeToSample:
    def test_time_zero_maps_to_zero(self) -> None:
        assert time_to_sample(0.0) == 0

    def test_one_second_maps_to_fs(self) -> None:
        assert time_to_sample(1.0, fs=FS_TARGET) == FS_TARGET

    def test_small_time_rounds_correctly(self) -> None:
        # 0.00025 s * 4000 = 1.0 → idx = 1
        assert time_to_sample(0.00025, fs=FS_TARGET) == 1


class TestSegmentSignal:
    def _make_signal(self, duration_s: float = 60.0) -> np.ndarray:
        n = int(duration_s * FS_TARGET)
        rng = np.random.default_rng(0)
        return rng.standard_normal(n)

    def _make_markers(
        self, insp_dur: float = 0.5, exp_dur: float = 0.5, n_cycles: int = 3
    ) -> np.ndarray:
        rows = []
        t = 0.0
        for _ in range(n_cycles):
            rows.append([t, t + insp_dur, t + insp_dur, t + insp_dur + exp_dur])
            t += insp_dur + exp_dur + 0.5  # gap
        return np.array(rows)

    def test_segment_length_matches_time_window(self) -> None:
        signal = self._make_signal()
        insp_dur = 0.5
        markers = self._make_markers(insp_dur=insp_dur)
        segs = segment_signal(signal, markers)
        insp_segs = [s for s, p in segs if p == PHASE_INSPIRATION]
        expected = round(insp_dur * FS_TARGET)
        for seg in insp_segs:
            assert abs(len(seg) - expected) <= 1

    def test_inspiration_end_leq_expiration_start(self) -> None:
        """Inspiration must end at or before expiration starts — no overlap."""
        markers = self._make_markers()
        for row in markers:
            assert row[1] <= row[2], "insp_end must be <= exp_start"

    def test_segment_skips_too_short_cycles(self) -> None:
        signal = self._make_signal()
        # Cycle of 10 ms = 40 samples < MIN_SEGMENT_SAMPLES (200)
        tiny_markers = np.array([[0.0, 0.01, 0.01, 0.02]])
        segs = segment_signal(signal, tiny_markers)
        assert len(segs) == 0

    def test_time_to_sample_boundary_values(self) -> None:
        assert time_to_sample(0.0, FS_TARGET) == 0
        assert time_to_sample(1.0, FS_TARGET) == 4000
        assert time_to_sample(0.00025, FS_TARGET) == 1
