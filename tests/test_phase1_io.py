"""Tests for src/phase1_io.py — read_signals()."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

from src.phase1_io import read_signals


class TestReadSignalsKeys:
    def test_read_signals_returns_correct_keys(self, mock_mat_file: Path) -> None:
        result = read_signals(mock_mat_file)
        assert set(result.keys()) >= {"signals", "nchannels", "nblocks", "samplerate", "titles"}

    def test_read_signals_structure_shape(self, mock_mat_file: Path) -> None:
        result = read_signals(mock_mat_file)
        signals = result["signals"]
        assert len(signals) == 2, "Expected 2 channels"
        assert len(signals[0]) == 6, "Expected 6 blocks"

    def test_read_signals_signal_lengths_nonzero(self, mock_mat_file: Path) -> None:
        result = read_signals(mock_mat_file)
        for ch in result["signals"]:
            for block in ch:
                assert len(block) > 0

    def test_read_signals_samplerate_value(self, mock_mat_file: Path) -> None:
        result = read_signals(mock_mat_file)
        assert np.all(result["samplerate"] == 12500)

    def test_read_signals_invalid_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_signals(Path("/nonexistent/path/fake.mat"))

    def test_read_signals_matlab_index_offset(self, tmp_path: Path) -> None:
        """MATLAB datastart=1 (1-based) must map to Python slice starting at index 0."""
        block_len = 100
        data = np.arange(block_len, dtype=float)
        mat_path = tmp_path / "offset_test.mat"
        scipy.io.savemat(
            str(mat_path),
            {
                "data": data,
                "datastart": np.array([[1]]),   # MATLAB 1-based
                "dataend": np.array([[block_len]]),
                "samplerate": np.array([[12500]]),
                "titles": np.array(["CH0"]),
            },
        )
        result = read_signals(mat_path)
        segment = result["signals"][0][0]
        # First element must be data[0] == 0.0 (correct 0-based offset)
        assert segment[0] == pytest.approx(0.0)
        assert len(segment) == block_len
