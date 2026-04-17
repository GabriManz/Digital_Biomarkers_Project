"""Shared pytest fixtures for all test modules.

All synthetic signals are generated here; no test file should construct data
inline or load real .mat files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

FS = 4_000      # Hz — target sample rate used in all fixtures
DURATION = 1.0  # seconds


# ---------------------------------------------------------------------------
# Signal fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sine_100hz() -> np.ndarray:
    """Pure 100 Hz sine wave at fs=4000 Hz, duration 1 s, amplitude 1."""
    t = np.arange(int(FS * DURATION)) / FS
    return np.sin(2 * np.pi * 100 * t).astype(np.float64)


@pytest.fixture
def sine_50hz() -> np.ndarray:
    """Pure 50 Hz sine — must be fully attenuated by the comb notch filter."""
    t = np.arange(int(FS * DURATION)) / FS
    return np.sin(2 * np.pi * 50 * t).astype(np.float64)


@pytest.fixture
def sine_500hz() -> np.ndarray:
    """Pure 500 Hz sine — must pass through the bandpass filter unattenuated."""
    t = np.arange(int(FS * DURATION)) / FS
    return np.sin(2 * np.pi * 500 * t).astype(np.float64)


@pytest.fixture
def white_noise() -> np.ndarray:
    """White Gaussian noise, fs=4000 Hz, duration 1 s, seed fixed."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(int(FS * DURATION))


# ---------------------------------------------------------------------------
# .mat file fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mat_file(tmp_path: Path) -> Path:
    """Minimal synthetic .mat file matching the LabChart export structure.

    Creates a file with 2 channels × 6 blocks, each block containing 1250
    samples (0.1 s at 12500 Hz) of white noise.
    """
    n_channels, n_blocks, block_len = 2, 6, 1250
    data = np.random.default_rng(0).standard_normal(n_channels * n_blocks * block_len)
    datastart = np.zeros((n_channels, n_blocks), dtype=int)
    dataend = np.zeros((n_channels, n_blocks), dtype=int)
    idx = 0
    for ch in range(n_channels):
        for bl in range(n_blocks):
            # MATLAB 1-based indices
            datastart[ch, bl] = idx + 1
            dataend[ch, bl] = idx + block_len
            idx += block_len

    mat_path = tmp_path / "mock_subject.mat"
    scipy.io.savemat(
        str(mat_path),
        {
            "data": data,
            "datastart": datastart,
            "dataend": dataend,
            "samplerate": np.full((n_channels, n_blocks), 12500),
            "titles": np.array(["RS_LL", "RS_LR"]),
            "unittext": np.array(["V"]),
            "unittextmap": np.ones((n_channels, n_blocks), dtype=int),
        },
    )
    return mat_path


@pytest.fixture
def mock_markers_file(tmp_path: Path) -> Path:
    """Synthetic markers file: 6 manoeuvres × 3 respiratory cycles each.

    Times are in seconds. Each cycle: insp 0.5 s, exp 0.5 s, gap 2 s.
    """
    cycles_per_manoeuvre = 3
    markers = []
    for m in range(6):
        t_offset = m * 10.0
        rows = []
        for c in range(cycles_per_manoeuvre):
            t0 = t_offset + c * 3.0
            rows.append([t0, t0 + 1.0, t0 + 1.0, t0 + 2.0])
        markers.append(np.array(rows))

    markers_path = tmp_path / "mock_markers.mat"
    scipy.io.savemat(
        str(markers_path), {"markers": np.array(markers, dtype=object)}
    )
    return markers_path
