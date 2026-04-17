"""Phase 2 — Segmentation: load temporal markers and slice signals into
inspiratory and expiratory segments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FS_TARGET: int = 4_000  # Hz — target sample rate after resampling
MIN_SEGMENT_SAMPLES: int = int(0.05 * FS_TARGET)  # 200 samples = 50 ms

# Phase labels (1-based, matching metadata convention)
PHASE_INSPIRATION: int = 1
PHASE_EXPIRATION: int = 2

# Column indices in the (n_cycles × 4) markers array
COL_INSP_START: int = 0
COL_INSP_END: int = 1
COL_EXP_START: int = 2
COL_EXP_END: int = 3

N_MANOEUVRES: int = 6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_markers(path: Path | str) -> list[np.ndarray]:
    """Load a LabChart temporal markers .mat file.

    Args:
        path: Path to the markers .mat file.

    Returns:
        List of 6 numpy arrays, one per manoeuvre, each with shape
        (n_cycles, 4) — columns [insp_start, insp_end, exp_start, exp_end]
        in **seconds**.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file structure is unexpected.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Markers file not found: {path}")

    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)

    # The cell array is stored under a key that matches the variable name.
    # Try 'markers' first, then fall back to the first non-meta key.
    markers_raw = _extract_markers_array(mat)
    return _unwrap_markers(markers_raw)


def time_to_sample(t: float, fs: int = FS_TARGET) -> int:
    """Convert a time in seconds to a sample index.

    Args:
        t: Time in seconds.
        fs: Sample rate in Hz.

    Returns:
        Integer sample index (round half-even).

    Raises:
        ValueError: If fs is not positive.
    """
    if fs <= 0:
        raise ValueError(f"Sample rate must be positive. Got fs={fs}.")
    return int(round(t * fs))


def segment_signal(
    signal: np.ndarray,
    markers: np.ndarray,
    fs: int = FS_TARGET,
) -> list[tuple[np.ndarray, int]]:
    """Slice a signal into inspiratory and expiratory segments for one manoeuvre.

    Args:
        signal: 1D preprocessed signal at sample rate fs.
        markers: (n_cycles, 4) array of cycle timing in seconds.
        fs: Sample rate in Hz.

    Returns:
        List of (segment_array, phase_label) tuples where phase_label is
        PHASE_INSPIRATION (1) or PHASE_EXPIRATION (2). Short segments
        (< MIN_SEGMENT_SAMPLES) are silently skipped.

    Raises:
        ValueError: If signal is not 1D or markers shape is wrong.
    """
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}.")
    markers = np.atleast_2d(markers)
    if markers.shape[1] != 4:
        raise ValueError(f"Markers must have 4 columns, got {markers.shape[1]}.")

    n_samples = len(signal)
    segments: list[tuple[np.ndarray, int]] = []

    for row in markers:
        insp_start = _clamped_sample(row[COL_INSP_START], fs, n_samples)
        insp_end = _clamped_sample(row[COL_INSP_END], fs, n_samples)
        exp_start = _clamped_sample(row[COL_EXP_START], fs, n_samples)
        exp_end = _clamped_sample(row[COL_EXP_END], fs, n_samples)

        insp_seg = signal[insp_start:insp_end]
        if len(insp_seg) >= MIN_SEGMENT_SAMPLES:
            segments.append((insp_seg.copy(), PHASE_INSPIRATION))

        exp_seg = signal[exp_start:exp_end]
        if len(exp_seg) >= MIN_SEGMENT_SAMPLES:
            segments.append((exp_seg.copy(), PHASE_EXPIRATION))

    return segments


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_markers_array(mat: dict) -> np.ndarray:
    """Return the first non-meta array from a loaded .mat dict."""
    meta_keys = {"__header__", "__version__", "__globals__"}
    for key, value in mat.items():
        if key not in meta_keys:
            return np.asarray(value)
    raise ValueError("No data found in markers .mat file.")


def _unwrap_markers(raw: np.ndarray) -> list[np.ndarray]:
    """Convert loaded markers data to a list of (n_cycles, 4) float arrays.

    Handles two formats produced by scipy.io.loadmat:

    1. **3-D float array** ``(N_MANOEUVRES, n_cycles, 4)`` — arises when
       ``np.array(list_of_2d_arrays, dtype=object)`` is saved and loaded
       with ``squeeze_me=True``.
    2. **Object array** of shape ``(N_MANOEUVRES,)`` — arises from a real
       LabChart MATLAB cell array.

    Args:
        raw: Array as returned by scipy.io.loadmat for the markers variable.

    Returns:
        List of (n_cycles, 4) numpy arrays.

    Raises:
        ValueError: If the array cannot be interpreted as N_MANOEUVRES × 4-column markers.
    """
    # Case 1: 3-D numeric array (N_MANOEUVRES, n_cycles, 4)
    if raw.ndim == 3 and raw.shape[2] == 4:
        return [np.asarray(raw[i], dtype=float) for i in range(min(raw.shape[0], N_MANOEUVRES))]

    # Case 2: Object array — each element is a (n_cycles, 4) array
    if raw.dtype == object:
        result: list[np.ndarray] = []
        for item in raw.flat:
            candidate = np.atleast_2d(np.asarray(item, dtype=float))
            if candidate.ndim == 2 and candidate.shape[-1] == 4:
                result.append(candidate)
            if len(result) == N_MANOEUVRES:
                break
        if len(result) >= N_MANOEUVRES:
            return result[:N_MANOEUVRES]

    raise ValueError(
        f"Cannot parse markers: shape={raw.shape}, dtype={raw.dtype}. "
        f"Expected a (N_MANOEUVRES, n_cycles, 4) array or an object array."
    )


def _clamped_sample(t: float, fs: int, n_samples: int) -> int:
    """Convert time to sample index clamped to [0, n_samples - 1]."""
    idx = time_to_sample(t, fs)
    return max(0, min(idx, n_samples - 1))
