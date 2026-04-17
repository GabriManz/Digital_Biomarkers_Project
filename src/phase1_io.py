"""Phase 1 — Data I/O: Python port of the MATLAB read_signals() function.

Loads LabChart .mat files and reconstructs the signal matrix as a 2D list
of 1D numpy arrays indexed by [channel][block].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_SAMPLERATE: int = 12_500  # Hz — original LabChart sample rate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_signals(path: Path | str) -> dict:
    """Load a LabChart .mat file and reconstruct the signal matrix.

    Args:
        path: Path to the .mat file (str or Path).

    Returns:
        A dict with keys:
            - ``signals``: list[list[np.ndarray]] — signals[ch][block] → 1D array
            - ``nchannels``: int — number of channels
            - ``nblocks``: int — number of blocks (manoeuvres)
            - ``samplerate``: np.ndarray — shape (nchannels, nblocks)
            - ``titles``: list[str] — channel names

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required fields are missing or the data is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)

    _validate_mat_keys(mat)

    data: np.ndarray = np.asarray(mat["data"]).ravel()
    datastart: np.ndarray = np.atleast_2d(mat["datastart"])
    dataend: np.ndarray = np.atleast_2d(mat["dataend"])
    samplerate: np.ndarray = np.atleast_2d(mat["samplerate"])

    nchannels, nblocks = datastart.shape

    signals = _extract_signals(data, datastart, dataend, nchannels, nblocks)
    titles = _extract_titles(mat, nchannels)

    return {
        "signals": signals,
        "nchannels": nchannels,
        "nblocks": nblocks,
        "samplerate": samplerate,
        "titles": titles,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_mat_keys(mat: dict) -> None:
    """Raise ValueError if required keys are absent from the loaded .mat dict."""
    required = {"data", "datastart", "dataend", "samplerate"}
    missing = required - set(mat.keys())
    if missing:
        raise ValueError(f"Missing required fields in .mat file: {missing}")


def _extract_signals(
    data: np.ndarray,
    datastart: np.ndarray,
    dataend: np.ndarray,
    nchannels: int,
    nblocks: int,
) -> list[list[np.ndarray]]:
    """Slice `data` into signals[channel][block] using MATLAB 1-based indices.

    Args:
        data: Flat 1D array of all concatenated signal samples.
        datastart: (nchannels × nblocks) matrix of 1-based start indices.
        dataend: (nchannels × nblocks) matrix of 1-based end indices.
        nchannels: Number of channels.
        nblocks: Number of blocks.

    Returns:
        2D list of 1D numpy arrays.

    Raises:
        ValueError: If any slice produces an empty array.
    """
    signals: list[list[np.ndarray]] = []
    for ch in range(nchannels):
        channel_signals: list[np.ndarray] = []
        for bl in range(nblocks):
            # MATLAB is 1-based → subtract 1 for Python slicing
            start = int(datastart[ch, bl]) - 1
            end = int(dataend[ch, bl])  # end is inclusive in MATLAB, exclusive slice
            segment = data[start:end].astype(np.float64)
            if segment.size == 0:
                raise ValueError(
                    f"Empty segment for channel {ch}, block {bl}. "
                    f"start={start}, end={end}, data length={len(data)}"
                )
            channel_signals.append(segment)
        signals.append(channel_signals)
    return signals


def _extract_titles(mat: dict, nchannels: int) -> list[str]:
    """Extract channel title strings from the .mat dict.

    Args:
        mat: Loaded .mat dict.
        nchannels: Expected number of channels.

    Returns:
        List of channel name strings.
    """
    raw = mat.get("titles", None)
    if raw is None:
        return [f"CH{i}" for i in range(nchannels)]

    raw = np.asarray(raw)
    if raw.ndim == 0:
        raw = raw.reshape(1)

    titles: list[str] = []
    for item in raw.flat:
        if hasattr(item, "item"):
            titles.append(str(item.item()).strip())
        else:
            titles.append(str(item).strip())
    return titles
