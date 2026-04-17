"""Phase 2 — Dataset construction: iterate all subjects, preprocess signals,
segment into inspiratory/expiratory epochs, and build the metadata vectors.

Expected output: 14 900 segments with four integer metadata vectors.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from src.phase1_io import read_signals
from src.phase1_preprocessing import preprocess_signal
from src.phase2_segmentation import load_markers, segment_signal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FS_TARGET: int = 4_000

# Subject numbering convention
PATIENT_IDS: range = range(1, 24)    # P1–P23 → subject IDs 1–23
CONTROL_IDS: range = range(24, 29)   # C1–C5  → subject IDs 24–28

N_MANOEUVRES: int = 6
N_PRE_BD: int = 3  # first 3 manoeuvres are pre-bronchodilator

# Channel labels (1-based)
CHANNEL_LOWER: int = 1
CHANNEL_UPPER: int = 2

# Bronchodilator labels (1-based)
BD_PRE: int = 1
BD_POST: int = 2

EXPECTED_TOTAL_SEGMENTS: int = 14_900


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_dataset(
    data_dir: Path | str,
    output_dir: Path | str | None = None,
) -> dict:
    """Build the full segment dataset from all subjects.

    Args:
        data_dir: Directory containing the .mat signal and marker files.
        output_dir: Optional path to save dataset.npz and all_signals.pkl.
            If None, files are not saved.

    Returns:
        Dict with keys:
            - ``all_signals``: list of 1D numpy arrays (variable length)
            - ``v_subject``: (N,) int32 array — subject IDs 1–28
            - ``v_bd``: (N,) int32 array — 1=pre, 2=post
            - ``v_channel``: (N,) int32 array — 1=lower, 2=upper
            - ``v_phase``: (N,) int32 array — 1=inspiration, 2=expiration
    """
    data_dir = Path(data_dir)

    all_signals: list[np.ndarray] = []
    v_subject: list[int] = []
    v_bd: list[int] = []
    v_channel: list[int] = []
    v_phase: list[int] = []

    # --- Patients P1–P23 (subject IDs 1–23) ---
    for pid in PATIENT_IDS:
        sig_path = data_dir / f"P{pid}.mat"
        mrk_path = data_dir / f"tP{pid}.mat"
        _process_subject(
            sig_path, mrk_path, pid,
            all_signals, v_subject, v_bd, v_channel, v_phase,
        )

    # --- Controls C1–C5 (subject IDs 24–28) ---
    for cid_offset, cid in enumerate(range(1, 6), start=0):
        subject_id = 24 + cid_offset
        sig_path = data_dir / f"C{cid}.mat"
        mrk_path = data_dir / f"tC{cid}.mat"
        _process_subject(
            sig_path, mrk_path, subject_id,
            all_signals, v_subject, v_bd, v_channel, v_phase,
        )

    result = {
        "all_signals": all_signals,
        "v_subject": np.array(v_subject, dtype=np.int32),
        "v_bd": np.array(v_bd, dtype=np.int32),
        "v_channel": np.array(v_channel, dtype=np.int32),
        "v_phase": np.array(v_phase, dtype=np.int32),
    }

    if output_dir is not None:
        _save_dataset(result, Path(output_dir))

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _process_subject(
    sig_path: Path,
    mrk_path: Path,
    subject_id: int,
    all_signals: list[np.ndarray],
    v_subject: list[int],
    v_bd: list[int],
    v_channel: list[int],
    v_phase: list[int],
) -> None:
    """Load, preprocess, and segment one subject; append to accumulators."""
    mat = read_signals(sig_path)
    markers_list = load_markers(mrk_path)

    n_channels = mat["nchannels"]
    n_blocks = mat["nblocks"]

    for block_idx in range(n_blocks):
        bd_label = BD_PRE if block_idx < N_PRE_BD else BD_POST
        markers = markers_list[block_idx]

        for ch_idx in range(n_channels):
            raw_signal = mat["signals"][ch_idx][block_idx]
            processed = preprocess_signal(raw_signal)
            ch_label = CHANNEL_LOWER if ch_idx == 0 else CHANNEL_UPPER

            segs = segment_signal(processed, markers, fs=FS_TARGET)
            for seg_array, phase_label in segs:
                all_signals.append(seg_array)
                v_subject.append(subject_id)
                v_bd.append(bd_label)
                v_channel.append(ch_label)
                v_phase.append(phase_label)


def _save_dataset(result: dict, output_dir: Path) -> None:
    """Persist the dataset to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "dataset.npz"
    np.savez(
        str(npz_path),
        v_subject=result["v_subject"],
        v_bd=result["v_bd"],
        v_channel=result["v_channel"],
        v_phase=result["v_phase"],
    )

    pkl_path = output_dir / "all_signals.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(result["all_signals"], fh)
