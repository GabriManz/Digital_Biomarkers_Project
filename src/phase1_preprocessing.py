"""Phase 1 — Preprocessing pipeline: resample → bandpass → comb notch.

All steps operate on 1D numpy arrays at the target sample rate (4 000 Hz).
"""

from __future__ import annotations

import math

import numpy as np
import scipy.signal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FS_ORIGINAL: int = 12_500   # Hz — original LabChart sample rate
FS_TARGET: int = 4_000      # Hz — target sample rate after resampling

# Reduced fraction: 4000 / 12500 = 8 / 25
RESAMPLE_UP: int = 8
RESAMPLE_DOWN: int = 25

BANDPASS_LOW: float = 70.0    # Hz — lower cutoff frequency
BANDPASS_HIGH: float = 1900.0  # Hz — upper cutoff frequency
BUTTERWORTH_ORDER: int = 8

NOTCH_FUNDAMENTAL: float = 50.0  # Hz — power line interference
NOTCH_BANDWIDTH: float = 1.0     # Hz — -3 dB notch bandwidth


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resample_signal(
    signal: np.ndarray,
    fs_in: int = FS_ORIGINAL,
    fs_out: int = FS_TARGET,
) -> np.ndarray:
    """Resample a 1D signal from fs_in to fs_out using polyphase filtering.

    Args:
        signal: 1D numpy array of input signal samples.
        fs_in: Original sample rate in Hz. Defaults to FS_ORIGINAL (12 500).
        fs_out: Target sample rate in Hz. Defaults to FS_TARGET (4 000).

    Returns:
        Resampled 1D numpy array.

    Raises:
        ValueError: If signal is not 1D, or if sample rates are not positive.
    """
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}.")
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError(
            f"Sample rates must be positive. Got fs_in={fs_in}, fs_out={fs_out}."
        )

    up, down = _compute_resample_ratio(fs_out, fs_in)
    return scipy.signal.resample_poly(signal, up, down)


def apply_bandpass(signal: np.ndarray, fs: int = FS_TARGET) -> np.ndarray:
    """Apply 8th-order Butterworth bandpass filter (70–1900 Hz), zero-phase.

    Uses SOS form to avoid numerical instability at high filter order.

    Args:
        signal: 1D numpy array at sample rate fs.
        fs: Sample rate in Hz. Defaults to FS_TARGET (4 000).

    Returns:
        Bandpass-filtered 1D numpy array.

    Raises:
        ValueError: If signal is not 1D or fs is not positive.
    """
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}.")
    if fs <= 0:
        raise ValueError(f"Sample rate must be positive. Got fs={fs}.")

    sos = scipy.signal.butter(
        BUTTERWORTH_ORDER,
        [BANDPASS_LOW, BANDPASS_HIGH],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return scipy.signal.sosfiltfilt(sos, signal)


def apply_comb_notch(signal: np.ndarray, fs: int = FS_TARGET) -> np.ndarray:
    """Remove 50 Hz and all harmonics up to Nyquist using frequency-domain notching.

    For each harmonic of NOTCH_FUNDAMENTAL, the FFT bins whose centre frequency
    falls within ±(NOTCH_BANDWIDTH / 2) Hz are set to zero.  An inverse FFT then
    reconstructs the cleaned signal.

    This block-processing approach achieves exact attenuation at the notch
    frequencies with no IIR start-up transients, which would dominate for
    narrow-band (Q ≈ 50) filters applied to short (<5 s) signals.

    Args:
        signal: 1D numpy array at sample rate fs.
        fs: Sample rate in Hz. Defaults to FS_TARGET (4 000).

    Returns:
        Notch-filtered 1D numpy array of the same length as the input.

    Raises:
        ValueError: If signal is not 1D or fs is not positive.
    """
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}.")
    if fs <= 0:
        raise ValueError(f"Sample rate must be positive. Got fs={fs}.")

    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.fft.rfft(signal)

    half_bw = NOTCH_BANDWIDTH / 2.0
    for f in range(int(NOTCH_FUNDAMENTAL), fs // 2, int(NOTCH_FUNDAMENTAL)):
        mask = np.abs(freqs - float(f)) <= half_bw
        spectrum[mask] = 0.0

    return np.fft.irfft(spectrum, n=n)


def preprocess_signal(
    signal: np.ndarray,
    fs_in: int = FS_ORIGINAL,
    fs_out: int = FS_TARGET,
) -> np.ndarray:
    """Apply the full preprocessing pipeline: resample → bandpass → comb notch.

    Args:
        signal: 1D numpy array at the original sample rate fs_in.
        fs_in: Original sample rate in Hz. Defaults to FS_ORIGINAL.
        fs_out: Target sample rate in Hz. Defaults to FS_TARGET.

    Returns:
        Preprocessed 1D numpy array at fs_out.

    Raises:
        ValueError: If signal is not 1D or sample rates are invalid.
    """
    resampled = resample_signal(signal, fs_in=fs_in, fs_out=fs_out)
    bandpassed = apply_bandpass(resampled, fs=fs_out)
    notched = apply_comb_notch(bandpassed, fs=fs_out)
    return notched


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_resample_ratio(fs_out: int, fs_in: int) -> tuple[int, int]:
    """Compute the reduced up/down ratio for polyphase resampling.

    Args:
        fs_out: Target sample rate.
        fs_in: Original sample rate.

    Returns:
        Tuple (up, down) representing the reduced fraction fs_out / fs_in.
    """
    g = math.gcd(fs_out, fs_in)
    return fs_out // g, fs_in // g
