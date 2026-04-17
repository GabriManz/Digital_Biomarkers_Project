"""Tests for src/phase1_preprocessing.py."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal

from src.phase1_preprocessing import (
    FS_ORIGINAL,
    FS_TARGET,
    apply_bandpass,
    apply_comb_notch,
    preprocess_signal,
    resample_signal,
)

FS = 4_000
DURATION = 1.0


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def _make_sine(freq: float, fs: int = FS_ORIGINAL, duration: float = 0.5) -> np.ndarray:
    t = np.arange(int(fs * duration)) / fs
    return np.sin(2 * np.pi * freq * t).astype(np.float64)


class TestResample:
    def test_resample_output_length(self) -> None:
        n_in = 12500  # 1 s at 12500 Hz
        signal = np.random.default_rng(0).standard_normal(n_in)
        out = resample_signal(signal, fs_in=FS_ORIGINAL, fs_out=FS_TARGET)
        expected = int(n_in * FS_TARGET / FS_ORIGINAL)
        assert abs(len(out) - expected) <= 2

    def test_resample_preserves_frequency_content(self) -> None:
        """Resample a 200 Hz sine; peak frequency must still be 200 Hz ±5 Hz."""
        signal = _make_sine(200.0, fs=FS_ORIGINAL, duration=1.0)
        out = resample_signal(signal)
        freqs, psd = scipy.signal.welch(out, fs=FS_TARGET, nperseg=512)
        peak_freq = freqs[np.argmax(psd)]
        assert abs(peak_freq - 200.0) <= 5.0


class TestBandpass:
    def test_bandpass_attenuates_below_cutoff(self) -> None:
        """30 Hz sine must be attenuated to < 1% of input RMS."""
        sig = _make_sine(30.0, fs=FS_TARGET, duration=1.0)
        out = apply_bandpass(sig, fs=FS_TARGET)
        assert _rms(out) < 0.01 * _rms(sig)

    def test_bandpass_attenuates_above_cutoff(self) -> None:
        """1950 Hz sine must be attenuated to < 1% of input RMS."""
        sig = _make_sine(1950.0, fs=FS_TARGET, duration=1.0)
        out = apply_bandpass(sig, fs=FS_TARGET)
        assert _rms(out) < 0.01 * _rms(sig)

    def test_bandpass_passes_in_band(self, sine_500hz: np.ndarray) -> None:
        """500 Hz sine must pass with RMS > 90% of input."""
        out = apply_bandpass(sine_500hz, fs=FS_TARGET)
        assert _rms(out) > 0.90 * _rms(sine_500hz)


class TestCombNotch:
    def test_notch_attenuates_50hz(self, sine_50hz: np.ndarray) -> None:
        """50 Hz sine must be attenuated to < 1% after full comb filter."""
        out = apply_comb_notch(sine_50hz, fs=FS_TARGET)
        assert _rms(out) < 0.01 * _rms(sine_50hz)

    def test_notch_attenuates_150hz_harmonic(self) -> None:
        """150 Hz (3rd harmonic) must be attenuated to < 1%."""
        t = np.arange(FS) / FS
        sig = np.sin(2 * np.pi * 150 * t).astype(np.float64)
        out = apply_comb_notch(sig, fs=FS_TARGET)
        assert _rms(out) < 0.01 * _rms(sig)

    def test_notch_preserves_off_harmonic_frequency(self) -> None:
        """175 Hz (not a harmonic of 50) must keep RMS > 90%."""
        t = np.arange(FS) / FS
        sig = np.sin(2 * np.pi * 175 * t).astype(np.float64)
        out = apply_comb_notch(sig, fs=FS_TARGET)
        assert _rms(out) > 0.90 * _rms(sig)


class TestFullPipeline:
    def test_full_pipeline_no_nan_no_inf(self, white_noise: np.ndarray) -> None:
        """Pipeline on white noise must produce all-finite output."""
        # white_noise is at FS_TARGET already; we need FS_ORIGINAL-length input
        rng = np.random.default_rng(7)
        sig = rng.standard_normal(FS_ORIGINAL)  # 1 s at 12500 Hz
        out = preprocess_signal(sig)
        assert np.isfinite(out).all()

    def test_full_pipeline_output_length_ratio(self) -> None:
        """Output/input length ratio must be within 1% of 4000/12500."""
        n_in = 25000  # 2 s at 12500 Hz
        sig = np.random.default_rng(3).standard_normal(n_in)
        out = preprocess_signal(sig)
        ratio = len(out) / n_in
        expected = FS_TARGET / FS_ORIGINAL
        assert abs(ratio - expected) / expected < 0.01
