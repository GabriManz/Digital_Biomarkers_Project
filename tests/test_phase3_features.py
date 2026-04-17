"""Tests for src/phase3_features.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.phase3_features import (
    FEATURE_NAMES,
    FS_TARGET,
    N_FEATURES,
    build_feature_matrix,
    extract_all_features,
    extract_cas_features,
    extract_spectral_features,
    extract_temporal_features,
)

FS = 4_000
DURATION = 1.0


def _sine(freq: float, fs: int = FS, duration: float = DURATION) -> np.ndarray:
    t = np.arange(int(fs * duration)) / fs
    return np.sin(2 * np.pi * freq * t).astype(np.float64)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


class TestTemporalFeatures:
    def test_temporal_features_return_correct_shape(self) -> None:
        sig = _sine(300.0)
        out = extract_temporal_features(sig)
        assert out.shape == (6,)

    def test_rms_of_constant_signal(self) -> None:
        sig = np.ones(1000, dtype=np.float64)
        feats = extract_temporal_features(sig)
        rms_val = feats[0]  # index 0 = rms
        assert rms_val == pytest.approx(1.0, rel=1e-6)

    def test_rms_of_sine_signal(self) -> None:
        A = 2.0
        sig = A * _sine(300.0)
        feats = extract_temporal_features(sig)
        expected = A / np.sqrt(2)
        assert feats[0] == pytest.approx(expected, rel=0.01)

    def test_zcr_all_positive_signal_is_zero(self) -> None:
        sig = np.ones(100, dtype=np.float64)
        feats = extract_temporal_features(sig)
        assert feats[1] == pytest.approx(0.0)

    def test_zcr_alternating_signal_is_maximum(self) -> None:
        sig = np.tile([1.0, -1.0], 50).astype(np.float64)
        feats = extract_temporal_features(sig)
        # Every sample crosses zero → zcr ≈ 1 (99/100 transitions for 100 samples)
        assert feats[1] == pytest.approx(1.0, abs=0.05)


class TestSpectralFeatures:
    def test_spectral_features_return_correct_shape(self) -> None:
        sig = _sine(500.0)
        out = extract_spectral_features(sig)
        assert out.shape == (9,)

    def test_mean_freq_of_pure_sine(self, sine_500hz: np.ndarray) -> None:
        feats = extract_spectral_features(sine_500hz)
        mean_f = feats[0]
        assert abs(mean_f - 500.0) <= 10.0

    def test_band_power_fractions_sum_to_one(self) -> None:
        # Use a bandpass-filtered signal so all power falls within 70–1900 Hz,
        # matching the assumption that the pipeline runs bandpass before features.
        from src.phase1_preprocessing import apply_bandpass

        rng = np.random.default_rng(7)
        sig = apply_bandpass(rng.standard_normal(FS), fs=FS)
        feats = extract_spectral_features(sig)
        band_sum = float(np.sum(feats[4:8]))  # four band powers
        assert band_sum == pytest.approx(1.0, abs=0.05)

    def test_spectral_flatness_white_noise_near_one(
        self, white_noise: np.ndarray
    ) -> None:
        feats = extract_spectral_features(white_noise)
        flatness = feats[8]
        assert flatness > 0.5  # white noise is spectrally flat

    def test_spectral_flatness_pure_sine_near_zero(
        self, sine_100hz: np.ndarray
    ) -> None:
        feats = extract_spectral_features(sine_100hz)
        flatness = feats[8]
        assert flatness < 0.2  # tonal signal → low flatness


class TestCasFeatures:
    def test_n_spectral_peaks_for_single_sine(self, sine_100hz: np.ndarray) -> None:
        feats = extract_cas_features(sine_100hz)
        n_peaks = int(round(feats[0]))
        assert n_peaks >= 1  # at least one peak at 100 Hz


class TestFeatureMatrix:
    def test_feature_matrix_contains_no_nan(self) -> None:
        rng = np.random.default_rng(99)
        signals = [rng.standard_normal(FS) for _ in range(20)]
        X = build_feature_matrix(signals)
        assert np.isnan(X).sum() == 0

    def test_feature_matrix_shape(self) -> None:
        rng = np.random.default_rng(1)
        signals = [rng.standard_normal(FS) for _ in range(100)]
        X = build_feature_matrix(signals)
        assert X.shape == (100, N_FEATURES)

    def test_feature_names_length_matches_n_features(self) -> None:
        assert len(FEATURE_NAMES) == N_FEATURES
