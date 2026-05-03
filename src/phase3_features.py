"""Phase 3 — Feature extraction: temporal, spectral, and CAS-specific features.

All features are computed per segment (1D numpy array). The module exposes:
- extract_temporal_features()
- extract_spectral_features()
- extract_cas_features()
- extract_all_features()
- build_feature_matrix()
- normalize_amplitude_features_per_subject()
- extract_labeling_features()
- build_labeling_feature_matrix()
- FEATURE_NAMES
- LABELING_FEATURE_NAMES
"""

from __future__ import annotations

import numpy as np
import scipy.signal
import scipy.stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FS_TARGET: int = 4_000         # Hz — target sample rate
WELCH_NPERSEG: int = 512       # Default Welch segment length
SPECTRAL_PEAK_PROMINENCE: float = 0.1  # Relative prominence threshold

# Band boundaries (Hz)
BAND_LOW_LOW: float = 70.0
BAND_LOW_HIGH: float = 200.0
BAND_MID_LOW: float = 200.0
BAND_MID_HIGH: float = 500.0
BAND_HIGH_LOW: float = 500.0
BAND_HIGH_HIGH: float = 1000.0
BAND_VHIGH_LOW: float = 1000.0
BAND_VHIGH_HIGH: float = 1900.0

# CAS feature limits
CAS_FREQ_LOW: float = 100.0   # Hz — lower bound for spectral peak search
CAS_FREQ_HIGH: float = 1000.0  # Hz — upper bound for spectral peak search
NEIGHBOURHOOD_HZ: float = 50.0  # Hz half-width around dominant peak

# ---------------------------------------------------------------------------
# Feature name registry (must match extraction order)
# ---------------------------------------------------------------------------

# ML feature matrix — 16 features (circular labeling features excluded)
FEATURE_NAMES: list[str] = [
    # Temporal (5)
    "rms",
    "zero_crossing_rate",
    "kurtosis",
    "skewness",
    "peak_to_peak",
    # Spectral (8)
    "mean_freq",
    "median_freq",
    "peak_freq",
    "spectral_entropy",
    "band_power_70_200",
    "band_power_200_500",
    "band_power_500_1000",
    "band_power_1000_1900",
    # CAS-specific (3)
    "peak_sharpness",
    "autocorr_peak",
    "harmonic_ratio",
]

N_FEATURES: int = len(FEATURE_NAMES)  # 16

# Labeling-only features — used exclusively by make_rule_based_labels().
# These are NOT in the ML feature matrix to prevent circular label/feature dependency.
LABELING_FEATURE_NAMES: list[str] = [
    "spectral_flatness",
    "n_spectral_peaks_100_1000",
    "duration_s",
    "max_peak_prominence_frac",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_temporal_features(signal: np.ndarray, fs: int = FS_TARGET) -> np.ndarray:
    """Compute 5 temporal features for a 1D signal segment.

    Args:
        signal: 1D numpy array.
        fs: Sample rate in Hz.

    Returns:
        1D array of length 5: [rms, zcr, kurtosis, skewness, peak_to_peak].

    Raises:
        ValueError: If signal is not 1D or is empty.
    """
    _validate_signal(signal)
    rms = float(np.sqrt(np.mean(signal**2)))
    zcr = float(np.sum(np.diff(np.sign(signal)) != 0) / len(signal))
    kurt = float(scipy.stats.kurtosis(signal))
    skew = float(scipy.stats.skew(signal))
    ptp = float(np.ptp(signal))
    return np.array([rms, zcr, kurt, skew, ptp], dtype=np.float64)


def extract_spectral_features(signal: np.ndarray, fs: int = FS_TARGET) -> np.ndarray:
    """Compute 8 spectral features using Welch's PSD estimate.

    Args:
        signal: 1D numpy array.
        fs: Sample rate in Hz.

    Returns:
        1D array of length 8: [mean_freq, median_freq, peak_freq, spectral_entropy,
        band_power_70_200, band_power_200_500, band_power_500_1000, band_power_1000_1900].

    Raises:
        ValueError: If signal is not 1D or is empty.
    """
    _validate_signal(signal)
    nperseg = min(WELCH_NPERSEG, len(signal))
    freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=nperseg)

    total_power = np.sum(psd)
    if total_power == 0.0:
        total_power = 1e-12  # avoid division by zero

    mean_f = float(np.sum(freqs * psd) / total_power)
    median_f = _median_frequency(freqs, psd)
    peak_f = float(freqs[np.argmax(psd)])

    psd_norm = psd / total_power
    entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

    bp_70_200 = _band_power_fraction(freqs, psd, BAND_LOW_LOW, BAND_LOW_HIGH, total_power)
    bp_200_500 = _band_power_fraction(freqs, psd, BAND_MID_LOW, BAND_MID_HIGH, total_power)
    bp_500_1000 = _band_power_fraction(freqs, psd, BAND_HIGH_LOW, BAND_HIGH_HIGH, total_power)
    bp_1000_1900 = _band_power_fraction(freqs, psd, BAND_VHIGH_LOW, BAND_VHIGH_HIGH, total_power)

    return np.array(
        [mean_f, median_f, peak_f, entropy, bp_70_200, bp_200_500,
         bp_500_1000, bp_1000_1900],
        dtype=np.float64,
    )


def extract_cas_features(signal: np.ndarray, fs: int = FS_TARGET) -> np.ndarray:
    """Compute 3 CAS-specific features.

    Args:
        signal: 1D numpy array.
        fs: Sample rate in Hz.

    Returns:
        1D array of length 3: [peak_sharpness, autocorr_peak, harmonic_ratio].

    Raises:
        ValueError: If signal is not 1D or is empty.
    """
    _validate_signal(signal)
    nperseg = min(WELCH_NPERSEG, len(signal))
    freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=nperseg)

    sharpness = _peak_sharpness(freqs, psd)
    ac_peak = _autocorr_peak(signal)
    harm_ratio = _harmonic_ratio(freqs, psd, fs)

    return np.array([sharpness, ac_peak, harm_ratio], dtype=np.float64)


def extract_all_features(signal: np.ndarray, fs: int = FS_TARGET) -> np.ndarray:
    """Concatenate temporal + spectral + CAS features into a single vector.

    Args:
        signal: 1D numpy array.
        fs: Sample rate in Hz.

    Returns:
        1D array of length N_FEATURES (16).
    """
    temporal = extract_temporal_features(signal, fs)
    spectral = extract_spectral_features(signal, fs)
    cas = extract_cas_features(signal, fs)
    return np.concatenate([temporal, spectral, cas])


def build_feature_matrix(
    all_signals: list[np.ndarray],
    fs: int = FS_TARGET,
) -> np.ndarray:
    """Extract features for all segments and return a 2D feature matrix.

    NaN and Inf values are replaced by the column median.

    Args:
        all_signals: List of 1D numpy arrays (variable length).
        fs: Sample rate in Hz.

    Returns:
        (N, N_FEATURES) float64 array. Call normalize_amplitude_features_per_subject()
        on the result before classifier training to remove per-subject amplitude bias.
    """
    rows = [extract_all_features(s, fs) for s in all_signals]
    X = np.vstack(rows)
    X = _impute_nonfinite(X)
    return X


def normalize_amplitude_features_per_subject(
    X: np.ndarray,
    feature_names: list[str],
    v_subject: np.ndarray,
) -> np.ndarray:
    """Z-score 'rms' and 'peak_to_peak' per subject to remove sensor-level amplitude bias.

    Mean and std are computed from that subject's segments only.
    Subjects with a single segment or zero std are left unchanged.

    Args:
        X: (N, n_features) feature matrix.
        feature_names: List of feature names matching columns of X.
        v_subject: (N,) int array of subject IDs.

    Returns:
        Copy of X with rms and peak_to_peak columns z-scored per subject.
    """
    X = X.copy()
    amp_cols = [i for i, n in enumerate(feature_names) if n in ("rms", "peak_to_peak")]
    for sid in np.unique(v_subject):
        mask = v_subject == sid
        for col in amp_cols:
            vals = X[mask, col]
            std = float(np.std(vals, ddof=1)) if int(np.sum(mask)) > 1 else 0.0
            if std > 0.0:
                X[mask, col] = (vals - float(np.mean(vals))) / std
    return X


def extract_labeling_features(signal: np.ndarray, fs: int = FS_TARGET) -> np.ndarray:
    """Compute 4 features used exclusively by the CAS rule-based labeling function.

    These features are NOT included in the ML feature matrix (FEATURE_NAMES) to
    prevent a circular label/feature dependency.

    Args:
        signal: 1D numpy array.
        fs: Sample rate in Hz.

    Returns:
        1D array of length 4 matching LABELING_FEATURE_NAMES:
        [spectral_flatness, n_spectral_peaks_100_1000, duration_s, max_peak_prominence_frac].
    """
    _validate_signal(signal)
    nperseg = min(WELCH_NPERSEG, len(signal))
    freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=nperseg)
    flatness = float(scipy.stats.gmean(psd + 1e-12) / np.mean(psd + 1e-12))
    n_peaks = _count_spectral_peaks(freqs, psd)
    duration_s = float(len(signal) / fs)
    max_prom = _max_peak_prominence_fraction(freqs, psd)
    return np.array([flatness, n_peaks, duration_s, max_prom], dtype=np.float64)


def build_labeling_feature_matrix(
    all_signals: list[np.ndarray],
    fs: int = FS_TARGET,
) -> np.ndarray:
    """Build (N, 4) labeling feature matrix for use with make_rule_based_labels().

    Separate from the ML feature matrix — these features are used only to
    generate CAS labels and must not be passed to the classifier.

    Args:
        all_signals: List of 1D numpy arrays.
        fs: Sample rate in Hz.

    Returns:
        (N, 4) float64 array with columns matching LABELING_FEATURE_NAMES.
    """
    rows = [extract_labeling_features(s, fs) for s in all_signals]
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_signal(signal: np.ndarray) -> None:
    """Raise ValueError if signal is not a non-empty 1D array."""
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}.")
    if len(signal) == 0:
        raise ValueError("Signal must not be empty.")


def _median_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Return the frequency where cumulative power first reaches 50% of total."""
    cumulative = np.cumsum(psd)
    total = cumulative[-1]
    if total == 0.0:
        return float(freqs[0])
    idx = np.searchsorted(cumulative, 0.5 * total)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def _band_power_fraction(
    freqs: np.ndarray,
    psd: np.ndarray,
    f_low: float,
    f_high: float,
    total_power: float,
) -> float:
    """Fraction of total PSD power within [f_low, f_high]."""
    mask = (freqs >= f_low) & (freqs < f_high)
    band_p = float(np.sum(psd[mask]))
    return band_p / total_power if total_power > 0 else 0.0


def _count_spectral_peaks(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Count PSD peaks whose frequency falls in [CAS_FREQ_LOW, CAS_FREQ_HIGH].

    Peaks are detected on the full PSD to avoid edge effects from masking
    (a band boundary can clip a real peak and prevent its detection).
    Only peaks whose centre frequency lies within the target band are counted.
    """
    if len(psd) < 3:
        return 0.0
    prominence_threshold = SPECTRAL_PEAK_PROMINENCE * float(np.max(psd))
    peaks, _ = scipy.signal.find_peaks(psd, prominence=prominence_threshold)
    in_band = (freqs[peaks] >= CAS_FREQ_LOW) & (freqs[peaks] <= CAS_FREQ_HIGH)
    return float(np.sum(in_band))


def _max_peak_prominence_fraction(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Max topographic prominence of peaks in CAS band, relative to max(psd).

    Uses scipy prominence (height above surrounding valleys), not the
    relative threshold used by _count_spectral_peaks.
    Returns 0.0 if no peaks exist in the band or psd is flat.
    """
    if len(psd) < 3:
        return 0.0
    peaks, props = scipy.signal.find_peaks(psd, prominence=0)
    if len(peaks) == 0:
        return 0.0
    in_band = (freqs[peaks] >= CAS_FREQ_LOW) & (freqs[peaks] <= CAS_FREQ_HIGH)
    if not np.any(in_band):
        return 0.0
    max_psd = float(np.max(psd))
    if max_psd == 0.0:
        return 0.0
    return float(np.max(props["prominences"][in_band]) / max_psd)


def _peak_sharpness(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Power at dominant peak divided by power in ±50 Hz neighbourhood."""
    peak_idx = int(np.argmax(psd))
    peak_power = float(psd[peak_idx])
    f0 = float(freqs[peak_idx])
    mask = np.abs(freqs - f0) <= NEIGHBOURHOOD_HZ
    neighbourhood_power = float(np.sum(psd[mask]))
    if neighbourhood_power == 0.0:
        return 0.0
    return peak_power / neighbourhood_power


def _autocorr_peak(signal: np.ndarray) -> float:
    """Max normalised autocorrelation at lags 1 to len(signal)//2."""
    n = len(signal)
    if n < 4:
        return 0.0
    # Full autocorrelation via FFT
    sig = signal - np.mean(signal)
    var = float(np.var(sig))
    if var == 0.0:
        return 0.0
    corr = np.correlate(sig, sig, mode="full")
    corr = corr[n - 1:]  # keep lags >= 0
    corr_norm = corr / (var * n)
    max_lag = n // 2
    return float(np.max(corr_norm[1:max_lag + 1]))


def _harmonic_ratio(freqs: np.ndarray, psd: np.ndarray, fs: int) -> float:
    """Power at fundamental frequency f0 divided by power at 2×f0."""
    nyquist = fs / 2.0
    mask = (freqs >= CAS_FREQ_LOW) & (freqs <= CAS_FREQ_HIGH)
    if not np.any(mask):
        return 0.0
    psd_band = psd.copy()
    psd_band[~mask] = 0.0
    peak_idx = int(np.argmax(psd_band))
    f0 = float(freqs[peak_idx])
    if 2 * f0 >= nyquist:
        return 0.0
    # Find closest bin to 2×f0
    second_idx = int(np.argmin(np.abs(freqs - 2 * f0)))
    denom = float(psd[second_idx])
    if denom == 0.0:
        return 0.0
    return float(psd[peak_idx]) / denom


def _impute_nonfinite(X: np.ndarray) -> np.ndarray:
    """Replace NaN and Inf in each column with the column median."""
    X = X.copy()
    for col in range(X.shape[1]):
        bad = ~np.isfinite(X[:, col])
        if np.any(bad):
            good_vals = X[~bad, col]
            median = float(np.median(good_vals)) if len(good_vals) > 0 else 0.0
            X[bad, col] = median
    return X
