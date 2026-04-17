# Action Plan — Bronchodilator Response Assessment via Respiratory Sound Analysis

## Project overview

**Goal**: Evaluate bronchodilator response (BDR) in asthmatic patients by analysing respiratory
sounds (RS) recorded before and after bronchodilator administration. The key hypothesis is that
the number of continuous adventitious sounds (CAS) — wheezes, rhonchi, stridors — can serve
as a digital biomarker for BDR.

**Course**: Biomarcadores Digitales e Inteligencia Artificial en la Asistencia Sanitaria  
**Master**: Máster Universitario en Tecnologías Biomédicas Avanzadas — UPC BarcelonaTech  
**Language**: Python 3.10+  
**Team size**: 5 people  
**Test runner**: `pytest`

---

## Dataset

- **23 asthmatic patients**: `P1.mat` – `P23.mat` + temporal markers `tP1.mat` – `tP23.mat`
- **5 healthy controls**: `C1.mat` – `C5.mat` + temporal markers `tC1.mat` – `tC5.mat`
- **Per subject**: 6 breathing manoeuvres (3 pre-bronchodilator + 3 post-bronchodilator)
- **Per manoeuvre**: 2 sound channels (lower: RS_LL / RS_LR, upper: RS_UL / RS_UR)
- **Original sample rate**: 12,500 Hz
- **BDR label**: see `database/subject_metadata.csv` — each subject is labelled BDR+ or BDR−

### `.mat` file structure (from `read_signals.m`)

Each `.mat` file contains:
- `data`: flat 1D vector with all signal samples concatenated
- `datastart`: n_channels × n_blocks matrix — start index of each signal in `data`
- `dataend`: n_channels × n_blocks matrix — end index of each signal in `data`
- `samplerate`: n_channels × n_blocks matrix — sample rate per channel/block (always 12500)
- `titles`: channel names
- `unittext`: unit strings
- `unittextmap`: index into `unittext` per channel/block

Each temporal markers file (`tPX.mat` / `tCX.mat`) contains:
- A cell array of 6 cells (one per manoeuvre)
- Each cell: n_cycles × 4 array — columns [insp_start, insp_end, exp_start, exp_end] in **seconds**

---

## Expected final output

After preprocessing and segmentation of all subjects:
- **14,900 signal segments** stored in a Python list of numpy arrays
- **4 metadata vectors** of shape (14900,) dtype int:
  - `v_subject`: subject ID — 1–23 = patients, 24–28 = controls
  - `v_bd`: 1 = pre-bronchodilator, 2 = post-bronchodilator
  - `v_channel`: 1 = lower channel, 2 = upper channel
  - `v_phase`: 1 = inspiration, 2 = expiration

---

## Repository structure

```
project/
├── ACTION_PLAN.md               ← this file
├── README.md
├── requirements.txt
├── pytest.ini                   ← pytest configuration
├── .gitignore                   ← exclude data/ and outputs/
├── data/                        ← raw .mat files (NOT committed to git)
│   ├── P1.mat ... P23.mat
│   ├── tP1.mat ... tP23.mat
│   ├── C1.mat ... C5.mat
│   └── tC1.mat ... tC5.mat
├── database/
│   └── subject_metadata.csv     ← subject_id, sex, bdr_label
├── src/
│   ├── __init__.py
│   ├── phase1_io.py             ← read_signals() Python port
│   ├── phase1_preprocessing.py  ← resample + Butterworth + comb notch
│   ├── phase2_segmentation.py   ← load markers, segment insp/exp
│   ├── phase2_dataset.py        ← build the 14900-segment dataset
│   ├── phase3_features.py       ← temporal, spectral, CAS-specific features
│   ├── phase4_classifier.py     ← CAS vs normal RS classifier
│   ├── phase4_validation.py     ← LOSO cross-validation + metrics
│   └── phase5_biomarker.py      ← CAS count analysis + statistics
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_check.ipynb
│   ├── 03_segmentation_check.ipynb
│   ├── 04_features_exploration.ipynb
│   ├── 05_classifier_results.ipynb
│   └── 06_biomarker_analysis.ipynb
├── outputs/
│   ├── figures/
│   └── results/
└── tests/
    ├── conftest.py              ← shared pytest fixtures (synthetic signals, mock .mat files)
    ├── test_phase1_io.py
    ├── test_phase1_preprocessing.py
    ├── test_phase2_segmentation.py
    ├── test_phase2_dataset.py
    ├── test_phase3_features.py
    ├── test_phase4_classifier.py
    └── test_phase5_biomarker.py
```

---

## Coding standards (apply to all phases)

These rules must be followed consistently across the entire codebase.

### Style
- Follow **PEP 8** — use `ruff` for linting: `ruff check src/ tests/`
- Maximum line length: **100 characters**
- Use **type hints** on all function signatures (Python 3.10+ syntax)
- Use **Google-style docstrings** on every public function and class

### Naming conventions
- Functions and variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Classes: `PascalCase`
- Private helpers: prefix with single underscore `_helper_name`

### Function design
- Each function does **one thing** — keep functions under ~40 lines
- **No magic numbers** — define all numeric constants at the top of each module
- Functions must **validate their inputs** and raise `ValueError` with a descriptive message
- Functions must **never silently swallow exceptions**

### Example of expected function style

```python
# src/phase1_preprocessing.py

FS_ORIGINAL: int = 12_500    # Hz — original LabChart sample rate
FS_TARGET: int = 4_000       # Hz — target sample rate after resampling
RESAMPLE_UP: int = 8         # numerator of reduced fraction 4000/12500
RESAMPLE_DOWN: int = 25      # denominator of reduced fraction 4000/12500

BANDPASS_LOW: float = 70.0   # Hz — lower cutoff frequency
BANDPASS_HIGH: float = 1900.0  # Hz — upper cutoff frequency
BUTTERWORTH_ORDER: int = 8

NOTCH_FUNDAMENTAL: float = 50.0  # Hz — power line interference frequency
NOTCH_BANDWIDTH: float = 1.0     # Hz — -3 dB notch bandwidth


def resample_signal(signal: np.ndarray, fs_in: int = FS_ORIGINAL, fs_out: int = FS_TARGET) -> np.ndarray:
    """Resample a 1D signal from fs_in to fs_out using polyphase filtering.

    Args:
        signal: 1D numpy array of input signal samples.
        fs_in: Original sample rate in Hz. Defaults to FS_ORIGINAL (12500).
        fs_out: Target sample rate in Hz. Defaults to FS_TARGET (4000).

    Returns:
        Resampled 1D numpy array.

    Raises:
        ValueError: If signal is not 1D, or if sample rates are not positive integers.
    """
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}.")
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError(f"Sample rates must be positive. Got fs_in={fs_in}, fs_out={fs_out}.")

    up, down = _compute_resample_ratio(fs_out, fs_in)
    return scipy.signal.resample_poly(signal, up, down)
```

---

## Testing strategy

### Philosophy
- **Test every function in isolation** before integrating it into the pipeline
- Use **synthetic signals with known properties** to verify correctness (e.g. a pure 100 Hz sine
  to test the notch filter) — never use real `.mat` files in unit tests
- Tests must be **fast** — no single test should take more than 1 second
- Tests must be **deterministic** — fix all random seeds with `np.random.seed(42)`
- Run the full suite after completing each phase: `pytest tests/ -v`
- Aim for **> 80% code coverage**: `pytest --cov=src --cov-report=term-missing`
- A phase is only considered **done** when all its tests pass and coverage is above the threshold

### `tests/conftest.py` — shared fixtures

Define all reusable fixtures here. No test file should construct synthetic data inline.

```python
# tests/conftest.py

import numpy as np
import pytest
import scipy.io
from pathlib import Path

FS = 4_000       # Hz — target sample rate used in all fixtures
DURATION = 1.0   # seconds


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


@pytest.fixture
def mock_mat_file(tmp_path: Path) -> Path:
    """Minimal synthetic .mat file matching the LabChart export structure.

    Creates a file with 2 channels × 6 blocks, each block containing 1250 samples
    (0.1 s at 12500 Hz) of white noise.
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
    scipy.io.savemat(str(mat_path), {
        "data": data,
        "datastart": datastart,
        "dataend": dataend,
        "samplerate": np.full((n_channels, n_blocks), 12500),
        "titles": np.array(["RS_LL", "RS_LR"]),
        "unittext": np.array(["V"]),
        "unittextmap": np.ones((n_channels, n_blocks), dtype=int),
    })
    return mat_path


@pytest.fixture
def mock_markers_file(tmp_path: Path) -> Path:
    """Synthetic markers file: 6 manoeuvres × 3 respiratory cycles each.

    Times are in seconds. Each cycle: insp 0.5 s, exp 0.5 s.
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
    scipy.io.savemat(str(markers_path), {"markers": np.array(markers, dtype=object)})
    return markers_path
```

---

## Phase 1 — Data reading and preprocessing

**Owner**: Person 1  
**Input**: raw `.mat` files  
**Output**: preprocessed signals dict

### 1.1 Port `read_signals()` to Python — `src/phase1_io.py`

Reimplement the MATLAB `read_signals()` function using `scipy.io.loadmat`.

The function must:
- Accept a `pathlib.Path` or `str` file path
- Load with `scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)`
- Reconstruct `signals` as a 2D list `signals[channel_idx][block_idx]` → 1D numpy array
- Return a dict with keys: `signals`, `nchannels`, `nblocks`, `samplerate`, `titles`

**Critical indexing note**: MATLAB indices are 1-based. Subtract 1 from every value in
`datastart` and `dataend` before using them as Python slice indices.

**MATLAB cell array note**: nested arrays from `scipy.io.loadmat` may be wrapped in object
arrays — unwrap with `.item()` or `.flat[0]` as needed.

### 1.2 Preprocessing pipeline — `src/phase1_preprocessing.py`

Apply the following steps **in this exact order** to every signal (channel × manoeuvre):

**Step A — Resample to 4,000 Hz**
- `scipy.signal.resample_poly(signal, up=RESAMPLE_UP, down=RESAMPLE_DOWN)` (8/25)
- Rationale: polyphase resampling avoids the spectral leakage of FFT-based methods

**Step B — 8th-order Butterworth bandpass, 70–1900 Hz**
- Design: `scipy.signal.butter(BUTTERWORTH_ORDER, [BANDPASS_LOW, BANDPASS_HIGH], btype='bandpass', fs=FS_TARGET, output='sos')`
- Apply: `scipy.signal.sosfiltfilt(sos, signal)` — zero-phase, SOS form avoids numerical instability

**Step C — Comb notch filter at 50 Hz and harmonics**
- Iterate: `for f in range(50, FS_TARGET // 2, 50)` → harmonics 50, 100, ..., 1950 Hz (39 notches)
- Per harmonic: `b, a = scipy.signal.iirnotch(f, Q=f / NOTCH_BANDWIDTH, fs=FS_TARGET)`
- Apply: `scipy.signal.filtfilt(b, a, signal)`

### 1.3 Tests — `tests/test_phase1_io.py` and `tests/test_phase1_preprocessing.py`

**`test_phase1_io.py`**

```
test_read_signals_returns_correct_keys
    → output dict contains: signals, nchannels, nblocks, samplerate, titles

test_read_signals_structure_shape
    → len(signals) == 2 (nchannels), len(signals[0]) == 6 (nblocks)

test_read_signals_signal_lengths_nonzero
    → every signals[ch][block] has len > 0

test_read_signals_samplerate_value
    → all samplerate entries equal 12500

test_read_signals_invalid_path_raises
    → non-existent path raises FileNotFoundError

test_read_signals_matlab_index_offset
    → synthetic .mat with datastart=1 (MATLAB 1-based) → Python slice starts at index 0
```

**`test_phase1_preprocessing.py`**

```
test_resample_output_length
    → input N samples at 12500 Hz → output length ≈ N * (4000 / 12500) within ±2 samples

test_resample_preserves_frequency_content
    → resample a 200 Hz sine → peak frequency in Welch PSD of output is still 200 Hz ±5 Hz

test_bandpass_attenuates_below_cutoff
    → input 30 Hz sine → output RMS < 1% of input RMS

test_bandpass_attenuates_above_cutoff
    → input 1950 Hz sine → output RMS < 1% of input RMS

test_bandpass_passes_in_band
    → input 500 Hz sine → output RMS > 90% of input RMS

test_notch_attenuates_50hz
    → input: sine_50hz fixture → output RMS after full comb filter < 1% of input RMS

test_notch_attenuates_150hz_harmonic
    → input: pure 150 Hz sine → output RMS after full comb filter < 1% of input RMS

test_notch_preserves_off_harmonic_frequency
    → input: pure 175 Hz sine → output RMS after full comb filter > 90% of input RMS

test_full_pipeline_no_nan_no_inf
    → apply complete pipeline to white_noise fixture → np.isfinite(output).all() is True

test_full_pipeline_output_length_ratio
    → len(output) / len(input) is within 1% of 4000 / 12500
```

### 1.4 Gate to Phase 2

`pytest tests/test_phase1_io.py tests/test_phase1_preprocessing.py -v` — **all 10 tests must pass**.  
Visual check: plot raw vs filtered signal in `notebooks/02_preprocessing_check.ipynb` for one
patient and one control before handing off to Person 2.

---

## Phase 2 — Segmentation and dataset construction

**Owner**: Person 2  
**Input**: preprocessed signals + temporal marker files  
**Output**: 14,900 segments + 4 metadata vectors

### 2.1 Load temporal markers — `src/phase2_segmentation.py`

- Load with `scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)`
- Unwrap MATLAB cell array to a Python list of 6 numpy arrays (one per manoeuvre)
- Each array shape: (n_cycles, 4) — columns [insp_start, insp_end, exp_start, exp_end] in seconds
- Convert to sample indices: `idx = int(round(time_s * FS_TARGET))`
- Clamp to `[0, len(signal) - 1]` to guard against floating-point rounding at boundaries

### 2.2 Segment each signal

For each subject → manoeuvre (0–5) → channel (0–1) → respiratory cycle:
- Inspiration: `signal[insp_start_idx : insp_end_idx]`
- Expiration: `signal[exp_start_idx : exp_end_idx]`
- Skip any segment shorter than `MIN_SEGMENT_SAMPLES = int(0.05 * FS_TARGET)` (50 ms)
- Append each valid segment to the accumulator

### 2.3 Build the dataset — `src/phase2_dataset.py`

Accumulate into:
- `all_signals`: `list[np.ndarray]` — variable-length 1D arrays
- `v_subject`: `np.ndarray(14900,)` dtype `np.int32` — 1–23 patients, 24–28 controls
- `v_bd`: `np.ndarray(14900,)` dtype `np.int32` — 1 if manoeuvre_idx < 3 else 2
- `v_channel`: `np.ndarray(14900,)` dtype `np.int32` — 1 = lower, 2 = upper
- `v_phase`: `np.ndarray(14900,)` dtype `np.int32` — 1 = inspiration, 2 = expiration

Save:
- `numpy.savez("outputs/results/dataset.npz", v_subject=..., v_bd=..., v_channel=..., v_phase=...)`
- `pickle.dump(all_signals, open("outputs/results/all_signals.pkl", "wb"))`

### 2.4 Tests — `tests/test_phase2_segmentation.py` and `tests/test_phase2_dataset.py`

**`test_phase2_segmentation.py`**

```
test_load_markers_returns_6_manoeuvres
    → output list has exactly 6 elements

test_load_markers_each_element_has_4_columns
    → each element has shape (n_cycles, 4)

test_segment_length_matches_time_window
    → segment length == round((end_time - start_time) * FS_TARGET) within ±1 sample

test_inspiration_end_leq_expiration_start
    → insp_end_idx <= exp_start_idx for every cycle (no overlap)

test_segment_skips_too_short_cycles
    → synthetic cycle with duration < 50 ms produces zero segments

test_time_to_sample_boundary_values
    → t=0.0 s → idx=0; t=1.0 s → idx=4000; t=0.00025 s → idx=1
```

**`test_phase2_dataset.py`**

```
test_total_segment_count
    → len(all_signals) == 14900

test_all_metadata_vectors_have_correct_length
    → all four vectors have shape (14900,)

test_v_subject_valid_range
    → np.all((v_subject >= 1) & (v_subject <= 28))

test_v_bd_only_valid_values
    → set(v_bd.tolist()) == {1, 2}

test_v_channel_only_valid_values
    → set(v_channel.tolist()) == {1, 2}

test_v_phase_only_valid_values
    → set(v_phase.tolist()) == {1, 2}

test_pre_post_bd_roughly_balanced
    → abs(np.sum(v_bd == 1) - np.sum(v_bd == 2)) / 14900 < 0.10

test_no_empty_segments
    → all(len(s) > 0 for s in all_signals)
```

### 2.5 Gate to Phase 3

`pytest tests/test_phase2_segmentation.py tests/test_phase2_dataset.py -v` — **all 14 tests must pass**.  
Print a per-subject breakdown table and assert total is exactly **14,900** before proceeding.

---

## Phase 3 — Feature extraction

**Owners**: Person 3 (temporal + spectral) + Person 4 (CAS-specific)  
**Input**: `all_signals` list  
**Output**: feature matrix (14900, N_features) + `FEATURE_NAMES` list

### 3.1 Temporal features — `src/phase3_features.py` (Person 3)

Per segment:
- `rms`: `np.sqrt(np.mean(signal ** 2))`
- `zero_crossing_rate`: `np.sum(np.diff(np.sign(signal)) != 0) / len(signal)`
- `kurtosis`: `scipy.stats.kurtosis(signal)`
- `skewness`: `scipy.stats.skew(signal)`
- `duration_s`: `len(signal) / FS_TARGET`
- `peak_to_peak`: `np.ptp(signal)`

### 3.2 Spectral features — `src/phase3_features.py` (Person 3)

`freqs, psd = scipy.signal.welch(signal, fs=FS_TARGET, nperseg=min(512, len(signal)))`

Derive:
- `mean_freq`: `np.sum(freqs * psd) / np.sum(psd)`
- `median_freq`: frequency where cumulative power first reaches 50% of total
- `peak_freq`: `freqs[np.argmax(psd)]`
- `spectral_entropy`: `-np.sum(psd_norm * np.log2(psd_norm + 1e-12))`
- `band_power_70_200`, `band_power_200_500`, `band_power_500_1000`, `band_power_1000_1900`: each as fraction of total power

### 3.3 CAS-specific features — `src/phase3_features.py` (Person 4)

- `spectral_flatness`: `scipy.stats.gmean(psd + 1e-12) / np.mean(psd)` — low → tonal
- `n_spectral_peaks_100_1000`: count peaks in PSD in [100, 1000] Hz via `scipy.signal.find_peaks` with prominence threshold
- `peak_sharpness`: power at dominant peak / power in ±50 Hz neighbourhood
- `autocorr_peak`: max of normalised autocorrelation at lags 1 to `len(signal) // 2`
- `harmonic_ratio`: power at f0 / power at 2×f0 (when 2×f0 < Nyquist)

### 3.4 Build and save feature matrix

- Stack all feature vectors: `X = np.vstack([extract_all_features(s) for s in all_signals])`
- Replace NaN and Inf with **column median** — never drop rows
- `FEATURE_NAMES: list[str]` must be defined at module level
- Save: `numpy.savez("outputs/results/features.npz", X=X, feature_names=np.array(FEATURE_NAMES))`

### 3.5 Tests — `tests/test_phase3_features.py`

```
test_temporal_features_return_correct_shape
    → extract_temporal_features(signal) returns 1D array of length 6

test_rms_of_constant_signal
    → rms(np.ones(1000)) == 1.0

test_rms_of_sine_signal
    → rms(A * sin(2πft)) ≈ A / sqrt(2) within 1%

test_zcr_all_positive_signal_is_zero
    → zero_crossing_rate(np.ones(100)) == 0.0

test_zcr_alternating_signal_is_maximum
    → zero_crossing_rate(np.tile([1, -1], 50)) ≈ 1.0 within 5%

test_spectral_features_return_correct_shape
    → extract_spectral_features(signal) returns 1D array of expected length

test_mean_freq_of_pure_sine
    → mean_freq(sine_500hz) ≈ 500 Hz within ±10 Hz

test_band_power_fractions_sum_to_one
    → sum of four band power fractions ≈ 1.0 within ±0.01

test_spectral_flatness_white_noise_near_one
    → spectral_flatness(white_noise) > 0.8

test_spectral_flatness_pure_sine_near_zero
    → spectral_flatness(sine_100hz) < 0.1

test_n_spectral_peaks_for_single_sine
    → n_spectral_peaks_100_1000(sine_100hz) == 1

test_feature_matrix_contains_no_nan
    → build matrix on 100 synthetic signals → np.isnan(X).sum() == 0

test_feature_matrix_shape
    → X.shape == (100, len(FEATURE_NAMES))
```

### 3.6 Gate to Phase 4

`pytest tests/test_phase3_features.py -v` — **all 13 tests must pass**.  
In `notebooks/04_features_exploration.ipynb`, confirm that `spectral_flatness` and
`n_spectral_peaks_100_1000` visually separate tonal from noisy signals.

---

## Phase 4 — Classification: normal RS vs CAS

**Owners**: Person 3 (model training) + Person 4 (validation)  
**Input**: feature matrix + metadata vectors  
**Output**: CAS labels for all 14,900 segments + LOSO performance metrics

### 4.1 Ground truth labels

No expert annotations are provided. Use rule-based labelling to bootstrap ground truth:

A segment is labelled **CAS = 1** if ALL hold:
- `spectral_flatness < 0.15`
- `n_spectral_peaks_100_1000 >= 1`
- `duration_s > 0.1`

All other segments: **CAS = 0**.  
Tune thresholds by visually inspecting a stratified random sample of ≥50 segments per class.  
If expert annotations become available, replace rule-based labels immediately.

### 4.2 Model training — `src/phase4_classifier.py` (Person 3)

Train and compare via `sklearn.pipeline.Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

PIPELINES: dict[str, Pipeline] = {
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)),
    ]),
    "rf": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
    ]),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7)),
    ]),
}
```

### 4.3 Validation — `src/phase4_validation.py` (Person 4)

**Leave-One-Subject-Out (LOSO)** cross-validation — mandatory:
- `cv = LeaveOneGroupOut()`
- `groups = v_subject`
- 28 folds total (one per subject)
- No subject may appear in both train and test sets in the same fold

Report per fold and as mean ± std:
- Accuracy, sensitivity, specificity, precision, F1-score
- AUC-ROC: `sklearn.metrics.roc_auc_score`
- Aggregate confusion matrix

Save: `outputs/results/classifier_results.csv`

### 4.4 Tests — `tests/test_phase4_classifier.py`

```
test_pipeline_fits_without_error
    → pipeline.fit(X_synthetic, y_synthetic) completes with no exception

test_pipeline_predict_output_shape
    → pipeline.predict(X).shape == (n_samples,)

test_pipeline_predict_proba_in_range
    → all values in predict_proba(X) are in [0.0, 1.0]

test_pipeline_predict_binary_output
    → set(pipeline.predict(X).tolist()) ⊆ {0, 1}

test_loso_produces_correct_number_of_folds
    → LeaveOneGroupOut with 28 subjects → exactly 28 folds

test_loso_no_subject_leakage_across_folds
    → in every fold, set(train_subjects) ∩ set(test_subjects) == ∅

test_rule_based_labels_cas_rate_higher_in_patients
    → mean CAS rate in patients (subjects 1–23) > mean CAS rate in controls (24–28)

test_results_csv_has_required_columns
    → classifier_results.csv contains: fold, subject_id, accuracy, sensitivity,
      specificity, precision, f1, auc
```

### 4.5 Gate to Phase 5

`pytest tests/test_phase4_classifier.py -v` — **all 8 tests must pass**.  
Mean AUC across LOSO folds must be > 0.70 before proceeding. If not, revisit feature
engineering or the labelling thresholds in section 4.1.

---

## Phase 5 — Biomarker analysis

**Owner**: Person 5  
**Input**: CAS labels + metadata vectors + `database/subject_metadata.csv`  
**Output**: statistical results + publication-quality figures + written conclusions

### 5.1 CAS rate per subject and condition — `src/phase5_biomarker.py`

For each subject compute:
- `cas_rate_pre`: `n_CAS_pre / n_total_pre` — segments with `v_bd == 1`
- `cas_rate_post`: `n_CAS_post / n_total_post` — segments with `v_bd == 2`
- `delta_cas`: `cas_rate_pre - cas_rate_post` (positive = improvement after bronchodilator)

Stratify by: BDR group, channel (lower / upper), phase (inspiration / expiration).

### 5.2 Statistical analysis

For each comparison:
1. Normality: `scipy.stats.shapiro` — threshold p < 0.05
2. Pre vs post within group:
   - Normal → `scipy.stats.ttest_rel`
   - Non-normal → `scipy.stats.wilcoxon`
3. BDR+ vs BDR− between groups: `scipy.stats.mannwhitneyu`
4. Effect size: Cohen's d (parametric) or rank-biserial correlation (non-parametric)
5. Apply Bonferroni correction when multiple comparisons are performed
6. Significance threshold: α = 0.05

### 5.3 Visualisations

Save all figures to `outputs/figures/` at 300 dpi.

- `boxplot_cas_pre_post.png`: CAS rate pre vs post, BDR+ and BDR− side by side
- `roc_curve_biomarker.png`: ROC using `delta_cas` as biomarker score for BDR+
- `barplot_cas_per_subject.png`: mean CAS rate per subject, colour-coded by BDR label
- `heatmap_cas_subject_manoeuvre.png`: CAS rate matrix, subjects × manoeuvres

### 5.4 Tests — `tests/test_phase5_biomarker.py`

```
test_cas_rate_values_in_valid_range
    → all cas_rate values are in [0.0, 1.0]

test_cas_rate_vectors_have_correct_length
    → cas_rate_pre and cas_rate_post have length == n_subjects (28)

test_delta_cas_equals_pre_minus_post
    → delta_cas[i] == cas_rate_pre[i] - cas_rate_post[i] for all i

test_shapiro_pvalue_in_unit_interval
    → shapiro_wilk(group).pvalue is in [0.0, 1.0]

test_wilcoxon_identical_arrays_pvalue_is_one
    → wilcoxon(x, x).pvalue == 1.0

test_cohen_d_identical_groups_is_zero
    → cohen_d(x, x) == 0.0

test_cohen_d_known_value
    → cohen_d(np.zeros(30), np.ones(30)) ≈ -1.0 within ±0.05

test_output_figures_are_created
    → after generate_all_figures(), all four expected .png files exist in outputs/figures/
```

### 5.5 Gate: project complete

`pytest tests/ -v --cov=src --cov-report=term-missing` — **all tests across all phases must pass**
and total coverage must be > 80%.

### 5.6 Conclusions

Answer the following questions in the final report:
1. Does `delta_cas` decrease significantly after bronchodilator in BDR+ patients (p < 0.05)?
2. Is `delta_cas` significantly different between BDR+ and BDR− patients?
3. What is the AUC of `delta_cas` as a biomarker for BDR? Compare to the FEV1 > 12% standard criterion.
4. Are results consistent across both channels and both respiratory phases?

---

## Dependencies — `requirements.txt`

```
numpy>=1.26
scipy>=1.12
matplotlib>=3.8
seaborn>=0.13
pandas>=2.1
scikit-learn>=1.4
imbalanced-learn>=0.12
jupyter>=1.0
pytest>=8.0
pytest-cov>=5.0
ruff>=0.4
```

---

## `pytest.ini`

```ini
[pytest]
testpaths = tests
addopts = -v --tb=short
```

---

## Key constraints and notes for Claude Code

- All signal processing uses `FS_TARGET = 4000` Hz after resampling (original: `FS_ORIGINAL = 12500` Hz)
- MATLAB indices are 1-based: subtract 1 from `datastart` and `dataend` values before slicing in Python
- `.mat` files exported from LabChart may contain nested object arrays — unwrap with `.item()` or `.flat[0]`
- Total expected segments: **14,900** — assert this explicitly at the end of Phase 2
- LOSO cross-validation is mandatory — never use `train_test_split` with random shuffling across subjects
- Feature extraction must handle variable-length segments — guard Welch's `nperseg` with `min(512, len(signal))`
- Comb notch filter: `range(50, FS_TARGET // 2, 50)` → 39 harmonics (50, 100, ..., 1950 Hz)
- Butterworth filter: always use `output='sos'` + `sosfiltfilt` to avoid numerical instability at order 8
- Define all numeric constants (`FS_TARGET`, `BUTTERWORTH_ORDER`, etc.) at module level — no magic numbers inside function bodies
- All test fixtures live in `conftest.py` — no test file may load real `.mat` files
- Run `pytest --cov=src --cov-report=term-missing` before closing each phase; coverage must be > 80%
- Use `ruff check src/ tests/` to lint before every commit

---

## Checkpoint summary

| Phase | Owner | Tests to pass | Key output | Gate |
|-------|-------|--------------|-----------|------|
| 1 — IO + preprocessing | Person 1 | `test_phase1_*.py` — 10 tests | Preprocessed signals dict | All green + visual check |
| 2 — Segmentation + dataset | Person 2 | `test_phase2_*.py` — 14 tests | `dataset.npz`, 4 vectors | `len == 14900` asserted |
| 3 — Feature extraction | P3 + P4 | `test_phase3_features.py` — 13 tests | `features.npz` (14900 × N) | No NaN, distributions inspected |
| 4 — Classification | P3 + P4 | `test_phase4_classifier.py` — 8 tests | LOSO metrics + CAS labels | Mean AUC > 0.70 |
| 5 — Biomarker analysis | Person 5 | `test_phase5_biomarker.py` — 8 tests | Stats + 4 figures | All figures saved, p-values reported |
| **Final** | All | **All 53 tests + coverage > 80%** | Complete pipeline | `pytest --cov` green |
