# Digital Biomarkers Project: Bronchodilator Response from Respiratory Sounds

This project evaluates bronchodilator response (BDR) in asthma using respiratory sound recordings collected before and after bronchodilator administration.

The main hypothesis is that continuous adventitious sounds (CAS), such as wheezes and rhonchi, can be quantified as a digital biomarker of treatment response.

## What this project does

The codebase implements an end-to-end pipeline:

1. Read LabChart `.mat` respiratory recordings and reconstruct channel/block signals.
2. Preprocess each signal (resampling, bandpass filtering, and 50 Hz harmonic notch removal).
3. Segment recordings into inspiratory and expiratory epochs using temporal marker files.
4. Build a full subject-level dataset with metadata vectors (subject, pre/post BD, channel, phase).
5. Extract temporal, spectral, and CAS-oriented features from each segment.
6. Generate rule-based CAS labels and train classifiers (SVM, Random Forest, KNN).
7. Validate models with Leave-One-Subject-Out (LOSO) cross-validation.
8. Compute biomarker statistics and generate publication-ready figures.

## Dataset summary

- 23 asthma patients (`P1` to `P23`) and 5 healthy controls (`C1` to `C5`).
- Each subject includes 6 breathing maneuvers (3 pre-BD, 3 post-BD).
- Marker files (`tP*`, `tC*`) provide cycle boundaries in seconds.
- Typical workflow expects data under `Data/Datos/`.

Note: Raw data and generated outputs are excluded by `.gitignore`.

## Repository structure

- `src/`: pipeline modules by project phase.
- `tests/`: unit tests for each phase.
- `database/subject_metadata.csv`: subject-level labels and metadata.
- `Data/`: raw recordings (ignored by git).
- `outputs/`: generated results and figures (ignored by git).
- `ACTION_PLAN.md`: project roadmap and technical conventions.

## Installation

Requirements:

- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick usage example

```python
from pathlib import Path

from src.phase2_dataset import build_dataset
from src.phase3_features import build_feature_matrix, FEATURE_NAMES
from src.phase4_classifier import make_rule_based_labels, PIPELINES
from src.phase4_validation import run_loso

# 1) Build segmented dataset from raw files
result = build_dataset(data_dir=Path("Data/Datos"))

# 2) Extract features
X = build_feature_matrix(result["all_signals"])

# 3) Build weak labels for CAS detection
y = make_rule_based_labels(X, FEATURE_NAMES)

# 4) Validate classifier with LOSO (group by subject)
groups = result["v_subject"]
loso_results = run_loso(PIPELINES["svm"], X, y, groups)

print(f"Folds: {len(loso_results)}")
```

## Run tests

```bash
pytest
```

Optional coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

## Outputs

The biomarker analysis module can generate:

- CAS pre/post comparison boxplots.
- ROC curve using delta CAS as biomarker score.
- Subject-level CAS bar plots.
- Subject-condition CAS heatmaps.

Generated artifacts are typically saved in `outputs/results/` and `outputs/figures/`.

## Project context

- Course: Digital Biomarkers and AI in Healthcare.
- Program: MSc in Advanced Biomedical Technologies (UPC BarcelonaTech).

## License

No license file is currently included in this repository.
