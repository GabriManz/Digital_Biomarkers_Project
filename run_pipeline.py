"""Full pipeline runner.

Loads the saved dataset (all_signals.pkl + dataset.npz), re-extracts features
with the updated 16-feature code, applies new CAS thresholds, runs LOSO, and
prints the evaluation report.  Intermediate results are cached so the expensive
feature-extraction and LOSO steps are skipped on re-runs.
"""

from __future__ import annotations

import copy
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
DB_PATH     = PROJECT_ROOT / "database" / "subject_metadata.csv"
CACHE_DIR   = RESULTS_DIR / "_pipeline_cache"


# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------

def load_dataset():
    print("Loading saved dataset ...", flush=True)
    with open(RESULTS_DIR / "all_signals.pkl", "rb") as fh:
        all_signals = pickle.load(fh)
    npz       = np.load(RESULTS_DIR / "dataset.npz")
    v_subject = npz["v_subject"]
    v_bd      = npz["v_bd"]
    print(f"  Segments: {len(all_signals):,}   Subjects: {len(np.unique(v_subject))}")
    return all_signals, v_subject, v_bd


# ---------------------------------------------------------------------------
# 2. Feature extraction (with cache)
# ---------------------------------------------------------------------------

def build_features(all_signals, v_subject):
    from src.phase3_features import (
        FEATURE_NAMES, LABELING_FEATURE_NAMES, N_FEATURES,
        build_feature_matrix, build_labeling_feature_matrix,
        normalize_amplitude_features_per_subject,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    x_cache  = CACHE_DIR / "X_16.npy"
    xl_cache = CACHE_DIR / "X_label_4.npy"

    if x_cache.exists() and xl_cache.exists():
        print("\nLoading cached feature matrices ...", flush=True)
        X       = np.load(x_cache)
        X_label = np.load(xl_cache)
        print(f"  ML matrix      : {X.shape}  (cache)")
        print(f"  Labeling matrix: {X_label.shape}  (cache)")
        return X, X_label, FEATURE_NAMES, LABELING_FEATURE_NAMES

    print(f"\nBuilding ML feature matrix ({N_FEATURES} features) ...", flush=True)
    t0 = time.time()
    X  = build_feature_matrix(all_signals)
    print(f"  ML matrix shape: {X.shape}  [{time.time()-t0:.1f}s]", flush=True)

    print("Normalising amplitude features per-subject ...", flush=True)
    X = normalize_amplitude_features_per_subject(X, FEATURE_NAMES, v_subject)

    print(f"\nBuilding labeling feature matrix ({len(LABELING_FEATURE_NAMES)} features) ...", flush=True)
    t0      = time.time()
    X_label = build_labeling_feature_matrix(all_signals)
    print(f"  Labeling matrix: {X_label.shape}  [{time.time()-t0:.1f}s]", flush=True)

    np.save(x_cache,  X)
    np.save(xl_cache, X_label)
    return X, X_label, FEATURE_NAMES, LABELING_FEATURE_NAMES


# ---------------------------------------------------------------------------
# 3. CAS labels
# ---------------------------------------------------------------------------

def make_labels(X_label, LABELING_FEATURE_NAMES):
    from src.phase4_classifier import (
        CAS_FLATNESS_THRESHOLD, CAS_MIN_PEAKS,
        CAS_MIN_DURATION_S, CAS_MIN_PEAK_PROMINENCE,
        make_rule_based_labels,
    )
    y    = make_rule_based_labels(X_label, LABELING_FEATURE_NAMES)
    rate = float(np.mean(y))
    print(
        f"\nCAS labeling: flatness<{CAS_FLATNESS_THRESHOLD}, "
        f"peaks>={CAS_MIN_PEAKS}, dur>{CAS_MIN_DURATION_S}s, "
        f"prom>={CAS_MIN_PEAK_PROMINENCE}"
    )
    print(f"  Global CAS positivity rate: {rate:.1%}  ({int(np.sum(y)):,}/{len(y):,})")
    return y


# ---------------------------------------------------------------------------
# 4. LOSO classification (with cache)
# ---------------------------------------------------------------------------

def run_classification(X, y, v_subject):
    from src.phase4_classifier import PIPELINES
    from src.phase4_validation import run_loso

    clf_cache = CACHE_DIR / "clf_results.json"
    if clf_cache.exists():
        print("\nLoading cached classifier results ...", flush=True)
        with open(clf_cache) as fh:
            return json.load(fh)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    n_folds = len(np.unique(v_subject))

    for name, pipeline in PIPELINES.items():
        print(f"\nLOSO - {name.upper()} ({n_folds} folds) ...", flush=True)
        t0          = time.time()
        fold_res    = run_loso(copy.deepcopy(pipeline), X, y, v_subject)
        elapsed     = time.time() - t0

        accs = [r["accuracy"]    for r in fold_res]
        sens = [r["sensitivity"] for r in fold_res]
        spec = [r["specificity"] for r in fold_res]
        prec = [r["precision"]   for r in fold_res]
        f1s  = [r["f1"]          for r in fold_res]
        aucs = [r["auc"]         for r in fold_res if not np.isnan(r["auc"])]

        results[name] = {
            "accuracy":    [float(np.mean(accs)), float(np.std(accs))],
            "sensitivity": [float(np.mean(sens)), float(np.std(sens))],
            "specificity": [float(np.mean(spec)), float(np.std(spec))],
            "precision":   [float(np.mean(prec)), float(np.std(prec))],
            "f1":          [float(np.mean(f1s)),  float(np.std(f1s))],
            "auc":         [float(np.mean(aucs)), float(np.std(aucs))],
        }
        auc_m, auc_s = results[name]["auc"]
        print(f"  AUC {auc_m:.4f} +/- {auc_s:.4f}  [{elapsed:.0f}s]", flush=True)

    with open(clf_cache, "w") as fh:
        json.dump(results, fh, indent=2)
    return results


# ---------------------------------------------------------------------------
# 5. Biomarker
# ---------------------------------------------------------------------------

def run_biomarker(y, v_subject, v_bd, db_path):
    from src.phase5_biomarker import compute_cas_rates
    import pandas as pd

    bdr_map = {}
    with open(db_path) as fh:
        for row in csv.DictReader(fh):
            bdr_map[int(row["subject_num"])] = row["bdr_label"]

    cas_df = compute_cas_rates(y, v_subject, v_bd)
    cas_df["bdr_label"] = cas_df["subject_id"].map(bdr_map)
    return cas_df, bdr_map


# ---------------------------------------------------------------------------
# 6. Report (ASCII only)
# ---------------------------------------------------------------------------

def print_report(X, X_label, y, v_subject, v_bd, clf_results, cas_df):
    from sklearn.metrics import roc_auc_score
    import scipy.stats

    sep  = "=" * 70
    line = "-" * 70

    print(f"\n{sep}")
    print("EVALUATION REPORT - POST-FIX PIPELINE")
    print(sep)

    # ------------------------------------------------------------------
    # 2.1 CAS detection
    # ------------------------------------------------------------------
    print("\n--- 2.1 CAS DETECTION ---")
    global_rate = float(np.mean(y))
    in_target   = 0.20 <= global_rate <= 0.50
    print(f"Global CAS positivity rate: {global_rate:.1%}  "
          f"({'OK within 20-50% target' if in_target else 'OUTSIDE 20-50% target'})")

    print(f"\n{'ID':>4}  {'Type':>5}  {'BDR':>4}  {'pre':>6}  {'post':>6}  {'delta':>7}  flag")
    print(f"{'--':>4}  {'----':>5}  {'---':>4}  {'---':>6}  {'----':>6}  {'-----':>7}")
    for _, row in cas_df.sort_values("subject_id").iterrows():
        sid   = int(row["subject_id"])
        stype = "P" if sid <= 23 else "C"
        bdr   = str(row["bdr_label"])
        pre   = row["cas_rate_pre"]
        post  = row["cas_rate_post"]
        delta = row["delta_cas"]
        flag  = ""
        if pre > 0.80 or post > 0.80:
            flag = "WARN rate>80%"
        elif pre < 0.10 or post < 0.10:
            flag = "WARN rate<10%"
        print(f"{sid:>4}  {stype:>5}  {bdr:>4}  {pre:>6.3f}  {post:>6.3f}  {delta:>+7.3f}  {flag}")

    # ------------------------------------------------------------------
    # 2.2 Feature matrix
    # ------------------------------------------------------------------
    print(f"\n--- 2.2 FEATURE MATRIX ---")
    print(f"ML feature matrix shape:  {X.shape}  "
          f"({'OK' if X.shape[1] == 16 else 'FAIL'} expected [N, 16])")
    print(f"Labeling matrix shape:    {X_label.shape}  "
          f"({'OK' if X_label.shape[1] == 4 else 'FAIL'} expected [N, 4])")

    # ------------------------------------------------------------------
    # 2.3 Classifier results
    # ------------------------------------------------------------------
    print(f"\n--- 2.3 CLASSIFIER RESULTS (LOSO, 28 folds) ---")
    print(f"  {'Model':>5}  {'Acc':>7}  {'Sens':>7}  {'Spec':>7}  "
          f"{'Prec':>7}  {'F1':>7}  {'AUC':>7}")
    print("  " + "-" * 55)
    for name, m in clf_results.items():
        auc_m = m["auc"][0]
        flag  = "  <-- HIGH" if auc_m > 0.95 else ("  <-- LOW" if auc_m < 0.55 else "")
        print(f"  {name.upper():>5}  "
              f"{m['accuracy'][0]:>7.4f}  "
              f"{m['sensitivity'][0]:>7.4f}  "
              f"{m['specificity'][0]:>7.4f}  "
              f"{m['precision'][0]:>7.4f}  "
              f"{m['f1'][0]:>7.4f}  "
              f"{auc_m:>7.4f}{flag}")

    print(f"\n  Std deviations across 28 folds:")
    for name, m in clf_results.items():
        print(f"  {name.upper():>5}  "
              f"acc+/-{m['accuracy'][1]:.4f}  "
              f"sens+/-{m['sensitivity'][1]:.4f}  "
              f"spec+/-{m['specificity'][1]:.4f}  "
              f"auc+/-{m['auc'][1]:.4f}")

    rf_auc = clf_results["rf"]["auc"][0]
    print(f"\n  RF AUC check: {rf_auc:.4f}  "
          f"{'WARN still > 0.95' if rf_auc > 0.95 else 'OK in plausible range'}")

    # ------------------------------------------------------------------
    # 2.4 delta_CAS biomarker
    # ------------------------------------------------------------------
    print(f"\n--- 2.4 delta_CAS BIOMARKER ---")
    delta_plus  = cas_df[cas_df["bdr_label"] == "BDR+"]["delta_cas"].dropna().values
    delta_minus = cas_df[cas_df["bdr_label"] == "BDR-"]["delta_cas"].dropna().values

    bdr_binary = cas_df["bdr_label"].map({"BDR+": 1, "BDR-": 0}).values
    scores     = cas_df["delta_cas"].values
    valid      = np.isfinite(scores) & np.isfinite(bdr_binary.astype(float))
    if len(np.unique(bdr_binary[valid])) > 1:
        roc_auc = roc_auc_score(bdr_binary[valid], scores[valid])
    else:
        roc_auc = float("nan")
    print(f"delta_CAS ROC AUC (BDR predictor): {roc_auc:.4f}  "
          f"(pre-fix was 0.51)")

    # Wilcoxon: all subjects
    pre_all  = cas_df["cas_rate_pre"].dropna().values
    post_all = cas_df["cas_rate_post"].dropna().values
    if np.allclose(pre_all, post_all):
        print("Wilcoxon pre vs post (all):   trivial (pre == post)")
    else:
        s, p = scipy.stats.wilcoxon(pre_all, post_all, zero_method="wilcox")
        print(f"Wilcoxon pre vs post (all):   stat={s:.2f}  p={p:.4f}  "
              f"({'sig' if p < 0.05 else 'ns'})")

    # Wilcoxon: BDR+
    pre_p  = cas_df[cas_df["bdr_label"] == "BDR+"]["cas_rate_pre"].dropna().values
    post_p = cas_df[cas_df["bdr_label"] == "BDR+"]["cas_rate_post"].dropna().values
    if len(pre_p) >= 3 and not np.allclose(pre_p, post_p):
        s, p = scipy.stats.wilcoxon(pre_p, post_p, zero_method="wilcox")
        print(f"Wilcoxon pre vs post (BDR+):  stat={s:.2f}  p={p:.4f}  "
              f"({'sig' if p < 0.05 else 'ns'})")

    # Wilcoxon: BDR-
    pre_m  = cas_df[cas_df["bdr_label"] == "BDR-"]["cas_rate_pre"].dropna().values
    post_m = cas_df[cas_df["bdr_label"] == "BDR-"]["cas_rate_post"].dropna().values
    if len(pre_m) >= 3 and not np.allclose(pre_m, post_m):
        s, p = scipy.stats.wilcoxon(pre_m, post_m, zero_method="wilcox")
        print(f"Wilcoxon pre vs post (BDR-):  stat={s:.2f}  p={p:.4f}  "
              f"({'sig' if p < 0.05 else 'ns'})")

    print(f"\ndelta_CAS  BDR+: mean={np.mean(delta_plus):+.4f}  "
          f"std={np.std(delta_plus):.4f}  n={len(delta_plus)}")
    print(f"delta_CAS  BDR-: mean={np.mean(delta_minus):+.4f}  "
          f"std={np.std(delta_minus):.4f}  n={len(delta_minus)}")

    # Mann-Whitney BDR+ vs BDR-
    if len(delta_plus) >= 3 and len(delta_minus) >= 3:
        s, p = scipy.stats.mannwhitneyu(delta_plus, delta_minus, alternative="two-sided")
        print(f"Mann-Whitney BDR+ vs BDR-:    U={s:.1f}  p={p:.4f}  "
              f"({'sig' if p < 0.05 else 'ns'})")

    # ------------------------------------------------------------------
    # 2.5 Sanity checks
    # ------------------------------------------------------------------
    print(f"\n--- 2.5 SANITY CHECKS ---")
    sorted_df = cas_df.sort_values("delta_cas", ascending=False)
    print("Top-5 highest delta_CAS (most CAS reduction post-BD):")
    for _, row in sorted_df.head(5).iterrows():
        print(f"  Subject {int(row['subject_id']):>2} ({row['bdr_label']})  "
              f"delta={row['delta_cas']:+.4f}  pre={row['cas_rate_pre']:.3f}  "
              f"post={row['cas_rate_post']:.3f}")
    print("Bottom-5 lowest delta_CAS (most CAS increase post-BD):")
    for _, row in sorted_df.tail(5).iterrows():
        print(f"  Subject {int(row['subject_id']):>2} ({row['bdr_label']})  "
              f"delta={row['delta_cas']:+.4f}  pre={row['cas_rate_pre']:.3f}  "
              f"post={row['cas_rate_post']:.3f}")

    # ------------------------------------------------------------------
    # 3. Interpretation
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("INTERPRETATION")
    print(sep)

    print("\nQ1: CAS positivity rate")
    if 0.20 <= global_rate <= 0.50:
        print(f"  OK. Rate is {global_rate:.1%}, within 20-50% target.")
        print(f"  Fix 2 thresholds (flatness<0.10, peaks>=2, dur>0.15s) are effective.")
    else:
        pct = global_rate * 100
        print(f"  OUTSIDE TARGET. Rate is {pct:.1f}%.")
        if global_rate > 0.50:
            print("  Still too permissive. Consider tightening CAS_MIN_PEAKS to 3 or")
            print("  CAS_FLATNESS_THRESHOLD to 0.07.")
        else:
            print("  Too restrictive. Consider loosening one threshold.")

    print("\nQ2: RF AUC")
    if rf_auc > 0.95:
        print(f"  WARN: RF AUC = {rf_auc:.4f} still > 0.95.")
        print("  Remaining features correlated with labeling criteria may still be driving")
        print("  easy classification. Inspect feature importances (see clf_results cache).")
        print("  Candidate: peak_sharpness and harmonic_ratio both measure tonal structure,")
        print("  which is also captured by spectral_flatness in the labeling rule.")
    elif rf_auc >= 0.60:
        print(f"  OK: RF AUC = {rf_auc:.4f} (pre-fix: 1.0000).")
        print("  Circular dependency broken. Results are no longer artifactual.")
        print("  AUC in 0.60-0.95 range is clinically plausible for CAS detection.")
    else:
        print(f"  LOW: RF AUC = {rf_auc:.4f}. CAS label quality may be too low")
        print("  to support reliable classification with these features.")

    print("\nQ3: delta_CAS as BDR biomarker")
    print(f"  ROC AUC: {roc_auc:.4f}  (pre-fix: 0.51)")
    direction_ok = (np.mean(delta_plus) > np.mean(delta_minus))
    print(f"  Clinical direction (BDR+ delta > BDR- delta): "
          f"{'CONSISTENT' if direction_ok else 'INVERTED'}")
    if roc_auc >= 0.65:
        print("  Clear improvement. delta_CAS shows discriminative signal after fix.")
    elif roc_auc >= 0.55:
        print("  Modest improvement. Detector partially saturated or CAS not strongly")
        print("  correlated with BDR at current thresholds. Further tuning may help.")
    else:
        print("  No improvement vs chance. Detector saturation may persist, or")
        print("  CAS is not BDR-discriminative at these threshold values.")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_signals, v_subject, v_bd       = load_dataset()
    X, X_label, FEAT, LAB_FEAT        = build_features(all_signals, v_subject)
    y                                  = make_labels(X_label, LAB_FEAT)
    clf_results                        = run_classification(X, y, v_subject)
    cas_df, bdr_map                    = run_biomarker(y, v_subject, v_bd, DB_PATH)
    print_report(X, X_label, y, v_subject, v_bd, clf_results, cas_df)
