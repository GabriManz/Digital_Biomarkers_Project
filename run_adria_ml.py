"""Runner: Adria ML-only classification (no TF/DL required)."""
import sys, os, pickle, warnings, time, csv
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import welch
import librosa
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVC
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics        import (accuracy_score, precision_score, recall_score,
                                    f1_score, roc_auc_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

FS       = 4000
N_MFCC   = 20
N_SPLITS = 5
SEED     = 42

# ── Feature extraction (exact copy from Adria/classification.py) ─────────────

def _safe(seg, mn=2048):
    if len(seg) < mn:
        seg = np.pad(seg, (0, mn - len(seg)))
    return seg.astype(np.float64)

def feat_temporal(s):
    s   = _safe(s)
    rms = np.sqrt(np.mean(s**2))
    d1  = np.diff(s); d2 = np.diff(d1); v = np.var(s)
    hm  = np.sqrt(np.var(d1) / (v + 1e-12))
    prob = s**2 / (np.sum(s**2) + 1e-12)
    return [np.mean(s), np.std(s), v, rms,
            np.max(np.abs(s)), np.max(s) - np.min(s),
            float(pd.Series(s).skew()), float(pd.Series(s).kurt()),
            np.sum(np.abs(np.diff(np.sign(s))) > 0) / len(s),
            np.max(np.abs(s)) / (rms + 1e-12),
            -np.sum(prob * np.log(prob + 1e-12)),
            np.sum(s**2), np.log(np.sum(s**2) + 1e-12),
            v, hm,
            np.sqrt(np.var(d2) / (np.var(d1) + 1e-12)) / (hm + 1e-12)]

def feat_spectral(s):
    s  = _safe(s)
    f, p = welch(s, fs=FS, nperseg=min(512, len(s)))
    tp = np.sum(p) + 1e-12; pn = p / tp; sc = np.sum(f * pn)
    feats = [sc, np.sqrt(np.sum(((f - sc)**2) * pn)),
             f[np.searchsorted(np.cumsum(pn), 0.85)],
             np.exp(np.mean(np.log(p + 1e-12))) / (np.mean(p) + 1e-12),
             -np.sum(pn * np.log(pn + 1e-12)), f[np.argmax(p)], sc,
             f[np.searchsorted(np.cumsum(pn), 0.50)]]
    for lo, hi in [(70,250),(250,500),(500,1000),(1000,1500),(1500,1900)]:
        feats.append(np.sum(p[(f>=lo)&(f<hi)]) / tp)
    return feats

def feat_mfcc(s):
    s  = _safe(s, 2048)
    m  = librosa.feature.mfcc(y=s, sr=FS, n_mfcc=N_MFCC)
    nf = m.shape[1]
    w  = min(9, nf if nf % 2 == 1 else max(nf-1, 1)); w = max(w, 3)
    mo = 'interp' if nf >= w else 'nearest'
    d  = librosa.feature.delta(m, width=w, mode=mo)
    d2 = librosa.feature.delta(m, width=w, mode=mo, order=2)
    return (list(np.mean(m, 1)) + list(np.std(m, 1)) +
            list(np.mean(d, 1)) + list(np.std(d, 1)) +
            list(np.mean(d2, 1)) + list(np.std(d2, 1)))

def feat_wavelet(s):
    s = _safe(s)
    coeffs = pywt.wavedec(s, 'db4', level=5); out = []
    for c in coeffs[1:]:
        e = np.sum(c**2); prob = c**2 / (e + 1e-12)
        out.extend([e, -np.sum(prob * np.log(prob + 1e-12)), np.std(c)])
    return out

def extract_features(signals):
    print(f"  Extracting {len(signals)} segments...")
    X = []
    for i, s in enumerate(signals):
        s = np.asarray(s, np.float64)
        if len(s) == 0: s = np.zeros(64)
        X.append(feat_temporal(s) + feat_spectral(s) + feat_mfcc(s) + feat_wavelet(s))
        if (i+1) % 500 == 0:
            print(f"    {i+1}/{len(signals)}")
    return np.array(X, dtype=np.float32)

def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        'Accuracy':    round(accuracy_score(y_true, y_pred), 4),
        'Precision':   round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall':      round(recall_score(y_true, y_pred, zero_division=0), 4),
        'Specificity': round(tn / (tn + fp + 1e-12), 4),
        'F1':          round(f1_score(y_true, y_pred, zero_division=0), 4),
        'ROC-AUC':     round(roc_auc_score(y_true, y_prob), 4),
    }

def cv_ml(clf, X, y, groups, use_smote=True):
    unique_groups = len(np.unique(groups))
    if unique_groups < 2:
        gkf    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = list(gkf.split(X, y))
    else:
        n_splits = min(N_SPLITS, unique_groups)
        gkf      = GroupKFold(n_splits=n_splits)
        splits   = list(gkf.split(X, y, groups))
    oof_prob = np.zeros(len(y))
    oof_pred = np.zeros(len(y), dtype=int)
    for tr, te in splits:
        sc  = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te]); ytr = y[tr]
        if use_smote and ytr.sum() > 1 and (ytr==0).sum() > 1:
            sm  = SMOTE(random_state=SEED, k_neighbors=min(5, ytr.sum()-1))
            Xtr, ytr = sm.fit_resample(Xtr, ytr)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:,1]
        oof_prob[te] = prob
        oof_pred[te] = (prob >= 0.5).astype(int)
    return oof_prob, oof_pred

# ── Main ─────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  ADRIA PIPELINE  –  ML Classification")
print("=" * 65)

print("\n[1] Loading preprocessed signals...")
with open('Adria/preprocessed/signals.pkl', 'rb') as f:
    all_signals = pickle.load(f)

mat_data   = scipy.io.loadmat('proy_labels.mat')
labels_raw = mat_data['labels'].flatten()
meta_df    = pd.read_csv('Adria/preprocessed/metadata.csv')
g_raw      = meta_df['participant'].values

min_len    = min(len(all_signals), len(labels_raw))
labels_raw = labels_raw[:min_len]
g_raw      = g_raw[:min_len]

idx  = np.where((labels_raw == 2) | (labels_raw == 3))[0]
sigs = [all_signals[i] for i in idx]
y    = np.where(labels_raw[idx] == 2, 1, 0).astype(int)
g    = g_raw[idx].astype(int)

n_cas, n_nocas = np.sum(y==1), np.sum(y==0)
print(f"  Labeled segments: {len(y)}  CAS={n_cas}  NO-CAS={n_nocas}")
print(f"  Unique participants: {np.unique(g)}")

print("\n[2] Feature extraction (temporal + spectral + MFCC + wavelet)...")
t0 = time.time()
X  = extract_features(sigs)
print(f"  Feature matrix: {X.shape}  ({(time.time()-t0)/60:.1f} min)")

print("\n[3] ML Cross-validation (GroupKFold + SMOTE)...")
ml_models = {
    'LR':      LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=SEED),
    'SVM-Lin': SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=SEED),
    'SVM-RBF': SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', probability=True, random_state=SEED),
    'RF':      RandomForestClassifier(n_estimators=500, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=SEED),
    'GBM':     GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=SEED),
    'XGB':     XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, objective='binary:logistic', eval_metric='logloss',
                              n_jobs=-1, random_state=SEED),
}

results = {}
for name, clf in ml_models.items():
    print(f"  {name}...", end=' ', flush=True)
    t1   = time.time()
    prob, pred = cv_ml(clf, X, y, g)
    m    = compute_metrics(y, pred, prob)
    results[name] = m
    print(f"Acc={m['Accuracy']:.3f}  F1={m['F1']:.3f}  AUC={m['ROC-AUC']:.3f}  "
          f"Recall={m['Recall']:.3f}  Spec={m['Specificity']:.3f}  ({time.time()-t1:.0f}s)")

print("\n" + "=" * 65)
print("  FINAL RESULTS — Adria Pipeline (ML only, GroupKFold-5 + SMOTE)")
print("=" * 65)
hdr = f"{'Model':<10} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'Spec':>7} {'F1':>7} {'AUC':>8}"
print(hdr)
print("-" * 65)
for name, m in results.items():
    print(f"{name:<10} {m['Accuracy']:>9.4f} {m['Precision']:>10.4f} "
          f"{m['Recall']:>8.4f} {m['Specificity']:>7.4f} {m['F1']:>7.4f} {m['ROC-AUC']:>8.4f}")

# Save results
os.makedirs('Adria/results', exist_ok=True)
rows = [{'Model': n, **m} for n, m in results.items()]
pd.DataFrame(rows).to_csv('Adria/results/ml_metrics.csv', index=False)
print("\nSaved to Adria/results/ml_metrics.csv")
