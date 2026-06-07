import sys, os, pickle, warnings, time, json
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import welch
import librosa
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

FS       = 4000
N_MFCC   = 20
SEED     = 42

# ─────────────────────────────────────────────────────────────────────────────
# Copia exacta del extractor de Adria
# ─────────────────────────────────────────────────────────────────────────────

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
    X = []
    n = len(signals)
    for i, s in enumerate(signals):
        if (i+1) % 1000 == 0:
            print(f"    {i+1}/{n} ({100*(i+1)/n:.1f}%)")
        s = np.asarray(s, np.float64)
        if len(s) == 0: s = np.zeros(64)
        X.append(feat_temporal(s) + feat_spectral(s) + feat_mfcc(s) + feat_wavelet(s))
    return np.array(X, dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Ejecución
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Cargando señales preprocesadas de Adria...")
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

    print(f"Extrayendo features del conjunto etiquetado ({len(sigs)} segmentos)...")
    X_labeled = extract_features(sigs)

    print(f"Extrayendo features de todas las 14,900 señales...")
    X_all = extract_features(all_signals)

    # Escalador y SMOTE
    sc = StandardScaler()
    X_labeled_sc = sc.fit_transform(X_labeled)
    X_all_sc = sc.transform(X_all)

    print("Aplicando SMOTE y entrenando Random Forest (el mejor modelo ML de Adria)...")
    sm = SMOTE(random_state=SEED, k_neighbors=5)
    X_train_res, y_train_res = sm.fit_resample(X_labeled_sc, y)

    clf = RandomForestClassifier(n_estimators=500, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=SEED)
    clf.fit(X_train_res, y_train_res)

    print("Prediciendo sobre las 14,900 señales...")
    y_prob_all = clf.predict_proba(X_all_sc)[:, 1]
    y_pred_all = (y_prob_all >= 0.5).astype(int)

    os.makedirs('outputs/results/adria', exist_ok=True)
    np.savez(
        'outputs/results/adria/predictions_all.npz',
        best_model_name='Random Forest',
        y_prob_all=y_prob_all,
        y_pred_all=y_pred_all
    )
    print("Inferencia de Adria guardada en outputs/results/adria/predictions_all.npz")

if __name__ == "__main__":
    main()
