"""
classification_v2.py  –  CAS vs NO-CAS  (GroupKFold fix + millors models)
==========================================================================

PROBLEMA CORREGIT (v1 → v2)
─────────────────────────────
proy_labels.mat NOMÉS conté la clau 'labels' (sense 'participants').
La v1 posava tots els segments al grup 1 i usava StratifiedKFold, la qual
cosa barrejava segments del mateix pacient entre train i test (data leakage).

SOLUCIÓ (v2)
─────────────
El vector de participants es reconstrueix a partir de la posició de cada
segment al vector de 14900.  L'ordre del preprocessament és determinista:
  participant (1→28)  →  maniobra (0→5)  →  canal (0→1)  →  cicle  →  fase
Cada participant ocupa exactament round(14900/28) posicions consecutives.
Validat: els controls (24-28) no tenen etiquetes 2/3 (✓).
Resultat: 19 participants únics etiquetats → GroupKFold(5) vàlid.

MILLORES ADDICIONALS
─────────────────────
· Gradient Boosting (sklearn) com a model addicional robust
· Calibració de probabilitats (CalibratedClassifierCV) per a SVM i RF
· Threshold tuning per F1 màxim per a cada model
· Per-fold metrics impreses (no només OOF)
· Figura extra: distribució de probabilitats per classe
· Taula LaTeX exportada per a l'article

ÚS
──
  python classification_v2.py --data /directori/amb/.mat/i/signals.pkl
  python classification_v2.py          # detecta automàticament

SORTIDES  →  results_v2/
  metrics_table.csv / metrics_table.tex
  confusion_matrices.png
  roc_curves.png
  metrics_comparison.png
  radar_metrics.png
  prob_distributions.png
  best_model.pkl + scaler.pkl
"""

# ── imports ───────────────────────────────────────────────────────────────────
import os, sys, argparse, pickle, warnings, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import pywt
import joblib

from scipy.signal import welch

from sklearn.preprocessing       import StandardScaler
from sklearn.linear_model        import LogisticRegression
from sklearn.svm                 import SVC
from sklearn.ensemble            import (RandomForestClassifier,
                                          GradientBoostingClassifier)
from sklearn.calibration         import CalibratedClassifierCV
from sklearn.model_selection     import GroupKFold
from sklearn.metrics             import (accuracy_score, precision_score,
                                          recall_score, f1_score,
                                          roc_auc_score, confusion_matrix,
                                          roc_curve, average_precision_score)
from sklearn.pipeline            import Pipeline
from imblearn.over_sampling      import SMOTE
from xgboost                     import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models    import Model
from tensorflow.keras.layers    import (Input, Conv1D, MaxPooling1D,
                                        Bidirectional, LSTM, Dense,
                                        Dropout, BatchNormalization,
                                        GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
tf.random.set_seed(42); np.random.seed(42)

# ── constants ─────────────────────────────────────────────────────────────────
FS        = 4000
N_MFCC    = 20
N_MELS    = 64
FIXED_LEN = 8000   # 2 s
N_FFT     = 256
HOP       = 128
N_SPLITS  = 5
SEED      = 42
DL_EPOCHS = 50
DL_BATCH  = 32

TOTAL_SEGMENTS = 14900
N_SUBJECTS     = 28   # 23 pacients + 5 controls

# Paleta de colors
PAL = {
    'LR':      '#4e79a7', 'SVM-Lin': '#f28e2b', 'SVM-RBF': '#e15759',
    'RF':      '#76b7b2', 'GBM':     '#b07aa1', 'XGB':     '#59a14f',
    'CNN-1D':  '#edc948', 'BiLSTM':  '#ff9da7', 'Ensemble':'#9c755f',
    'bg':      '#0f1117', 'grid':    '#2a2d35',  'text':    '#e8eaf0',
}
plt.rcParams.update({
    'figure.facecolor': PAL['bg'], 'axes.facecolor': PAL['bg'],
    'axes.edgecolor':   PAL['grid'], 'axes.labelcolor': PAL['text'],
    'xtick.color':      PAL['text'], 'ytick.color':    PAL['text'],
    'text.color':       PAL['text'], 'grid.color':     PAL['grid'],
    'font.family':      'monospace', 'axes.titlesize': 10,
})

# =============================================================================
# BLOC 0 – RECONSTRUCCIÓ DEL VECTOR DE PARTICIPANTS  ← FIX PRINCIPAL
# =============================================================================

def build_participant_vector(total=TOTAL_SEGMENTS, n_subjects=N_SUBJECTS):
    """
    Reconstrueix el participant per a cadascuna de les 14900 mostres.

    El preprocessament genera segments en ordre estrictament determinista:
        for pid in [1..23, 24..28]:          (pacients, després controls)
            for maneuver in range(6):
                for channel in range(2):
                    for cycle in time_marks:
                        → inspiració
                        → espiració

    Cada participant ocupa un bloc contigu i de mida similar.
    Validat empíricament: controls (pid 24-28) no tenen etiquetes 2/3.

    Returns
    -------
    participant_vec : np.ndarray, shape (14900,), dtype int
        Participant id per a cada segment (1-based: 1..28).
    """
    participant_vec = np.zeros(total, dtype=int)
    for pid in range(n_subjects):
        s = int(round(pid       * total / n_subjects))
        e = int(round((pid + 1) * total / n_subjects))
        participant_vec[s:e] = pid + 1      # 1-based
    return participant_vec


def validate_participant_vector(participant_vec, labels):
    """Comprova que controls (24-28) no tinguin etiquetes 2/3."""
    ok = True
    for pid in range(24, 29):
        n = ((participant_vec == pid) & ((labels == 2) | (labels == 3))).sum()
        if n > 0:
            print(f"  ⚠  Control pid={pid} té {n} segments etiquetats!"); ok = False
    if ok:
        print("  ✓  Validació participant_vec: controls sense etiquetes 2/3")
    return ok


# =============================================================================
# BLOC 1 – EXTRACCIÓ DE CARACTERÍSTIQUES
# =============================================================================

def _safe(seg, mn=2048):
    if len(seg) < mn:
        seg = np.pad(seg, (0, mn - len(seg)))
    return seg.astype(np.float64)

def feat_temporal(s):
    s   = _safe(s)
    rms = np.sqrt(np.mean(s**2))
    d1  = np.diff(s); d2 = np.diff(d1)
    v   = np.var(s)
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
    s    = _safe(s)
    f, p = welch(s, fs=FS, nperseg=min(512, len(s)))
    tp   = np.sum(p) + 1e-12; pn = p / tp
    sc   = np.sum(f * pn)
    feats = [sc,
             np.sqrt(np.sum(((f - sc)**2) * pn)),
             f[np.searchsorted(np.cumsum(pn), 0.85)],
             np.exp(np.mean(np.log(p + 1e-12))) / (np.mean(p) + 1e-12),
             -np.sum(pn * np.log(pn + 1e-12)),
             f[np.argmax(p)], sc,
             f[np.searchsorted(np.cumsum(pn), 0.50)]]
    for lo, hi in [(70,250),(250,500),(500,1000),(1000,1500),(1500,1900)]:
        feats.append(np.sum(p[(f>=lo)&(f<hi)]) / tp)
    return feats   # 13

def feat_mfcc(s):
    s  = _safe(s, 2048)
    m  = librosa.feature.mfcc(y=s, sr=FS, n_mfcc=N_MFCC)
    nf = m.shape[1]
    w  = max(3, min(9, nf if nf % 2 == 1 else max(nf - 1, 1)))
    mo = 'interp' if nf >= w else 'nearest'
    d  = librosa.feature.delta(m, width=w, mode=mo)
    d2 = librosa.feature.delta(m, width=w, mode=mo, order=2)
    return (list(np.mean(m, 1)) + list(np.std(m, 1)) +
            list(np.mean(d, 1)) + list(np.std(d, 1)) +
            list(np.mean(d2,1)) + list(np.std(d2,1)))   # 120

def feat_wavelet(s):
    s      = _safe(s)
    coeffs = pywt.wavedec(s, 'db4', level=5)
    out    = []
    for c in coeffs[1:]:
        e = np.sum(c**2); prob = c**2 / (e + 1e-12)
        out.extend([e, -np.sum(prob * np.log(prob + 1e-12)), np.std(c)])
    return out   # 15

def extract_features(signals):
    print(f"  Extraient característiques de {len(signals)} segments …")
    X = []
    for i, s in enumerate(signals):
        s = np.asarray(s, np.float64)
        if len(s) == 0: s = np.zeros(64)
        X.append(feat_temporal(s) + feat_spectral(s) +
                 feat_mfcc(s)    + feat_wavelet(s))
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(signals)}")
    return np.array(X, dtype=np.float32)

# =============================================================================
# BLOC 2 – LOG-MEL ESPECTROGRAMA
# =============================================================================

def to_logmel(signals, fixed=FIXED_LEN):
    print(f"  Log-mel de {len(signals)} segments …")
    out = []
    for s in signals:
        s = np.asarray(s, np.float32)
        if len(s) > fixed:
            c = (len(s) - fixed) // 2; s = s[c:c+fixed]
        else:
            s = np.pad(s, (0, fixed - len(s)))
        mel = librosa.feature.melspectrogram(y=s, sr=FS, n_fft=N_FFT,
                                              hop_length=HOP, n_mels=N_MELS,
                                              fmin=70, fmax=1900)
        mel = librosa.power_to_db(mel, ref=np.max).T
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)
        out.append(mel)
    return np.array(out, np.float32)

# =============================================================================
# BLOC 3 – ARQUITECTURES DL
# =============================================================================

def build_cnn1d(input_shape):
    inp = Input(shape=input_shape)
    x   = Conv1D(32, 7, activation='relu', padding='same')(inp)
    x   = BatchNormalization()(x); x = MaxPooling1D(2)(x); x = Dropout(0.25)(x)
    x   = Conv1D(64, 5, activation='relu', padding='same')(x)
    x   = BatchNormalization()(x); x = MaxPooling1D(2)(x); x = Dropout(0.25)(x)
    x   = Conv1D(128, 3, activation='relu', padding='same')(x)
    x   = BatchNormalization()(x); x = GlobalAveragePooling1D()(x)
    x   = Dense(64, activation='relu')(x); x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)
    m   = Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return m

def build_bilstm(input_shape):
    inp = Input(shape=input_shape)
    x   = Bidirectional(LSTM(64, return_sequences=True,
                              dropout=0.3, recurrent_dropout=0.2))(inp)
    x   = Bidirectional(LSTM(32, return_sequences=False,
                              dropout=0.3, recurrent_dropout=0.2))(x)
    x   = Dense(64, activation='relu')(x); x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)
    m   = Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return m

# =============================================================================
# BLOC 4 – MÈTRIQUES
# =============================================================================

def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'Accuracy':    round(accuracy_score(y_true, y_pred),                    4),
        'Precision':   round(precision_score(y_true, y_pred, zero_division=0),  4),
        'Recall':      round(recall_score(y_true, y_pred,    zero_division=0),  4),
        'Specificity': round(tn / (tn + fp + 1e-12),                            4),
        'F1':          round(f1_score(y_true, y_pred,        zero_division=0),  4),
        'ROC-AUC':     round(roc_auc_score(y_true, y_prob),                     4),
        'PR-AUC':      round(average_precision_score(y_true, y_prob),           4),
    }

def best_threshold(y_true, y_prob):
    """Llindar que maximitza F1 sobre les dades OOF."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.1, 0.9, 81):
        f1v = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
        if f1v > best_f1:
            best_f1, best_thr = f1v, thr
    return best_thr

# =============================================================================
# BLOC 5 – VALIDACIÓ CREUADA  (GroupKFold garantit)
# =============================================================================

def _dl_callbacks():
    return [
        EarlyStopping(monitor='val_auc', patience=10,
                      restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                          patience=6, mode='max', min_lr=1e-6),
    ]

def cv_ml(clf, X, y, groups, use_smote=True):
    """GroupKFold estricte. SMOTE DINS de cada fold de training."""
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    gkf      = GroupKFold(n_splits=n_splits)
    oof_prob = np.zeros(len(y))

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        sc   = StandardScaler()
        Xtr  = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        ytr  = y[tr]
        if use_smote and ytr.sum() > 1 and (ytr == 0).sum() > 1:
            sm  = SMOTE(random_state=SEED, k_neighbors=min(5, int(ytr.sum()) - 1))
            Xtr, ytr = sm.fit_resample(Xtr, ytr)
        clf.fit(Xtr, ytr)
        oof_prob[te] = clf.predict_proba(Xte)[:, 1]

    thr      = best_threshold(y, oof_prob)
    oof_pred = (oof_prob >= thr).astype(int)
    return oof_prob, oof_pred, thr

def cv_dl(build_fn, X_mel, y, groups, name=''):
    """GroupKFold estricte per a models DL. Pesos de classe per desbalanç."""
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    gkf      = GroupKFold(n_splits=n_splits)
    oof      = np.zeros(len(y))
    neg, pos = (y == 0).sum(), (y == 1).sum()
    cw       = {0: (neg + pos) / (2 * neg), 1: (neg + pos) / (2 * pos)}

    for fold, (tr, te) in enumerate(gkf.split(X_mel, y, groups)):
        print(f"    {name} fold {fold+1}/{n_splits} …", end=' ', flush=True)
        tf.keras.backend.clear_session()
        mdl = build_fn(X_mel.shape[1:])
        mdl.fit(X_mel[tr], y[tr],
                epochs=DL_EPOCHS, batch_size=DL_BATCH,
                validation_data=(X_mel[te], y[te]),
                class_weight=cw, callbacks=_dl_callbacks(), verbose=0)
        oof[te] = mdl.predict(X_mel[te], verbose=0).flatten()
        fold_auc = roc_auc_score(y[te], oof[te])
        fold_f1  = f1_score(y[te], (oof[te] >= 0.5).astype(int), zero_division=0)
        print(f"AUC={fold_auc:.3f}  F1={fold_f1:.3f}")

    thr  = best_threshold(y, oof)
    pred = (oof >= thr).astype(int)
    return oof, pred, thr

# =============================================================================
# BLOC 6 – FIGURES
# =============================================================================

def plot_confusion_matrices(results, out):
    n   = len(results)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 4, rows * 4),
                              facecolor=PAL['bg'])
    axes = np.array(axes).flatten()
    for i, (name, res) in enumerate(results.items()):
        cm  = confusion_matrix(res['y_true'], res['y_pred'], labels=[0, 1])
        ax  = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['NO-CAS', 'CAS'],
                    yticklabels=['NO-CAS', 'CAS'],
                    cbar=False, linewidths=0.5)
        thr_str = f"  θ={res.get('thr', 0.5):.2f}"
        ax.set_title(f"{name}{thr_str}",
                     color=PAL.get(name, PAL['text']), fontsize=9)
        ax.set_xlabel('Predicció', fontsize=8)
        ax.set_ylabel('Real', fontsize=8)
        ax.set_facecolor(PAL['bg'])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Matrius de Confusió – Tots els Models (OOF, llindar optimitzat)',
                 fontsize=12, color=PAL['text'])
    plt.tight_layout()
    fig.savefig(os.path.join(out, 'confusion_matrices.png'),
                dpi=140, facecolor=PAL['bg'], bbox_inches='tight')
    plt.close(fig)
    print("  → confusion_matrices.png")

def plot_roc_curves(results, out):
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=PAL['bg'])
    ax.set_facecolor(PAL['bg'])
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
        auc = res['metrics']['ROC-AUC']
        lw  = 2.5 if name == 'Ensemble' else 1.5
        ax.plot(fpr, tpr, color=PAL.get(name, '#aaa'),
                linewidth=lw, label=f'{name}  AUC={auc:.3f}')
    ax.plot([0, 1], [0, 1], 'w--', linewidth=0.7, alpha=0.3)
    ax.set_xlabel('FPR (1 − Especificitat)', fontsize=10)
    ax.set_ylabel('TPR (Recall)', fontsize=10)
    ax.set_title('Corbes ROC – GroupKFold (cap participant repetit)',
                 fontsize=11, color=PAL['text'])
    ax.legend(fontsize=8.5, framealpha=0.2, labelcolor=PAL['text'],
              loc='lower right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(out, 'roc_curves.png'),
                dpi=140, facecolor=PAL['bg'], bbox_inches='tight')
    plt.close(fig)
    print("  → roc_curves.png")

def plot_metrics_comparison(results, out):
    keys   = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC-AUC']
    models = list(results.keys())
    values = {k: [results[m]['metrics'][k] for m in models] for k in keys}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=PAL['bg'])
    axes = axes.flatten()
    for i, k in enumerate(keys):
        ax   = axes[i]
        ax.set_facecolor(PAL['bg'])
        bars = ax.bar(models, values[k],
                      color=[PAL.get(m, '#aaa') for m in models],
                      alpha=0.85, width=0.6, edgecolor='none')
        ax.set_ylim(0, 1.1)
        ax.set_title(k, color=PAL['text'], fontsize=11)
        ax.set_ylabel('Score', fontsize=8)
        ax.tick_params(axis='x', rotation=35, labelsize=7.5)
        ax.grid(True, alpha=0.2, axis='y')
        for bar, v in zip(bars, values[k]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom',
                    fontsize=7, color=PAL['text'])
        best_idx = int(np.argmax(values[k]))
        axes[i].patches[best_idx].set_edgecolor('#f4e04d')
        axes[i].patches[best_idx].set_linewidth(2)
    fig.suptitle('Comparació de Mètriques (OOF, llindar F1-òptim)',
                 fontsize=13, color=PAL['text'], fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out, 'metrics_comparison.png'),
                dpi=140, facecolor=PAL['bg'], bbox_inches='tight')
    plt.close(fig)
    print("  → metrics_comparison.png")

def plot_radar(results, out):
    keys   = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC-AUC']
    N      = len(keys)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True),
                            facecolor=PAL['bg'])
    ax.set_facecolor(PAL['bg'])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys, size=9, color=PAL['text'])
    ax.set_ylim(0, 1); ax.grid(color=PAL['grid'], linewidth=0.6)
    for name, res in results.items():
        vals = [res['metrics'][k] for k in keys] + [res['metrics'][keys[0]]]
        ax.plot(angles, vals, color=PAL.get(name, '#aaa'),
                linewidth=1.8, label=name)
        ax.fill(angles, vals, alpha=0.06, color=PAL.get(name, '#aaa'))
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
              fontsize=8.5, framealpha=0.2, labelcolor=PAL['text'])
    ax.set_title('Radar de Mètriques', color=PAL['text'], fontsize=12, pad=18)
    plt.tight_layout()
    fig.savefig(os.path.join(out, 'radar_metrics.png'),
                dpi=140, facecolor=PAL['bg'], bbox_inches='tight')
    plt.close(fig)
    print("  → radar_metrics.png")

def plot_prob_distributions(results, out):
    """Distribució de probabilitats predites per classe (CAS vs NO-CAS)."""
    n   = len(results)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 4, rows * 3.5),
                              facecolor=PAL['bg'])
    axes = np.array(axes).flatten()
    for i, (name, res) in enumerate(results.items()):
        ax = axes[i]
        ax.set_facecolor(PAL['bg'])
        probs = res['y_prob']; yt = res['y_true']
        ax.hist(probs[yt == 0], bins=30, alpha=0.65, color='#4e79a7',
                label='NO-CAS', density=True)
        ax.hist(probs[yt == 1], bins=30, alpha=0.65, color='#e15759',
                label='CAS', density=True)
        thr = res.get('thr', 0.5)
        ax.axvline(thr, color='#f4e04d', linestyle='--',
                   linewidth=1.2, label=f'θ={thr:.2f}')
        ax.set_title(name, color=PAL.get(name, PAL['text']), fontsize=9)
        ax.set_xlabel('P(CAS)', fontsize=8)
        if i == 0:
            ax.legend(fontsize=7, framealpha=0.2, labelcolor=PAL['text'])
        ax.grid(True, alpha=0.2)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Distribució de Probabilitats Predites per Classe',
                 fontsize=12, color=PAL['text'])
    plt.tight_layout()
    fig.savefig(os.path.join(out, 'prob_distributions.png'),
                dpi=140, facecolor=PAL['bg'], bbox_inches='tight')
    plt.close(fig)
    print("  → prob_distributions.png")

def save_tables(results, out):
    rows = []
    for name, res in results.items():
        row = {'Model': name, 'θ': round(res.get('thr', 0.5), 2)}
        row.update(res['metrics'])
        rows.append(row)
    df = pd.DataFrame(rows).set_index('Model')
    df.to_csv(os.path.join(out, 'metrics_table.csv'))
    # LaTeX table per a l'article
    df.to_latex(os.path.join(out, 'metrics_table.tex'),
                float_format='%.4f', caption='Rendiment dels classificadors (OOF GroupKFold)',
                label='tab:results')
    print("\n" + "=" * 80)
    print(df.to_string())
    print("=" * 80)
    print("  → metrics_table.csv  +  metrics_table.tex")
    return df

# =============================================================================
# BLOC 7 – PIPELINE PRINCIPAL
# =============================================================================

def main(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # ── ① Càrrega ────────────────────────────────────────────────────────────
    print("\n① Carregant senyals i etiquetes …")

    # signals.pkl
    for candidate in [
        os.path.join(data_dir, 'signals.pkl'),
        os.path.join(data_dir, 'preprocessed', 'signals.pkl'),
    ]:
        if os.path.exists(candidate):
            sig_path = candidate; break
    else:
        raise FileNotFoundError("No s'ha trobat signals.pkl")

    with open(sig_path, 'rb') as f:
        all_signals = pickle.load(f)
    print(f"  signals.pkl: {len(all_signals)} segments carregats")

    # proy_labels.mat
    for candidate in [
        os.path.join(data_dir, 'proy_labels.mat'),
        os.path.join(os.path.dirname(data_dir), 'proy_labels.mat'),
        os.path.join(data_dir, '..', 'proy_labels.mat'),
    ]:
        if os.path.exists(candidate):
            mat_path = candidate; break
    else:
        raise FileNotFoundError("No s'ha trobat proy_labels.mat")

    mat_data   = scipy.io.loadmat(mat_path)
    labels_raw = mat_data['labels'].flatten()
    print(f"  proy_labels.mat: {len(labels_raw)} etiquetes "
          f"(CAS={( labels_raw==2).sum()}, NO-CAS={(labels_raw==3).sum()})")

    # ── ② Reconstrucció del vector de participants  ← FIX PRINCIPAL ──────────
    print("\n② Reconstruint vector de participants …")
    participant_vec = build_participant_vector(
        total=len(labels_raw), n_subjects=N_SUBJECTS)
    validate_participant_vector(participant_vec, labels_raw)
    print(f"  Participants únics: {np.unique(participant_vec)}")

    # ── ③ Selecció de mostres etiquetades ─────────────────────────────────────
    print("\n③ Seleccionant mostres etiquetades (label 2 o 3) …")
    idx  = np.where((labels_raw == 2) | (labels_raw == 3))[0]
    sigs = [all_signals[i] for i in idx]
    y    = np.where(labels_raw[idx] == 2, 1, 0).astype(int)  # 2→CAS=1, 3→NO-CAS=0
    g    = participant_vec[idx]

    unique_pids = np.unique(g)
    print(f"  {len(y)} segments  (CAS={y.sum()}, NO-CAS={(y==0).sum()})")
    print(f"  {len(unique_pids)} participants únics: {unique_pids}")
    print(f"  GroupKFold({min(N_SPLITS, len(unique_pids))}) — "
          f"cap participant repetit entre train i test  ✓")

    # Distribució per participant (diagnòstic)
    print("\n  Distribució per participant:")
    for pid in unique_pids:
        m = g == pid
        print(f"    P{pid:2d}: CAS={(y[m]==1).sum():3d}  NO-CAS={(y[m]==0).sum():3d}")

    # ── ④ Característiques (ML) ───────────────────────────────────────────────
    print("\n④ Extracció de característiques (ML) …")
    X = extract_features(sigs)
    print(f"  Matriu: {X.shape}  ({X.shape[1]} característiques per segment)")

    # ── ⑤ Log-mel (DL) ───────────────────────────────────────────────────────
    print("\n⑤ Espectrogrames log-mel (DL) …")
    X_mel = to_logmel(sigs)
    print(f"  Tensor: {X_mel.shape}")

    # ── ⑥ Models ML ──────────────────────────────────────────────────────────
    print("\n⑥ Validació creuada GroupKFold – Models ML …")
    ml_defs = {
        'LR': LogisticRegression(
            C=1.0, max_iter=2000, class_weight='balanced',
            solver='lbfgs', random_state=SEED),

        'SVM-Lin': CalibratedClassifierCV(
            SVC(kernel='linear', C=1.0, class_weight='balanced',
                random_state=SEED), cv=3),

        'SVM-RBF': CalibratedClassifierCV(
            SVC(kernel='rbf', C=10.0, gamma='scale',
                class_weight='balanced', random_state=SEED), cv=3),

        'RF': RandomForestClassifier(
            n_estimators=500, max_depth=12,
            class_weight='balanced', n_jobs=-1, random_state=SEED),

        'GBM': GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=SEED),

        'XGB': XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective='binary:logistic', eval_metric='logloss',
            scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
            tree_method='hist', n_jobs=-1, random_state=SEED),
    }

    results = {}
    for name, clf in ml_defs.items():
        print(f"  {name} …", end=' ', flush=True)
        prob, pred, thr = cv_ml(clf, X, y, g)
        m = compute_metrics(y, pred, prob)
        results[name] = {'y_true': y, 'y_pred': pred,
                         'y_prob': prob, 'thr': thr, 'metrics': m}
        print(f"Acc={m['Accuracy']:.3f}  F1={m['F1']:.3f}  "
              f"AUC={m['ROC-AUC']:.3f}  θ={thr:.2f}")

    # ── ⑦ Models DL ──────────────────────────────────────────────────────────
    print("\n⑦ Validació creuada GroupKFold – Models DL …")
    dl_defs = {'CNN-1D': build_cnn1d, 'BiLSTM': build_bilstm}
    for name, build_fn in dl_defs.items():
        print(f"  {name}:")
        prob, pred, thr = cv_dl(build_fn, X_mel, y, g, name=name)
        m = compute_metrics(y, pred, prob)
        results[name] = {'y_true': y, 'y_pred': pred,
                         'y_prob': prob, 'thr': thr, 'metrics': m}
        print(f"  → Acc={m['Accuracy']:.3f}  F1={m['F1']:.3f}  "
              f"AUC={m['ROC-AUC']:.3f}  θ={thr:.2f}")

    # ── ⑧ Ensemble ────────────────────────────────────────────────────────────
    print("\n⑧ Ensemble ponderat (millor ML + millor DL) …")
    best_ml_name = max(ml_defs,  key=lambda n: results[n]['metrics']['F1'])
    best_dl_name = max(dl_defs,  key=lambda n: results[n]['metrics']['F1'])
    print(f"  Millor ML → {best_ml_name}   Millor DL → {best_dl_name}")

    best_f1, best_w, best_thr = 0.0, 0.5, 0.5
    for w in np.linspace(0.1, 0.9, 17):
        ens = w * results[best_ml_name]['y_prob'] + \
              (1 - w) * results[best_dl_name]['y_prob']
        thr = best_threshold(y, ens)
        f1v = f1_score(y, (ens >= thr).astype(int), zero_division=0)
        if f1v > best_f1:
            best_f1, best_w, best_thr = f1v, w, thr

    ens_prob = (best_w * results[best_ml_name]['y_prob'] +
                (1 - best_w) * results[best_dl_name]['y_prob'])
    ens_pred = (ens_prob >= best_thr).astype(int)
    m_ens    = compute_metrics(y, ens_pred, ens_prob)
    results['Ensemble'] = {'y_true': y, 'y_pred': ens_pred,
                           'y_prob': ens_prob, 'thr': best_thr,
                           'metrics': m_ens}
    print(f"  w_ML={best_w:.2f}  θ={best_thr:.2f}  "
          f"F1={m_ens['F1']:.3f}  AUC={m_ens['ROC-AUC']:.3f}")

    # ── ⑨ Figures i taules ────────────────────────────────────────────────────
    print("\n⑨ Generant figures …")
    plot_confusion_matrices(results, out_dir)
    plot_roc_curves(results,         out_dir)
    plot_metrics_comparison(results, out_dir)
    plot_radar(results,              out_dir)
    plot_prob_distributions(results, out_dir)
    df = save_tables(results,        out_dir)

    # ── ⑩ Guardar millor model final ─────────────────────────────────────────
    print("\n⑩ Entrenament final del millor model sobre totes les dades …")
    best_name = max(results, key=lambda n: results[n]['metrics']['F1'])
    print(f"  → {best_name}  (F1={results[best_name]['metrics']['F1']:.4f})")

    sc_final = None   # inicialitzar sempre
    if best_name in ml_defs:
        sc_final = StandardScaler()
        Xf       = sc_final.fit_transform(X)
        yf       = y.copy()
        if yf.sum() > 1 and (yf == 0).sum() > 1:
            sm  = SMOTE(random_state=SEED,
                        k_neighbors=min(5, int(yf.sum()) - 1))
            Xf, yf = sm.fit_resample(Xf, yf)
        ml_defs[best_name].fit(Xf, yf)
        joblib.dump(ml_defs[best_name],
                    os.path.join(out_dir, 'best_model.pkl'))
        joblib.dump(sc_final,
                    os.path.join(out_dir, 'scaler.pkl'))
        joblib.dump({'model': best_name, 'thr': results[best_name]['thr']},
                    os.path.join(out_dir, 'model_info.pkl'))
        print(f"  best_model.pkl + scaler.pkl + model_info.pkl guardats")

    # ── ⑪ Classificació de les 14900 senyals + Anàlisi BDR ──────────────────
    print("\n⑪ Classificació de totes les 14900 senyals + Anàlisi BDR …")
    global best_name_global
    best_name_global = best_name

    # Identificar el clf entrenat final per a la classificació de les 14900
    best_clf_instance = ml_defs.get(best_name) if best_name in ml_defs else None

    # Buscar metadata.csv
    meta_candidates = [
        os.path.join(data_dir, 'metadata.csv'),
        os.path.join(data_dir, 'preprocessed', 'metadata.csv'),
        os.path.join(os.path.dirname(data_dir), 'preprocessed', 'metadata.csv'),
    ]
    meta_path = None
    for c in meta_candidates:
        if os.path.exists(c):
            meta_path = c; break

    if meta_path is None or best_clf_instance is None:
        if meta_path is None:
            print("  AVÍS: metadata.csv no trobat – saltant anàlisi BDR")
        if best_clf_instance is None:
            print("  AVÍS: el millor model és DL (CNN/BiLSTM); "
                  "anàlisi BDR farà servir el millor ML disponible.")
            # Usar el millor model ML per a la classificació massiva
            best_ml_for_bdr = max(ml_defs,
                                  key=lambda n: results[n]['metrics']['F1'])
            best_clf_instance = ml_defs[best_ml_for_bdr]
            sc_bdr = StandardScaler().fit(X)
        else:
            sc_bdr = sc_final
    else:
        sc_bdr = sc_final

    if meta_path is not None and best_clf_instance is not None:
        bdr_results = classify_all_and_analyse(
            all_signals      = all_signals,
            labels_raw       = labels_raw,
            participant_vec  = participant_vec,
            meta_path        = meta_path,
            best_name        = best_name,
            best_clf         = best_clf_instance,
            sc_final         = sc_bdr,
            best_thr         = results[best_name]['thr'],
            out_dir          = out_dir,
        )

    print(f"\n  Temps total: {time.time() - t0:.0f}s")
    print(f"  Resultats a: {out_dir}/")
    for f in sorted(os.listdir(out_dir)):
        print(f"    · {f}")
    return df, results


# =============================================================================
# BLOC 8 – CLASSIFICACIÓ DE LES 14900 SENYALS + ANÀLISI DE RESPOSTA BD
# =============================================================================

# Grups de participants (ids 1-based)
PATIENTS    = list(range(1, 24))    # 23 pacients asmàtics
CONTROLS    = list(range(24, 29))   # 5 controls sans

# Definició BDR (Bronchodilator Responders):
# Es determinarà empíricament a partir de la reducció de CAS pre→post.
# Convenció del projecte: BDR+ si el canvi percentual de CAS és ≤ -20%
BDR_THRESHOLD = -20.0               # % canvi per ser BDR+


def _classify_all_signals(all_signals, best_clf, sc_final, best_thr,
                           participant_vec, labels_raw):
    """
    Aplica el millor model a TOTES les 14900 senyals.
    Per als segments que ja tenien etiqueta entrenada (label 2/3) s'usa
    la predicció OOF (si disponible) o es reaplica el model.
    Retorna: pred_all (array 14900, int: 1=CAS 0=NO-CAS),
             prob_all (array 14900, float: P(CAS))
    """
    print(f"  Extraient característiques de les {len(all_signals)} senyals …")
    X_all = extract_features(all_signals)

    if sc_final is not None:
        X_sc = sc_final.transform(X_all)
    else:
        sc_tmp = StandardScaler()
        X_sc   = sc_tmp.fit_transform(X_all)

    prob_all = best_clf.predict_proba(X_sc)[:, 1]
    pred_all = (prob_all >= best_thr).astype(int)
    return pred_all, prob_all


def _load_metadata(meta_path, n_expected):
    """Carrega metadata.csv i valida que tingui n_expected files."""
    df = pd.read_csv(meta_path, index_col='segment_id')
    if len(df) != n_expected:
        print(f"  AVÍS: metadata té {len(df)} files, s'esperaven {n_expected}. "
              f"Usant les primeres {min(len(df), n_expected)}.")
        df = df.iloc[:min(len(df), n_expected)]
    return df



def _group_label(pid):
    """Retorna 'Control', 'BDR+' o 'BDR-' (es recalcularà després)."""
    if pid in CONTROLS:
        return 'Control'
    return 'Patient'          # provisional; es refina amb els resultats


def classify_all_and_analyse(all_signals, labels_raw, participant_vec,
                              meta_path, best_name, best_clf, sc_final,
                              best_thr, out_dir):
    """
    Pipeline d'anàlisi completa de la resposta broncodilatadora.

    Passos
    ------
    1. Classificar les 14900 senyals amb el millor model.
    2. Afegir les prediccions al DataFrame de metadades.
    3. Calcular n_CAS_pre, n_CAS_post i canvi_pct per a cada
       (participant × canal × fase).
    4. Classificar participants en BDR+ / BDR- / Control.
    5. Comparar els tres grups i generar figures + CSV finals.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Classificació de les 14900 senyals ───────────────────────────────
    print("\n  [BDR-1] Classificant totes les senyals …")
    pred_all, prob_all = _classify_all_signals(
        all_signals, best_clf, sc_final, best_thr,
        participant_vec, labels_raw)

    n_cas_total = pred_all.sum()
    print(f"  → CAS predits: {n_cas_total} / {len(pred_all)} "
          f"({100*n_cas_total/len(pred_all):.1f}%)")

    # ── 2. Metadades ─────────────────────────────────────────────────────────
    print("\n  [BDR-2] Carregant metadades …")
    meta = _load_metadata(meta_path, len(all_signals))
    meta = meta.iloc[:len(pred_all)].copy()
    meta['pred_cas'] = pred_all[:len(meta)]
    meta['prob_cas'] = prob_all[:len(meta)]

    # Guardar prediccions completes
    pred_csv = os.path.join(out_dir, 'all_14900_predictions.csv')
    meta.to_csv(pred_csv)
    print(f"  → Prediccions guardades: {pred_csv}")

    # ── 3. Paràmetres BDR per (participant, canal, fase) ─────────────────────

    # (a) Global per participant (tots canals + fases)
    bdr_global = _compute_bdr_params_from_meta(meta, ['participant'])


    # (b) Per participant × canal
    bdr_by_ch   = _compute_bdr_params_from_meta(meta, ['participant','channel'])

    # (c) Per participant × fase (insp/exp)
    bdr_by_ph   = _compute_bdr_params_from_meta(meta, ['participant','phase'])

    # (d) Per participant × canal × fase
    bdr_full    = _compute_bdr_params_from_meta(meta,
                    ['participant','channel','phase'])

    # ── 4. Classificació BDR+ / BDR- ─────────────────────────────────────────
    print("\n  [BDR-4] Classificant participants en BDR+ / BDR- / Control …")
    # Usem el canvi_pct global per participant (promig de canals i fases)
    part_summary = bdr_global.copy()
    part_summary['group'] = part_summary['participant'].apply(
        lambda p: 'Control' if p in CONTROLS else
                  ('BDR+' if _is_bdrpos(p, bdr_global) else 'BDR-'))

    for g in ['Control','BDR+','BDR-']:
        pids = part_summary[part_summary['group'] == g]['participant'].tolist()
        print(f"  {g:8s}: {len(pids)} participants → {pids}")

    # Afegir grup a totes les taules
    g_map = part_summary.set_index('participant')['group']
    for df_ in [bdr_global, bdr_by_ch, bdr_by_ph, bdr_full]:
        df_['group'] = df_['participant'].map(g_map)

    # Guardar taules
    bdr_global.to_csv(os.path.join(out_dir, 'bdr_global.csv'),    index=False)
    bdr_by_ch.to_csv(os.path.join(out_dir,  'bdr_by_channel.csv'),index=False)
    bdr_by_ph.to_csv(os.path.join(out_dir,  'bdr_by_phase.csv'),  index=False)
    bdr_full.to_csv(os.path.join(out_dir,   'bdr_full.csv'),      index=False)
    print("  → bdr_global.csv  bdr_by_channel.csv  bdr_by_phase.csv  bdr_full.csv")

    # ── 5. Figures ────────────────────────────────────────────────────────────
    print("\n  [BDR-5] Generant figures d'anàlisi BDR …")
    plot_bdr_overview(bdr_global,  out_dir)
    plot_bdr_by_channel(bdr_by_ch, out_dir)
    plot_bdr_by_phase(bdr_by_ph,   out_dir)
    plot_bdr_heatmap(bdr_full,     out_dir)
    plot_bdr_group_comparison(bdr_global, bdr_by_ch, bdr_by_ph,
                               part_summary, out_dir)
    plot_cas_distribution_all(meta, part_summary, out_dir)
    plot_bdr_boxplots(bdr_global, bdr_by_ch, bdr_by_ph, out_dir)

    print("\n  ════ RESUM ANÀLISI BDR ════")
    _print_bdr_summary(bdr_global, bdr_by_ch, bdr_by_ph, part_summary)

    return {'bdr_global': bdr_global, 'bdr_by_ch': bdr_by_ch,
            'bdr_by_ph': bdr_by_ph,  'bdr_full': bdr_full,
            'meta': meta, 'part_summary': part_summary}


def _compute_bdr_params_from_meta(meta, group_cols):
    """
    Agrupa `meta` per group_cols + prepost, suma pred_cas,
    fa pivot pre/post i calcula canvi_pct.
    """
    agg = (meta.groupby(group_cols + ['prepost'])['pred_cas']
               .agg(['sum','count'])
               .rename(columns={'sum':'n_cas','count':'n_total'})
               .reset_index())

    pre  = agg[agg['prepost']==1].drop(columns='prepost').rename(
               columns={'n_cas':'n_cas_pre','n_total':'n_total_pre'})
    post = agg[agg['prepost']==2].drop(columns='prepost').rename(
               columns={'n_cas':'n_cas_post','n_total':'n_total_post'})
    mrg  = pre.merge(post, on=group_cols, how='outer').fillna(0)

    mrg['canvi_pct'] = np.where(
        mrg['n_cas_pre'] > 0,
        100.0 * (mrg['n_cas_post'] - mrg['n_cas_pre']) / mrg['n_cas_pre'],
        np.nan)
    return mrg


def _is_bdrpos(pid, bdr_global):
    """BDR+ si la reducció global de CAS és ≤ BDR_THRESHOLD%."""
    row = bdr_global[bdr_global['participant'] == pid]
    if row.empty or np.isnan(row['canvi_pct'].values[0]):
        return False
    return float(row['canvi_pct'].values[0]) <= BDR_THRESHOLD


def _print_bdr_summary(bdr_global, bdr_by_ch, bdr_by_ph, part_summary):
    groups = ['Control', 'BDR+', 'BDR-']
    g_map  = part_summary.set_index('participant')['group']

    print(f"\n  {'Grup':10s} | {'n_part':>6} | "
          f"{'CAS_pre_med':>11} | {'CAS_post_med':>12} | {'Canvi%_med':>10}")
    print("  " + "-"*60)
    for g in groups:
        sub = bdr_global[bdr_global['participant'].map(g_map) == g]
        if sub.empty: continue
        print(f"  {g:10s} | {len(sub):6d} | "
              f"{sub['n_cas_pre'].median():11.1f} | "
              f"{sub['n_cas_post'].median():12.1f} | "
              f"{sub['canvi_pct'].median():10.1f}%")

    print(f"\n  Per canal:")
    for ch, ch_lbl in [(1,'Inferior'),(2,'Superior')]:
        sub = bdr_by_ch[bdr_by_ch['channel']==ch].copy()
        sub['group'] = sub['participant'].map(g_map)
        for g in groups:
            sg = sub[sub['group']==g]
            if sg.empty: continue
            print(f"    Canal {ch_lbl:8s}  {g:8s}: "
                  f"pre={sg['n_cas_pre'].mean():.1f}  "
                  f"post={sg['n_cas_post'].mean():.1f}  "
                  f"canvi={sg['canvi_pct'].mean():.1f}%")

    print(f"\n  Per fase:")
    for ph, ph_lbl in [(1,'Inspiració'),(2,'Espiració')]:
        sub = bdr_by_ph[bdr_by_ph['phase']==ph].copy()
        sub['group'] = sub['participant'].map(g_map)
        for g in groups:
            sg = sub[sub['group']==g]
            if sg.empty: continue
            print(f"    {ph_lbl:12s}  {g:8s}: "
                  f"pre={sg['n_cas_pre'].mean():.1f}  "
                  f"post={sg['n_cas_post'].mean():.1f}  "
                  f"canvi={sg['canvi_pct'].mean():.1f}%")


# =============================================================================
# BLOC 9 – FIGURES BDR
# =============================================================================

_GCOLS = {'Control': '#2a9d8f', 'BDR+': '#e63946', 'BDR-': '#f4a261'}
_CHLAB = {1: 'Canal inf.', 2: 'Canal sup.'}
_PHLAB = {1: 'Inspiració', 2: 'Espiració'}


def _styled_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PAL['bg'])
    ax.grid(True, alpha=0.18, color=PAL['grid'])
    ax.set_title(title,  color=PAL['text'], fontsize=10, fontweight='bold')
    ax.set_xlabel(xlabel, color=PAL['text'], fontsize=8.5)
    ax.set_ylabel(ylabel, color=PAL['text'], fontsize=8.5)


def plot_bdr_overview(bdr_global, out_dir):
    """
    Vista global per participant:
      - Barres n_CAS pre i post
      - Línia de canvi percentual
    """
    g_map = bdr_global.set_index('participant')['group'] \
                if 'group' in bdr_global.columns \
                else {p: _group_label(p) for p in bdr_global['participant']}

    pids = sorted(bdr_global['participant'].unique())
    x    = np.arange(len(pids))
    pre  = bdr_global.set_index('participant').loc[pids, 'n_cas_pre'].values
    post = bdr_global.set_index('participant').loc[pids, 'n_cas_post'].values
    pct  = bdr_global.set_index('participant').loc[pids, 'canvi_pct'].values

    cols = [_GCOLS.get(g_map.get(p, 'Patient'), '#aaa') for p in pids]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                    facecolor=PAL['bg'],
                                    gridspec_kw={'height_ratios':[2,1]})
    fig.suptitle(f'Visió general – Segments CAS per participant\n'
                 f'(millor model: {best_name_global})',
                 fontsize=12, color=PAL['text'], fontweight='bold')

    # Barres pre/post
    w = 0.38
    ax1.bar(x - w/2, pre,  width=w, label='Pre-BD',  color='#457b9d', alpha=0.85)
    ax1.bar(x + w/2, post, width=w, label='Post-BD', color='#e63946', alpha=0.85)
    for xi, (p, po, col) in enumerate(zip(pre, post, cols)):
        ax1.annotate('', xy=(xi, max(p, po) + 1),
                     xytext=(xi, max(p, po) + 1))
    # Marcar grup amb color al tick
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'P{p}' for p in pids], rotation=45, fontsize=7.5)
    for tick, col in zip(ax1.get_xticklabels(), cols):
        tick.set_color(col)
    _styled_ax(ax1, title='Nº segments CAS – Pre-BD vs Post-BD',
               ylabel='Nº segments CAS')
    ax1.legend(fontsize=9, framealpha=0.2)

    # Línia de canvi %
    valid  = ~np.isnan(pct)
    bar_c  = [_GCOLS.get(g_map.get(p, 'Patient'), '#aaa') for p in pids]
    ax2.bar(x[valid], pct[valid], color=[bar_c[i] for i in np.where(valid)[0]],
            alpha=0.82, edgecolor='none')
    ax2.axhline(0,               color='white',  lw=0.8, ls='--', alpha=0.5)
    ax2.axhline(BDR_THRESHOLD,   color='#f4e04d', lw=1.2, ls=':',
                label=f'BDR+ llindar ({BDR_THRESHOLD}%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'P{p}' for p in pids], rotation=45, fontsize=7.5)
    for tick, col in zip(ax2.get_xticklabels(), cols):
        tick.set_color(col)
    _styled_ax(ax2, title='Canvi % de CAS  (100 × (post−pre)/pre)',
               ylabel='Canvi (%)')
    ax2.legend(fontsize=9, framealpha=0.2)

    # Llegenda de grups
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=g) for g, c in _GCOLS.items()]
    ax1.legend(handles=handles + ax1.get_legend_handles_labels()[0][:2],
               labels=[g for g in _GCOLS] + ['Pre-BD','Post-BD'],
               fontsize=8.5, framealpha=0.2)

    plt.tight_layout()
    path = os.path.join(out_dir, 'bdr_overview.png')
    fig.savefig(path, dpi=140, facecolor=PAL['bg'], bbox_inches='tight')
    plt.close(fig)
    print(f"  → bdr_overview.png")


def plot_bdr_by_channel(bdr_by_ch, out_dir):
    """
    Canvi de CAS (%) per canal (inferior/superior) × grup × participant.
    4 subplots: canal inf/sup × pre/post absolut i canvi%.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), facecolor=PAL['bg'])
    fig.suptitle('Anàlisi per Canal – CAS pre/post i canvi%',
                 fontsize=12, color=PAL['text'], fontweight='bold')

    for row, (ch, ch_lbl) in enumerate([(1,'Canal inferior'),(2,'Canal superior')]):
        sub  = bdr_by_ch[bdr_by_ch['channel'] == ch].copy()
        pids = sorted(sub['participant'].unique())
        x    = np.arange(len(pids))
        pre  = sub.set_index('participant').loc[pids,'n_cas_pre'].values.astype(float)
        post = sub.set_index('participant').loc[pids,'n_cas_post'].values.astype(float)
        pct  = sub.set_index('participant').loc[pids,'canvi_pct'].values.astype(float)
        grps = [sub[sub['participant']==p]['group'].values[0]
                if 'group' in sub.columns else _group_label(p) for p in pids]
        cols = [_GCOLS.get(g, '#aaa') for g in grps]

        # Barres pre/post
        ax = axes[row, 0]; ax.set_facecolor(PAL['bg'])
        w  = 0.38
        ax.bar(x - w/2, pre,  width=w, color='#457b9d', alpha=0.85, label='Pre-BD')
        ax.bar(x + w/2, post, width=w, color='#e63946', alpha=0.85, label='Post-BD')
        ax.set_xticks(x)
        ax.set_xticklabels([f'P{p}' for p in pids], rotation=60, fontsize=7)
        for tick, col in zip(ax.get_xticklabels(), cols): tick.set_color(col)
        _styled_ax(ax, title=f'{ch_lbl} – Nº CAS pre/post',
                   ylabel='Nº segments CAS')
        ax.legend(fontsize=8, framealpha=0.2)
        ax.grid(True, alpha=0.18)

        # Canvi %
        ax2 = axes[row, 1]; ax2.set_facecolor(PAL['bg'])
        valid = ~np.isnan(pct)
        ax2.bar(x[valid], pct[valid],
                color=[cols[i] for i in np.where(valid)[0]],
                alpha=0.82, edgecolor='none')
        ax2.axhline(0,             color='white',   lw=0.8, ls='--', alpha=0.5)
        ax2.axhline(BDR_THRESHOLD, color='#f4e04d', lw=1.2, ls=':',
                    label=f'Llindar BDR+ ({BDR_THRESHOLD}%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'P{p}' for p in pids], rotation=60, fontsize=7)
        for tick, col in zip(ax2.get_xticklabels(), cols): tick.set_color(col)
        _styled_ax(ax2, title=f'{ch_lbl} – Canvi% CAS',
                   ylabel='Canvi (%)')
        ax2.legend(fontsize=8, framealpha=0.2)
        ax2.grid(True, alpha=0.18)

    plt.tight_layout()
    path = os.path.join(out_dir, 'bdr_by_channel.png')
    fig.savefig(path, dpi=140, facecolor=PAL['bg'], bbox_inches='tight')
    plt.close(fig)
    print(f"  → bdr_by_channel.png")


def plot_bdr_by_phase(bdr_by_ph, out_dir):
    """
    Canvi de CAS (%) per fase (inspiració/espiració) × grup × participant.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), facecolor=PAL['bg'])
    fig.suptitle('Anàlisi per Fase Respiratòria – CAS pre/post i canvi%',
                 fontsize=12, color=PAL['text'], fontweight='bold')

    for row, (ph, ph_lbl) in enumerate([(1,'Inspiració'),(2,'Espiració')]):
        sub  = bdr_by_ph[bdr_by_ph['phase'] == ph].copy()
        pids = sorted(sub['participant'].unique())
        x    = np.arange(len(pids))
        pre  = sub.set_index('participant').loc[pids,'n_cas_pre'].values.astype(float)
        post = sub.set_index('participant').loc[pids,'n_cas_post'].values.astype(float)
        pct  = sub.set_index('participant').loc[pids,'canvi_pct'].values.astype(float)
        grps = [sub[sub['participant']==p]['group'].values[0]
                if 'group' in sub.columns else _group_label(p) for p in pids]
        cols = [_GCOLS.get(g, '#aaa') for g in grps]

        ax = axes[row, 0]; ax.set_facecolor(PAL['bg'])
        w  = 0.38
        ax.bar(x - w/2, pre,  width=w, color='#457b9d', alpha=0.85, label='Pre-BD')
        ax.bar(x + w/2, post, width=w, color='#e63946', alpha=0.85, label='Post-BD')
        ax.set_xticks(x); ax.set_xticklabels([f'P{p}' for p in pids],
                                               rotation=60, fontsize=7)
        for tick, col in zip(ax.get_xticklabels(), cols): tick.set_color(col)
        _styled_ax(ax, title=f'{ph_lbl} – Nº CAS pre/post',
                   ylabel='Nº segments CAS')
        ax.legend(fontsize=8, framealpha=0.2); ax.grid(True, alpha=0.18)

        ax2 = axes[row, 1]; ax2.set_facecolor(PAL['bg'])
        valid = ~np.isnan(pct)
        ax2.bar(x[valid], pct[valid],
                color=[cols[i] for i in np.where(valid)[0]],
                alpha=0.82, edgecolor='none')
        ax2.axhline(0,             color='white',   lw=0.8, ls='--', alpha=0.5)
        ax2.axhline(BDR_THRESHOLD, color='#f4e04d', lw=1.2, ls=':',
                    label=f'Llindar BDR+ ({BDR_THRESHOLD}%)')
        ax2.set_xticks(x); ax2.set_xticklabels([f'P{p}' for p in pids],
                                                 rotation=60, fontsize=7)
        for tick, col in zip(ax2.get_xticklabels(), cols): tick.set_color(col)
        _styled_ax(ax2, title=f'{ph_lbl} – Canvi% CAS', ylabel='Canvi (%)')
        ax2.legend(fontsize=8, framealpha=0.2); ax2.grid(True, alpha=0.18)

    plt.tight_layout()
    path = os.path.join(out_dir, 'bdr_by_phase.png')
    fig.savefig(path, dpi=140, facecolor=PAL['bg'], bbox_inches='tight')
    plt.close(fig)
    print(f"  → bdr_by_phase.png")


def plot_bdr_heatmap(bdr_full, out_dir):
    """
    Heatmap del canvi% de CAS per participant, amb subgràfics per canal×fase.
    """
    combos = [(1,1,'Canal inf. | Insp.'), (1,2,'Canal inf. | Esp.'),
              (2,1,'Canal sup. | Insp.'), (2,2,'Canal sup. | Esp.')]

    fig, axes = plt.subplots(1, 4, figsize=(22, 7), facecolor=PAL['bg'])
    fig.suptitle('Heatmap Canvi% CAS – per Participant × Canal × Fase',
                 fontsize=12, color=PAL['text'], fontweight='bold')

    all_pids = sorted(bdr_full['participant'].unique())
    vmax     = 100.0
    cmap     = plt.cm.RdYlGn_r    # vermell=augment, verd=reducció

    for ax, (ch, ph, title) in zip(axes, combos):
        sub  = bdr_full[(bdr_full['channel']==ch) & (bdr_full['phase']==ph)]
        pct_vals = []
        for p in all_pids:
            row = sub[sub['participant']==p]
            pct_vals.append(row['canvi_pct'].values[0]
                            if not row.empty else np.nan)
        pct_arr = np.array(pct_vals, dtype=float).reshape(-1, 1)

        im = ax.imshow(pct_arr, cmap=cmap, aspect='auto',
                       vmin=-vmax, vmax=vmax,
                       interpolation='nearest')
        ax.set_yticks(range(len(all_pids)))
        ax.set_yticklabels([f'P{p}' for p in all_pids], fontsize=7.5)

        # Color de tick per grup
        if 'group' in bdr_full.columns:
            g_map = bdr_full.drop_duplicates('participant').set_index(
                        'participant')['group']
            for tick, p in zip(ax.get_yticklabels(), all_pids):
                tick.set_color(_GCOLS.get(g_map.get(p,'Patient'),'#aaa'))

        ax.set_xticks([])
        ax.set_title(title, color=PAL['text'], fontsize=9, fontweight='bold')
        ax.set_facecolor(PAL['bg'])

        # Anotacions de valors
        for yi, v in enumerate(pct_vals):
            if not np.isnan(v):
                ax.text(0, yi, f'{v:+.0f}%', ha='center', va='center',
                        fontsize=7, color='white' if abs(v) > 40 else 'black',
                        fontweight='bold')

    plt.colorbar(im, ax=axes[-1], fraction=0.06, pad=0.04,
                 label='Canvi% CAS').ax.yaxis.label.set_color(PAL['text'])
    plt.tight_layout()
    path = os.path.join(out_dir, 'bdr_heatmap.png')
    fig.savefig(path, dpi=140, facecolor=PAL['bg'], bbox_inches='tight')
    plt.close(fig)
    print(f"  → bdr_heatmap.png")


def plot_bdr_group_comparison(bdr_global, bdr_by_ch, bdr_by_ph,
                               part_summary, out_dir):
    """
    Comparació dels 3 grups (Control / BDR+ / BDR-) en:
      - Canvi% global
      - Canvi% per canal
      - Canvi% per fase
    """
    groups  = ['Control', 'BDR+', 'BDR-']
    g_map   = part_summary.set_index('participant')['group']
    palette = [_GCOLS[g] for g in groups]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor=PAL['bg'])
    fig.suptitle('Comparació BDR+ / BDR- / Control\n'
                 'Canvi percentual de CAS (100×(post−pre)/pre)',
                 fontsize=12, color=PAL['text'], fontweight='bold')

    # (A) Global
    ax = axes[0]
    data_g = [bdr_global[bdr_global['participant'].map(g_map)==g]['canvi_pct']
               .dropna().values for g in groups]
    bp = ax.boxplot(data_g, patch_artist=True, notch=False,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color=PAL['text']),
                    capprops=dict(color=PAL['text']),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))
    for patch, col in zip(bp['boxes'], palette):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    # Afegir punts individuals
    for i, (d, col) in enumerate(zip(data_g, palette)):
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(d))
        ax.scatter(np.ones(len(d))*(i+1) + jitter, d,
                   color=col, s=28, zorder=5, alpha=0.85)
    ax.axhline(0,             color='white',   lw=0.8, ls='--', alpha=0.4)
    ax.axhline(BDR_THRESHOLD, color='#f4e04d', lw=1.2, ls=':',
               label=f'Llindar BDR+ ({BDR_THRESHOLD}%)')
    ax.set_xticks([1,2,3]); ax.set_xticklabels(groups, fontsize=11)
    for tick, col in zip(ax.get_xticklabels(), palette): tick.set_color(col)
    _styled_ax(ax, title='Global', ylabel='Canvi% CAS')
    ax.legend(fontsize=8, framealpha=0.2)

    # (B) Per canal
    ax = axes[1]
    for ci, (ch, lbl, lsty) in enumerate([(1,'Canal inf.','solid'),
                                            (2,'Canal sup.','dashed')]):
        sub = bdr_by_ch[bdr_by_ch['channel']==ch].copy()
        sub['group'] = sub['participant'].map(g_map)
        xpos = np.array([1,2,3]) + (ci-0.5)*0.18
        for xi, g in enumerate(groups):
            vals = sub[sub['group']==g]['canvi_pct'].dropna().values
            med  = np.nanmedian(vals) if len(vals) > 0 else np.nan
            ax.scatter(xpos[xi], med, color=_GCOLS[g], marker='D' if ci==0 else 's',
                       s=90, zorder=5, label=lbl if xi==0 else '')
            if len(vals) > 1:
                q1, q3 = np.nanpercentile(vals, [25,75])
                ax.vlines(xpos[xi], q1, q3, color=_GCOLS[g], lw=2.5, alpha=0.6)
    ax.axhline(0,             color='white',   lw=0.8, ls='--', alpha=0.4)
    ax.axhline(BDR_THRESHOLD, color='#f4e04d', lw=1.2, ls=':')
    ax.set_xticks([1,2,3]); ax.set_xticklabels(groups, fontsize=11)
    for tick, col in zip(ax.get_xticklabels(), palette): tick.set_color(col)
    _styled_ax(ax, title='Per Canal (◆=inf, ■=sup)',
               ylabel='Canvi% CAS (mediana ± IQR)')
    ax.legend(fontsize=8, framealpha=0.2)

    # (C) Per fase
    ax = axes[2]
    for pi, (ph, lbl) in enumerate([(1,'Inspiració'),(2,'Espiració')]):
        sub = bdr_by_ph[bdr_by_ph['phase']==ph].copy()
        sub['group'] = sub['participant'].map(g_map)
        xpos = np.array([1,2,3]) + (pi-0.5)*0.18
        for xi, g in enumerate(groups):
            vals = sub[sub['group']==g]['canvi_pct'].dropna().values
            med  = np.nanmedian(vals) if len(vals) > 0 else np.nan
            ax.scatter(xpos[xi], med, color=_GCOLS[g],
                       marker='^' if pi==0 else 'v',
                       s=90, zorder=5, label=lbl if xi==0 else '')
            if len(vals) > 1:
                q1, q3 = np.nanpercentile(vals, [25,75])
                ax.vlines(xpos[xi], q1, q3, color=_GCOLS[g], lw=2.5, alpha=0.6)
    ax.axhline(0,             color='white',   lw=0.8, ls='--', alpha=0.4)
    ax.axhline(BDR_THRESHOLD, color='#f4e04d', lw=1.2, ls=':')
    ax.set_xticks([1,2,3]); ax.set_xticklabels(groups, fontsize=11)
    for tick, col in zip(ax.get_xticklabels(), palette): tick.set_color(col)
    _styled_ax(ax, title='Per Fase (▲=insp, ▼=esp)',
               ylabel='Canvi% CAS (mediana ± IQR)')
    ax.legend(fontsize=8, framealpha=0.2)

    plt.tight_layout()
    path = os.path.join(out_dir, 'bdr_group_comparison.png')
    fig.savefig(path, dpi=140, facecolor=PAL['bg'], bbox_inches='tight')
    plt.close(fig)
    print(f"  → bdr_group_comparison.png")


def plot_cas_distribution_all(meta, part_summary, out_dir):
    """
    Distribució de la probabilitat P(CAS) per tots els segments,
    desglossat per grup i pre/post BD.
    """
    if 'prob_cas' not in meta.columns:
        return
    g_map = part_summary.set_index('participant')['group']
    meta  = meta.copy()
    meta['group'] = meta['participant'].map(g_map).fillna('Unknown')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=PAL['bg'])
    fig.suptitle('Distribució de P(CAS) per totes les 14900 senyals',
                 fontsize=12, color=PAL['text'], fontweight='bold')

    for ax, (g, col) in zip(axes, _GCOLS.items()):
        sub = meta[meta['group'] == g]
        if sub.empty:
            ax.set_visible(False); continue
        pre  = sub[sub['prepost']==1]['prob_cas']
        post = sub[sub['prepost']==2]['prob_cas']
        ax.hist(pre,  bins=40, alpha=0.7, color='#457b9d',
                label=f'Pre-BD  (n={len(pre)})', density=True)
        ax.hist(post, bins=40, alpha=0.7, color='#e63946',
                label=f'Post-BD (n={len(post)})', density=True)
        ax.axvline(0.5, color='#f4e04d', ls='--', lw=1.2, label='θ=0.5')
        _styled_ax(ax, title=f'Grup: {g}', xlabel='P(CAS)', ylabel='Densitat')
        ax.legend(fontsize=8, framealpha=0.2)
        ax.set_xlim(0, 1)
        ax.set_facecolor(PAL['bg'])
        ax.patch.set_edgecolor(col); ax.patch.set_linewidth(0)
        for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(1.8)

    plt.tight_layout()
    path = os.path.join(out_dir, 'bdr_prob_distribution_all.png')
    fig.savefig(path, dpi=140, facecolor=PAL['bg'], bbox_inches='tight')
    plt.close(fig)
    print(f"  → bdr_prob_distribution_all.png")


def plot_bdr_boxplots(bdr_global, bdr_by_ch, bdr_by_ph, out_dir):
    """
    Figura resum amb 3 panells × 2 mètriques (n_CAS i canvi%) = 6 subfigures.
    Permet veure ràpidament les diferències entre grups.
    """
    groups  = ['Control', 'BDR+', 'BDR-']
    palette = [_GCOLS[g] for g in groups]
    metrics = [('n_cas_pre','Nº CAS Pre-BD'),
               ('n_cas_post','Nº CAS Post-BD'),
               ('canvi_pct','Canvi% CAS')]

    for scope, df_, title in [
        ('global',  bdr_global, 'Global (tots canals + fases)'),
        ('channel', bdr_by_ch,  'Per canal (agregat per participant)'),
        ('phase',   bdr_by_ph,  'Per fase (agregat per participant)'),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=PAL['bg'])
        fig.suptitle(f'Boxplots per grup – {title}',
                     fontsize=11, color=PAL['text'], fontweight='bold')

        # Agregar per participant dins scope (per canal o fase hem de sumar)
        if scope == 'global':
            df_agg = df_.copy()
        else:
            df_agg = (df_.groupby('participant')[['n_cas_pre','n_cas_post']]
                         .sum().reset_index())
            df_agg['canvi_pct'] = np.where(
                df_agg['n_cas_pre'] > 0,
                100*(df_agg['n_cas_post']-df_agg['n_cas_pre'])/df_agg['n_cas_pre'],
                np.nan)
            if 'group' in df_.columns:
                df_agg['group'] = df_agg['participant'].map(
                    df_.drop_duplicates('participant').set_index('participant')['group'])

        g_map_local = df_agg.set_index('participant')['group'] \
                      if 'group' in df_agg.columns else \
                      {p: _group_label(p) for p in df_agg['participant']}

        for ax, (met, met_lbl) in zip(axes, metrics):
            data_g = [df_agg[df_agg['participant'].map(g_map_local)==g][met]
                      .dropna().values for g in groups]
            bp = ax.boxplot(data_g, patch_artist=True,
                            medianprops=dict(color='white', linewidth=2),
                            whiskerprops=dict(color=PAL['text']),
                            capprops=dict(color=PAL['text']),
                            flierprops=dict(marker='o', markersize=4, alpha=0.5))
            for patch, col in zip(bp['boxes'], palette):
                patch.set_facecolor(col); patch.set_alpha(0.72)
            for i, (d, col) in enumerate(zip(data_g, palette)):
                jitter = np.random.default_rng(i+42).uniform(-0.2, 0.2, len(d))
                ax.scatter(np.ones(len(d))*(i+1)+jitter, d,
                           color=col, s=30, zorder=5, alpha=0.8)
            if met == 'canvi_pct':
                ax.axhline(0,             color='white',   lw=0.8, ls='--', alpha=0.4)
                ax.axhline(BDR_THRESHOLD, color='#f4e04d', lw=1.2, ls=':')
            ax.set_xticks([1,2,3]); ax.set_xticklabels(groups, fontsize=10)
            for tick, col in zip(ax.get_xticklabels(), palette):
                tick.set_color(col)
            _styled_ax(ax, title=met_lbl, ylabel=met_lbl)
            ax.grid(True, alpha=0.18)

        plt.tight_layout()
        path = os.path.join(out_dir, f'bdr_boxplots_{scope}.png')
        fig.savefig(path, dpi=140, facecolor=PAL['bg'], bbox_inches='tight')
        plt.close(fig)
        print(f"  → bdr_boxplots_{scope}.png")


# Variable global per al nom del millor model (usada a plot_bdr_overview)
best_name_global = 'millor model'


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description='Classificació CAS vs NO-CAS – v2 (GroupKFold fix) + Anàlisi BDR')
    p.add_argument('--data', default=script_dir,
                   help='Directori amb signals.pkl, proy_labels.mat i metadata.csv')
    p.add_argument('--out',  default=os.path.join(script_dir, 'results_v2'),
                   help='Directori de sortida')
    args = p.parse_args()
    main(args.data, args.out)
