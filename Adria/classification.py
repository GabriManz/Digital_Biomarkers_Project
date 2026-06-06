"""
CLASSIFICACIÓ DE SONS RESPIRATORIS  –  CAS vs NO-CAS
=====================================================
Carrega les senyals preprocessades i les classifica amb:

  ML CLÀSSIC
  ① Logistic Regression
  ② SVM Lineal
  ③ SVM RBF
  ④ Random Forest
  ⑤ XGBoost

  DEEP LEARNING
  ⑥ CNN 1-D  (sobre espectrograma log-mel)
  ⑦ BiLSTM   (sobre espectrograma log-mel)

  HYBRID ENSEMBLE
  ⑧ Ensemble ponderat (millor ML + millor DL)

Validació: GroupKFold(5) – cap participant compartit entre train i test.
Sortides:  results/  →  taules CSV, matrius confusió PNG, ROC PNG,
                         comparació PNG, model.pkl / model.keras

Ús:
  python classification.py                    # usa ./preprocessed/
  python classification.py --data /ruta/dades
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
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.signal import welch
import librosa
import pywt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVC
from sklearn.ensemble      import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics        import (accuracy_score, precision_score,
                                    recall_score, f1_score,
                                    roc_auc_score, confusion_matrix,
                                    roc_curve, average_precision_score)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models    import Model
from tensorflow.keras.layers    import (Input, Conv1D, MaxPooling1D,
                                        Bidirectional, LSTM, Dense,
                                        Dropout, BatchNormalization,
                                        GlobalAveragePooling1D, Flatten)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
tf.random.set_seed(42);  np.random.seed(42)

# ── constants ─────────────────────────────────────────────────────────────────
FS         = 4000
N_MFCC     = 20
N_MELS     = 64
FIXED_LEN  = 8000        # 2 s padded/truncat per al DL
N_FFT      = 256
HOP        = 128
N_SPLITS   = 5
SEED       = 42
DL_EPOCHS  = 50
DL_BATCH   = 32

# ── paleta ────────────────────────────────────────────────────────────────────
PAL = {
    'LR':       '#4e79a7',
    'SVM-Lin':  '#f28e2b',
    'SVM-RBF':  '#e15759',
    'RF':       '#76b7b2',
    'XGB':      '#59a14f',
    'CNN-1D':   '#edc948',
    'BiLSTM':   '#b07aa1',
    'Ensemble': '#ff9da7',
    'bg':       '#0f1117',
    'grid':     '#2a2d35',
    'text':     '#e8eaf0',
}
plt.rcParams.update({
    'figure.facecolor': PAL['bg'], 'axes.facecolor':  PAL['bg'],
    'axes.edgecolor':   PAL['grid'], 'axes.labelcolor': PAL['text'],
    'xtick.color':      PAL['text'], 'ytick.color':    PAL['text'],
    'text.color':       PAL['text'], 'grid.color':     PAL['grid'],
    'font.family':      'monospace', 'axes.titlesize': 10,
})

# =============================================================================
# BLOC 1 – EXTRACCIÓ DE CARACTERÍSTIQUES  (ML clàssic)
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
    s  = _safe(s)
    f, p = welch(s, fs=FS, nperseg=min(512, len(s)))
    tp = np.sum(p) + 1e-12;  pn = p / tp
    sc = np.sum(f * pn)
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
    s = _safe(s, 2048)
    m = librosa.feature.mfcc(y=s, sr=FS, n_mfcc=N_MFCC)
    nf = m.shape[1]
    w  = min(9, nf if nf % 2 == 1 else max(nf-1,1))
    w  = max(w, 3)
    mo = 'interp' if nf >= w else 'nearest'
    d  = librosa.feature.delta(m, width=w, mode=mo)
    d2 = librosa.feature.delta(m, width=w, mode=mo, order=2)
    return (list(np.mean(m,1)) + list(np.std(m,1)) +
            list(np.mean(d,1)) + list(np.std(d,1)) +
            list(np.mean(d2,1)) + list(np.std(d2,1)))   # 120

def feat_wavelet(s):
    s = _safe(s)
    coeffs = pywt.wavedec(s, 'db4', level=5)
    out = []
    for c in coeffs[1:]:
        e = np.sum(c**2); prob = c**2 / (e + 1e-12)
        out.extend([e, -np.sum(prob * np.log(prob + 1e-12)), np.std(c)])
    return out   # 15

def extract_features(signals):
    print(f"  Extraient {len(signals)} segments …")
    X = []
    for i, s in enumerate(signals):
        s = np.asarray(s, np.float64)
        if len(s) == 0: s = np.zeros(64)
        X.append(feat_temporal(s) + feat_spectral(s) +
                 feat_mfcc(s)    + feat_wavelet(s))
        if (i+1) % 500 == 0:
            print(f"    {i+1}/{len(signals)}")
    return np.array(X, dtype=np.float32)

# =============================================================================
# BLOC 2 – LOG-MEL ESPECTROGRAMA  (DL)
# =============================================================================

def to_logmel(signals, fixed=FIXED_LEN):
    print(f"  Log-mel de {len(signals)} segments …")
    out = []
    for s in signals:
        s = np.asarray(s, np.float32)
        if len(s) > fixed:
            c = (len(s) - fixed) // 2
            s = s[c:c+fixed]
        else:
            s = np.pad(s, (0, fixed - len(s)))
        mel = librosa.feature.melspectrogram(y=s, sr=FS, n_fft=N_FFT,
                                              hop_length=HOP, n_mels=N_MELS,
                                              fmin=70, fmax=1900)
        mel = librosa.power_to_db(mel, ref=np.max).T   # (frames, mels)
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)
        out.append(mel)
    return np.array(out, np.float32)  # (N, frames, mels)

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
    m.compile(optimizer='adam', loss='binary_crossentropy',
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
    m.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return m

# =============================================================================
# BLOC 4 – MÈTRIQUES I PLOTS
# =============================================================================

def metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        'Accuracy':    round(accuracy_score(y_true, y_pred),        4),
        'Precision':   round(precision_score(y_true, y_pred,        zero_division=0), 4),
        'Recall':      round(recall_score(y_true, y_pred,           zero_division=0), 4),
        'Specificity': round(tn / (tn + fp + 1e-12),                4),
        'F1':          round(f1_score(y_true, y_pred,               zero_division=0), 4),
        'ROC-AUC':     round(roc_auc_score(y_true, y_prob),         4),
        'PR-AUC':      round(average_precision_score(y_true, y_prob),4),
    }

def plot_confusion_matrices(results, out):
    n = len(results)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor=PAL['bg'])
    axes = axes.flatten()
    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(res['y_true'], res['y_pred'], labels=[0,1])
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['NO-CAS','CAS'],
                    yticklabels=['NO-CAS','CAS'],
                    cbar=False, linewidths=0.5)
        ax.set_title(name, color=PAL.get(name, PAL['text']), fontsize=10)
        ax.set_xlabel('Predicció', fontsize=8)
        ax.set_ylabel('Real', fontsize=8)
        ax.set_facecolor(PAL['bg'])
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Matrius de Confusió – Tots els Models (OOF)',
                 fontsize=13, color=PAL['text'])
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
        color = PAL.get(name, '#aaaaaa')
        lw    = 2.5 if name == 'Ensemble' else 1.5
        ax.plot(fpr, tpr, color=color, linewidth=lw,
                label=f'{name}  AUC={auc:.3f}')
    ax.plot([0,1],[0,1], 'w--', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('FPR (1-Especificitat)', fontsize=10)
    ax.set_ylabel('TPR (Recall)',          fontsize=10)
    ax.set_title('Corbes ROC – Comparació de Models (OOF)', fontsize=12,
                 color=PAL['text'])
    ax.legend(fontsize=8.5, framealpha=0.2, labelcolor=PAL['text'],
              loc='lower right')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(out, 'roc_curves.png'),
                dpi=140, facecolor=PAL['bg'], bbox_inches='tight')
    plt.close(fig)
    print("  → roc_curves.png")

def plot_metrics_comparison(results, out):
    keys   = ['Accuracy','Precision','Recall','Specificity','F1','ROC-AUC']
    models = list(results.keys())
    values = {k: [results[m]['metrics'][k] for m in models] for k in keys}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=PAL['bg'])
    axes = axes.flatten()
    for i, k in enumerate(keys):
        ax   = axes[i]
        ax.set_facecolor(PAL['bg'])
        bars = ax.bar(models, values[k],
                      color=[PAL.get(m,'#aaaaaa') for m in models],
                      alpha=0.85, width=0.6, edgecolor='none')
        ax.set_ylim(0, 1.08)
        ax.set_title(k, color=PAL['text'], fontsize=11)
        ax.set_ylabel('Score', fontsize=8)
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.grid(True, alpha=0.2, axis='y')
        for bar, v in zip(bars, values[k]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                    f'{v:.3f}', ha='center', va='bottom',
                    fontsize=7.5, color=PAL['text'])
        # highlight best
        best_idx = int(np.argmax(values[k]))
        axes[i].patches[best_idx].set_edgecolor('#f4e04d')
        axes[i].patches[best_idx].set_linewidth(2)

    fig.suptitle('Comparació de Mètriques – Tots els Models',
                 fontsize=13, color=PAL['text'], fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out, 'metrics_comparison.png'),
                dpi=140, facecolor=PAL['bg'], bbox_inches='tight')
    plt.close(fig)
    print("  → metrics_comparison.png")

def plot_radar(results, out):
    keys    = ['Accuracy','Precision','Recall','Specificity','F1','ROC-AUC']
    N       = len(keys)
    angles  = [n/N*2*np.pi for n in range(N)] + [0]
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True),
                            facecolor=PAL['bg'])
    ax.set_facecolor(PAL['bg'])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys, size=9, color=PAL['text'])
    ax.set_ylim(0, 1)
    ax.grid(color=PAL['grid'], linewidth=0.6)
    for name, res in results.items():
        vals = [res['metrics'][k] for k in keys] + [res['metrics'][keys[0]]]
        ax.plot(angles, vals, color=PAL.get(name,'#aaa'),
                linewidth=1.8, linestyle='solid', label=name)
        ax.fill(angles, vals, alpha=0.06, color=PAL.get(name,'#aaa'))
    ax.legend(loc='upper right', bbox_to_anchor=(1.35,1.15),
              fontsize=8.5, framealpha=0.2, labelcolor=PAL['text'])
    ax.set_title('Radar de Mètriques', color=PAL['text'],
                 fontsize=12, pad=18)
    plt.tight_layout()
    fig.savefig(os.path.join(out,'radar_metrics.png'),
                dpi=140, facecolor=PAL['bg'], bbox_inches='tight')
    plt.close(fig)
    print("  → radar_metrics.png")

def save_table(results, out):
    rows = []
    for name, res in results.items():
        row = {'Model': name}
        row.update(res['metrics'])
        rows.append(row)
    df = pd.DataFrame(rows).set_index('Model')
    df.to_csv(os.path.join(out, 'metrics_table.csv'))
    # Pretty print
    print("\n" + "="*75)
    print(df.to_string())
    print("="*75)
    print(f"  → metrics_table.csv")
    return df

# =============================================================================
# BLOC 5 – VALIDACIÓ CREUADA AGRUPADA (AMB FALLBACK A ESTRATIFICADA)
# =============================================================================

def cv_ml(clf, X, y, groups, use_smote=True, name=''):
    unique_groups = len(np.unique(groups))
    
    if unique_groups < 2:
        print(f"    [Avís] 1 participant detectat. Usant StratifiedKFold({N_SPLITS})", end=' ... ')
        gkf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = list(gkf.split(X, y))
    else:
        n_splits = min(N_SPLITS, unique_groups)
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X, y, groups))
        
    oof_prob = np.zeros(len(y))
    oof_pred = np.zeros(len(y), dtype=int)
    
    for fold, (tr, te) in enumerate(splits):
        sc   = StandardScaler()
        Xtr  = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        ytr  = y[tr]
        if use_smote and ytr.sum() > 1 and (ytr==0).sum() > 1:
            sm = SMOTE(random_state=SEED, k_neighbors=min(5, ytr.sum()-1))
            Xtr, ytr = sm.fit_resample(Xtr, ytr)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:,1]
        oof_prob[te] = prob
        oof_pred[te] = (prob >= 0.5).astype(int)
    return oof_prob, oof_pred

def cv_dl(build_fn, X_mel, y, groups, name=''):
    unique_groups = len(np.unique(groups))
    
    if unique_groups < 2:
        gkf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = list(gkf.split(X_mel, y))
    else:
        n_splits = min(N_SPLITS, unique_groups)
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X_mel, y, groups))
        
    oof  = np.zeros(len(y))
    neg, pos = (y==0).sum(), (y==1).sum()
    cw   = {0: (neg+pos)/(2*neg), 1: (neg+pos)/(2*pos)}
    
    for fold, (tr, te) in enumerate(splits):
        print(f"    {name} fold {fold+1}/{N_SPLITS} …", end=' ', flush=True)
        tf.keras.backend.clear_session()
        mdl = build_fn(X_mel.shape[1:])
        cb  = [EarlyStopping(monitor='val_auc', patience=8,
                             restore_best_weights=True, mode='max'),
               ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                                 patience=5, mode='max', min_lr=1e-6)]
        mdl.fit(X_mel[tr], y[tr], epochs=DL_EPOCHS, batch_size=DL_BATCH,
                validation_data=(X_mel[te], y[te]),
                class_weight=cw, callbacks=cb, verbose=0)
        oof[te] = mdl.predict(X_mel[te], verbose=0).flatten()
        print(f"AUC≈{roc_auc_score(y[te], oof[te]):.3f}")
    return oof, (oof >= 0.5).astype(int)

# =============================================================================
# BLOC 6 – PIPELINE PRINCIPAL
# =============================================================================

def main(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # ── ① Càrrega de senyals preprocessades ────────────────────────────────────
    print("\n① Carregant senyals preprocessades i etiquetes …")
    sig_path = os.path.join(data_dir, 'preprocessed', 'signals.pkl')
    if not os.path.exists(sig_path):
        sig_path = os.path.join(data_dir, 'signals.pkl')
        
    with open(sig_path, 'rb') as f:
        all_signals = pickle.load(f)

    mat_path = os.path.join(data_dir, 'proy_labels.mat')
    if not os.path.exists(mat_path):
        mat_path = os.path.join(data_dir, '..', 'proy_labels.mat')
        
    if os.path.exists(mat_path):
        mat_data = scipy.io.loadmat(mat_path)
        claus_reals = [k for k in mat_data.keys() if not k.startswith('__')]
        
        if 'labels' in mat_data: labels_raw = mat_data['labels'].flatten()
        elif 'y' in mat_data: labels_raw = mat_data['y'].flatten()
        else: labels_raw = mat_data[claus_reals[0]].flatten()
            
        if 'participants' in mat_data: g_raw = mat_data['participants'].flatten()
        elif len(claus_reals) > 1: g_raw = mat_data[claus_reals[1]].flatten()
        else: g_raw = np.ones(len(labels_raw))
    else:
        raise FileNotFoundError(f"No s'ha trobat el fitxer: {mat_path}")

    # Igualem longituds per seguretat
    min_len = min(len(all_signals), len(labels_raw))
    labels_raw = labels_raw[:min_len]
    g_raw = g_raw[:min_len]

    # FILTREM NOMÉS LES DADES ETIQUETADES (2=CAS, 3=NO-CAS)
    idx  = np.where((labels_raw == 2) | (labels_raw == 3))[0]
    sigs = [all_signals[i] for i in idx]
    y    = np.where(labels_raw[idx] == 2, 1, 0).astype(int) # 2->1 (CAS), 3->0 (NO-CAS)
    g    = g_raw[idx].astype(int)

    n_cas, n_nocas = np.sum(y == 1), np.sum(y == 0)
    print(f"  Segments etiquetats a processar: {len(y)}  (CAS={n_cas}, NO-CAS={n_nocas})")
    print(f"  Participants únics: {np.unique(g)}")

    # ── ② Extracció de característiques (ML) ──────────────────────────────────
    print("\n② Extracció de característiques …")
    X = extract_features(sigs)
    print(f"  Matriu de característiques: {X.shape}")

    # ── ③ Log-mel (DL) ─────────────────────────────────────────────────────────
    print("\n③ Espectrogrames log-mel …")
    X_mel = to_logmel(sigs)
    print(f"  Tensor log-mel: {X_mel.shape}")

    # ── ④ Models ML ─────────────────────────────────────────────────────────────
    print("\n④ Validació creuada – Models ML …")
    ml_models = {
        'LR':      LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=SEED),
        'SVM-Lin': SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=SEED),
        'SVM-RBF': SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', probability=True, random_state=SEED),
        'RF':      RandomForestClassifier(n_estimators=500, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=SEED),
        'XGB':     XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, objective='binary:logistic', eval_metric='logloss',
                                 n_jobs=-1, random_state=SEED),
    }

    results = {}
    for name, clf in ml_models.items():
        print(f"  {name} …", end=' ', flush=True)
        prob, pred = cv_ml(clf, X, y, g, name=name)
        results[name] = {'y_true': y, 'y_pred': pred, 'y_prob': prob, 'metrics': metrics(y, pred, prob)}
        print(f"F1={results[name]['metrics']['F1']:.3f}  AUC={results[name]['metrics']['ROC-AUC']:.3f}")

    # ── ⑤ Models DL ─────────────────────────────────────────────────────────────
    print("\n⑤ Validació creuada – Models DL …")
    dl_builds = {'CNN-1D': build_cnn1d, 'BiLSTM': build_bilstm}
    for name, build_fn in dl_builds.items():
        print(f"  {name}:")
        prob, pred = cv_dl(build_fn, X_mel, y, g, name=name)
        results[name] = {'y_true': y, 'y_pred': pred, 'y_prob': prob, 'metrics': metrics(y, pred, prob)}

    # ── ⑥ Ensemble ─────────────────────────────────────────────────────────────
    print("\n⑥ Ensemble ponderat …")
    best_ml = max((n for n in ml_models), key=lambda n: results[n]['metrics']['F1'])
    best_dl = max(dl_builds, key=lambda n: results[n]['metrics']['F1'])
    print(f"  Millor ML → {best_ml}   Millor DL → {best_dl}")

    best_f1, best_w = 0, 0.5
    for w in np.linspace(0.1, 0.9, 17):
        ens = w * results[best_ml]['y_prob'] + (1-w) * results[best_dl]['y_prob']
        f1v = f1_score(y, (ens >= 0.5).astype(int), zero_division=0)
        if f1v > best_f1: best_f1, best_w = f1v, w
        
    ens_prob = best_w * results[best_ml]['y_prob'] + (1-best_w) * results[best_dl]['y_prob']
    ens_pred = (ens_prob >= 0.5).astype(int)
    results['Ensemble'] = {'y_true': y, 'y_pred': ens_pred, 'y_prob': ens_prob, 'metrics': metrics(y, ens_pred, ens_prob)}
    print(f"  w_ML={best_w:.2f}  F1={results['Ensemble']['metrics']['F1']:.3f}  AUC={results['Ensemble']['metrics']['ROC-AUC']:.3f}")

    # ── ⑦ Figures i taules ──────────────────────────────────────────────────────
    print("\n⑦ Generant figures …")
    plot_confusion_matrices(results, out_dir)
    plot_roc_curves(results,         out_dir)
    plot_metrics_comparison(results, out_dir)
    plot_radar(results,              out_dir)
    df_metrics = save_table(results, out_dir)

    # ── ⑧ Guardar model ─────────────────────────────────────────────────────────
    print("\n⑧ Guardant el millor model …")
    best_name = max(results, key=lambda n: results[n]['metrics']['F1'])
    
    if best_name in ml_models:
        sc_final = StandardScaler()
        Xf, yf = sc_final.fit_transform(X), y
        if y.sum() > 1 and (y==0).sum() > 1:
            sm = SMOTE(random_state=SEED)
            Xf, yf = sm.fit_resample(Xf, y)
        ml_models[best_name].fit(Xf, yf)
        joblib.dump(ml_models[best_name], os.path.join(out_dir,'best_model.pkl'))
        joblib.dump(sc_final,             os.path.join(out_dir,'scaler.pkl'))
        print(f"  → {best_name} (best_model.pkl + scaler.pkl guardats)")
    else:
        print(f"  → {best_name} (Model DL)")

    print(f"\n  Temps total: {time.time()-t0:.0f}s\n  Resultats a: {out_dir}")
    return df_metrics, results

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=script_dir, help='Directori amb signals.pkl i proy_labels.mat')
    p.add_argument('--out',  default=os.path.join(script_dir, 'results'), help='Directori de sortida')
    args = p.parse_args()

    main(args.data, args.out)