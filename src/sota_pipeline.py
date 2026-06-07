"""
Pipeline alternativo y minimalista para la detección de CAS y evaluación de BDR.

Extrae un conjunto de 4 características físicas e interpretables:
  1. Índice Tonal (Tonal Index - TI)
  2. Entropía de Picos Espectrales (Spectral Peaks Entropy - SPE)
  3. Curtosis Espectral (Spectral Kurtosis - SK)
  4. Ratio f50/f90

Realiza validación Leave-One-Subject-Out (LOSO) de 18 folds con un modelo auxiliar
para la eliminación lineal de la identidad del paciente (Nullspace Projection).
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import json
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneGroupOut

# Permitir importaciones locales desde src/
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from step4_dataset import (
    build_dataset,
    _load_metadata,
    _build_subject_list,
    METADATA_CSV,
    FS_TARGET,
)

# Directorios de salida
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "outputs", "results", "sota")
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "outputs", "figures", "sota")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Comprobar disponibilidad de PyTorch para el LSTM opcional
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    pass


# ===========================================================================
# 1. Extracción del Feature Set Minimalista
# ===========================================================================

def extract_sota_features_sequence(sig: np.ndarray, fs: int = FS_TARGET) -> np.ndarray:
    """
    Extrae la secuencia temporal de las 4 características físicas para cada ventana del STFT.
    Retorna un array de forma (n_frames, 4).
    """
    # Garantizar una longitud mínima para tener al menos 3 ventanas en el STFT
    if len(sig) < 512:
        sig = np.pad(sig, (0, 512 - len(sig)), mode='constant')

    f, t, Zxx = scipy.signal.stft(sig, fs=fs, nperseg=256, noverlap=128)
    n_frames = Zxx.shape[1]

    r = np.abs(Zxx)
    theta = np.angle(Zxx)
    eps = 1e-12

    # Restringir análisis a la banda de interés 70-2000 Hz para CAS
    freq_mask = (f >= 70) & (f <= 2000)
    f_band = f[freq_mask]
    r_band = r[freq_mask, :]
    theta_band = theta[freq_mask, :]
    Zxx_band = Zxx[freq_mask, :]

    # 1. Índice Tonal (TI) frame-a-frame
    ti_seq = np.zeros(n_frames)
    for t_idx in range(2, n_frames):
        r_hat = 2 * r_band[:, t_idx-1] - r_band[:, t_idx-2]
        theta_hat = 2 * theta_band[:, t_idx-1] - theta_band[:, t_idx-2]
        Z_hat = r_hat * (np.cos(theta_hat) + 1j * np.sin(theta_hat))
        
        Z_act = Zxx_band[:, t_idx]
        r_act = r_band[:, t_idx]
        
        err = np.abs(Z_act - Z_hat) / (r_act + np.abs(r_hat) + eps)
        ti_seq[t_idx] = float(np.mean(np.clip(1.0 - err, 0.0, 1.0)))
    
    # Copiar TI de frames iniciales
    ti_seq[0] = ti_seq[2]
    ti_seq[1] = ti_seq[2]

    # 2. Entropía de Picos, 3. Curtosis y 4. Ratio f50/f90 frame-a-frame
    spe_seq = np.zeros(n_frames)
    sk_seq = np.zeros(n_frames)
    ratio_seq = np.zeros(n_frames)

    for t_idx in range(n_frames):
        psd = r_band[:, t_idx] ** 2
        total_p = np.sum(psd)
        if total_p < eps:
            continue
        
        p = psd / total_p

        # Entropía de Picos Espectrales (SPE)
        peaks, _ = scipy.signal.find_peaks(psd)
        if len(peaks) > 0:
            peak_p = psd[peaks] / (np.sum(psd[peaks]) + eps)
            spe_seq[t_idx] = float(-np.sum(peak_p * np.log2(peak_p + eps)))
        else:
            spe_seq[t_idx] = 0.0

        # Curtosis Espectral (SK)
        mean_f = np.sum(f_band * p)
        var_f = np.sum(((f_band - mean_f) ** 2) * p)
        if var_f > eps:
            sk_seq[t_idx] = float(np.sum(((f_band - mean_f) ** 4) * p) / (var_f ** 2))
        else:
            sk_seq[t_idx] = 3.0  # valor para distribución gaussiana

        # Ratio f50/f90
        cum_p = np.cumsum(p)
        idx_50 = np.searchsorted(cum_p, 0.50)
        idx_90 = np.searchsorted(cum_p, 0.90)
        idx_50 = min(idx_50, len(f_band) - 1)
        idx_90 = min(idx_90, len(f_band) - 1)
        f50 = f_band[idx_50]
        f90 = f_band[idx_90]
        if f90 > eps:
            ratio_seq[t_idx] = float(f50 / f90)
        else:
            ratio_seq[t_idx] = 0.0

    return np.column_stack([ti_seq, spe_seq, sk_seq, ratio_seq])


def extract_sota_features_global(sig: np.ndarray, fs: int = FS_TARGET) -> np.ndarray:
    """Extrae las 4 características físicas globales (media temporal de la secuencia)."""
    seq = extract_sota_features_sequence(sig, fs)
    return np.mean(seq, axis=0)


# ===========================================================================
# 2. Mitigación del Sesgo del Paciente (Nullspace Projection)
# ===========================================================================

def compute_nullspace_projection(X: np.ndarray, y_subject: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de proyección hacia el espacio nulo de un clasificador lineal
    entrenado para identificar al sujeto (timbre/identidad).
    """
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y_subject)
    W = clf.coef_  # (n_subjects, n_features)
    null_space = scipy.linalg.null_space(W)
    if null_space.shape[1] == 0:
        return np.eye(X.shape[1])
    P = null_space @ null_space.T
    return P


# ===========================================================================
# 3. Modelos y LSTM
# ===========================================================================

if TORCH_AVAILABLE:
    class SequenceDataset(Dataset):
        def __init__(self, sequences: list[np.ndarray], labels: np.ndarray):
            self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
            self.labels = torch.tensor(labels, dtype=torch.float32)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.sequences[idx], self.labels[idx]

    def collate_fn(batch):
        sequences, labels = zip(*batch)
        lengths = [len(seq) for seq in sequences]
        padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        return padded_seqs, torch.tensor(lengths), torch.stack(labels)

    class LightweightLSTM(nn.Module):
        def __init__(self, input_dim=4, hidden_dim=16, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x, lengths):
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hn, _) = self.lstm(packed)
            # hn de forma (num_layers, batch, hidden_dim)
            out = self.fc(hn[-1])
            return out.squeeze(1)


# ===========================================================================
# 4. Flujo Principal del Pipeline
# ===========================================================================

def main():
    print("=== PIPELINE SOTA MINIMALISTA ===")
    eps = 1e-12
    
    # Intentar cargar señales etiquetadas desde el caché de step8
    cache_labeled_path = os.path.join(_PROJECT_ROOT, "outputs", "results", "step8", "labeled_signals.pkl")
    if os.path.exists(cache_labeled_path):
        print(f"Cargando 1923 señales etiquetadas desde el caché de step8...")
        with open(cache_labeled_path, "rb") as fh:
            data_cache = pickle.load(fh)
        signals_labeled = data_cache["signals"]
        y_labeled = data_cache["y"]
        groups_labeled = data_cache["groups"]
    else:
        print("Caché no encontrado. Reconstruyendo señales (step4)...")
        metadata = _load_metadata(METADATA_CSV)
        subjects = _build_subject_list(metadata)
        all_signals, v_subject, _, _, _, _ = build_dataset(subjects)
        
        # Cargar etiquetas de proy_labels.mat
        labels_mat = scipy.io.loadmat(os.path.join(_PROJECT_ROOT, "proy_labels.mat"), squeeze_me=True)
        labels = np.asarray(labels_mat["labels"]).ravel().astype(int)
        
        mask_labeled = (labels == 2) | (labels == 3)
        idx_labeled = np.where(mask_labeled)[0]
        signals_labeled = [all_signals[i] for i in idx_labeled]
        y_labeled = (labels[mask_labeled] == 2).astype(int)
        groups_labeled = v_subject[mask_labeled]

    # Extracción de características
    print("\nExtrayendo features minimalistas globales para el conjunto etiquetado...")
    t0 = time.time()
    X_labeled = np.array([extract_sota_features_global(sig) for sig in signals_labeled])
    print(f"Extracción completada en {time.time() - t0:.1f} segundos. Forma: {X_labeled.shape}")

    # Definir clasificadores estáticos
    models = {
        "Logistic Regression L1": LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", random_state=42),
        "Logistic Regression L2": LogisticRegression(penalty="l2", class_weight="balanced", random_state=42),
        "SVM Linear": SVC(kernel="linear", class_weight="balanced", probability=True, random_state=42),
        "SVM RBF": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42),
    }

    results = {name: {"y_true": [], "y_pred": [], "y_prob": [], "aucs": [], "accs": []} for name in models}
    if TORCH_AVAILABLE:
        print("PyTorch disponible. Habilitando modelo LSTM.")
        results["Lightweight LSTM"] = {"y_true": [], "y_pred": [], "y_prob": [], "aucs": [], "accs": []}
        
        # Extraer secuencias completas
        print("Extrayendo secuencias de features para LSTM...")
        X_labeled_seq = [extract_sota_features_sequence(sig) for sig in signals_labeled]

    loso = LeaveOneGroupOut()
    n_folds = loso.get_n_splits(X_labeled, y_labeled, groups_labeled)

    print(f"\nEjecutando validación cruzada LOSO ({n_folds} folds)...")
    for fold_idx, (train_idx, test_idx) in enumerate(loso.split(X_labeled, y_labeled, groups_labeled), 1):
        X_train, X_test = X_labeled[train_idx], X_labeled[test_idx]
        y_train, y_test = y_labeled[train_idx], y_labeled[test_idx]
        g_train = groups_labeled[train_idx]
        
        # Mitigación del Sesgo: Proyección espacial
        P = compute_nullspace_projection(X_train, g_train)
        X_train_proj = X_train @ P
        X_test_proj = X_test @ P

        # Normalización estándar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_proj)
        X_test_scaled = scaler.transform(X_test_proj)

        # Entrenar clasificadores estáticos
        for name, clf in models.items():
            clf.fit(X_train_scaled, y_train)
            pred = clf.predict(X_test_scaled)
            prob = clf.predict_proba(X_test_scaled)[:, 1]
            
            results[name]["y_true"].extend(y_test)
            results[name]["y_pred"].extend(pred)
            results[name]["y_prob"].extend(prob)
            
            results[name]["accs"].append(accuracy_score(y_test, pred))
            try:
                results[name]["aucs"].append(roc_auc_score(y_test, prob))
            except ValueError:
                results[name]["aucs"].append(0.5)

        # Entrenar LSTM opcional
        if TORCH_AVAILABLE:
            train_seqs = [X_labeled_seq[i] for i in train_idx]
            test_seqs = [X_labeled_seq[i] for i in test_idx]
            
            # Aplicar proyección del espacio nulo y normalización frame-a-frame
            train_seqs_proj = [seq @ P for seq in train_seqs]
            test_seqs_proj = [seq @ P for seq in test_seqs]
            
            # Ajustar scaler en frames concatenados para evitar distorsiones
            all_frames_tr = np.vstack(train_seqs_proj)
            scaler_lstm = StandardScaler()
            scaler_lstm.fit(all_frames_tr)
            
            train_seqs_scaled = [scaler_lstm.transform(seq) for seq in train_seqs_proj]
            test_seqs_scaled = [scaler_lstm.transform(seq) for seq in test_seqs_proj]

            train_ds = SequenceDataset(train_seqs_scaled, y_train)
            train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            lstm_model = LightweightLSTM().to(device)
            optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)
            # Calcular pesos de clase
            pos_weight = torch.tensor([sum(y_train == 0) / sum(y_train == 1)], dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            # Entrenamiento rápido (15 épocas)
            lstm_model.train()
            for epoch in range(15):
                for seqs_b, lens_b, labels_b in train_dl:
                    seqs_b, labels_b = seqs_b.to(device), labels_b.to(device)
                    optimizer.zero_grad()
                    out = lstm_model(seqs_b, lens_b)
                    loss = criterion(out, labels_b)
                    loss.backward()
                    optimizer.step()

            # Evaluación en test
            lstm_model.eval()
            test_ds = SequenceDataset(test_seqs_scaled, y_test)
            test_dl = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, collate_fn=collate_fn)
            
            with torch.no_grad():
                for seqs_b, lens_b, _ in test_dl:
                    seqs_b = seqs_b.to(device)
                    logits = lstm_model(seqs_b, lens_b)
                    probs = torch.sigmoid(logits).cpu().numpy()
            
            preds = (probs >= 0.5).astype(int)
            results["Lightweight LSTM"]["y_true"].extend(y_test)
            results["Lightweight LSTM"]["y_pred"].extend(preds)
            results["Lightweight LSTM"]["y_prob"].extend(probs)
            results["Lightweight LSTM"]["accs"].append(accuracy_score(y_test, preds))
            try:
                results["Lightweight LSTM"]["aucs"].append(roc_auc_score(y_test, probs))
            except ValueError:
                results["Lightweight LSTM"]["aucs"].append(0.5)

        print(f"  Fold {fold_idx}/{n_folds} procesado.")

    # Mostrar reporte de resultados
    metrics_summary = []
    print("\n=== RESUMEN MÉTRICAS LOSO ===")
    for name, res in results.items():
        y_t = np.array(res["y_true"])
        y_p = np.array(res["y_pred"])
        y_pr = np.array(res["y_prob"])
        
        acc = accuracy_score(y_t, y_p)
        sens = recall_score(y_t, y_p, zero_division=0)
        prec = precision_score(y_t, y_p, zero_division=0)
        f1 = f1_score(y_t, y_p, zero_division=0)
        auc = roc_auc_score(y_t, y_pr)
        
        # Especificidad
        tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"{name:25s} - AUC: {auc:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f}")
        metrics_summary.append({
            "model": name,
            "auc": float(auc),
            "f1": float(f1),
            "accuracy": float(acc),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "precision": float(prec),
        })

    # Guardar resumen de métricas
    with open(os.path.join(RESULTS_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)

    # Dibujar curvas ROC
    plt.figure(figsize=(8, 7))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(res['y_true'], res['y_prob']):.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - SOTA Minimalist Pipeline (LOSO)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_curves_loso.png"), dpi=150)
    plt.close()

    # Seleccionar el mejor modelo
    best_model_name = max(results.keys(), key=lambda n: roc_auc_score(results[n]["y_true"], results[n]["y_prob"]))
    print(f"\nMejor modelo seleccionado por AUC: {best_model_name}")

    # ===========================================================================
    # 5. Inferencia y Biomarcador Clínico
    # ===========================================================================
    print("\nCargando todas las 14 900 señales para la inferencia completa...")
    metadata = _load_metadata(METADATA_CSV)
    subjects = _build_subject_list(metadata)
    all_signals, v_subject, v_bd, v_channel, v_phase, _ = build_dataset(subjects)

    print("Extrayendo features minimalistas para las 14 900 señales...")
    t0 = time.time()
    X_all = np.array([extract_sota_features_global(sig) for sig in all_signals])
    print(f"Extracción de las 14 900 señales completada en {time.time() - t0:.1f} segundos.")

    # Re-entrenamiento final del mejor modelo sobre todo el conjunto etiquetado
    P_final = compute_nullspace_projection(X_labeled, groups_labeled)
    X_labeled_proj = X_labeled @ P_final
    X_all_proj = X_all @ P_final

    scaler_final = StandardScaler()
    X_labeled_scaled = scaler_final.fit_transform(X_labeled_proj)
    X_all_scaled = scaler_final.transform(X_all_proj)

    if best_model_name == "Lightweight LSTM" and TORCH_AVAILABLE:
        # Re-entrenar LSTM final
        print("Re-entrenando LSTM final...")
        X_all_seq = [extract_sota_features_sequence(sig) for sig in all_signals]
        X_labeled_seq_proj = [seq @ P_final for seq in X_labeled_seq]
        X_all_seq_proj = [seq @ P_final for seq in X_all_seq]
        
        all_frames_lbl = np.vstack(X_labeled_seq_proj)
        scaler_lstm_final = StandardScaler()
        scaler_lstm_final.fit(all_frames_lbl)
        
        X_labeled_seq_scaled = [scaler_lstm_final.transform(seq) for seq in X_labeled_seq_proj]
        X_all_seq_scaled = [scaler_lstm_final.transform(seq) for seq in X_all_seq_proj]
        
        train_ds_f = SequenceDataset(X_labeled_seq_scaled, y_labeled)
        train_dl_f = DataLoader(train_ds_f, batch_size=32, shuffle=True, collate_fn=collate_fn)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        final_lstm = LightweightLSTM().to(device)
        optimizer_f = optim.Adam(final_lstm.parameters(), lr=0.01)
        pos_weight_f = torch.tensor([sum(y_labeled == 0) / sum(y_labeled == 1)], dtype=torch.float32).to(device)
        criterion_f = nn.BCEWithLogitsLoss(pos_weight=pos_weight_f)
        
        final_lstm.train()
        for epoch in range(15):
            for seqs_b, lens_b, labels_b in train_dl_f:
                seqs_b, labels_b = seqs_b.to(device), labels_b.to(device)
                optimizer_f.zero_grad()
                out = final_lstm(seqs_b, lens_b)
                loss = criterion_f(out, labels_b)
                loss.backward()
                optimizer_f.step()
                
        # Inferencia
        final_lstm.eval()
        all_ds = SequenceDataset(X_all_seq_scaled, np.zeros(len(all_signals)))
        all_dl = DataLoader(all_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
        
        y_prob_all = []
        with torch.no_grad():
            for seqs_b, lens_b, _ in all_dl:
                seqs_b = seqs_b.to(device)
                logits = final_lstm(seqs_b, lens_b)
                probs = torch.sigmoid(logits).cpu().numpy()
                y_prob_all.extend(probs.tolist())
        y_prob_all = np.array(y_prob_all)
        y_pred_all = (y_prob_all >= 0.5).astype(int)
    else:
        # Si el mejor modelo no es la LSTM (o no está disponible)
        clf_final = models[best_model_name]
        y_pred_all = clf_final.predict(X_all_scaled)
        y_prob_all = clf_final.predict_proba(X_all_scaled)[:, 1]

    # Guardar predictions_all.npz para análisis posteriores
    np.savez(
        os.path.join(RESULTS_DIR, "predictions_all.npz"),
        best_model_name=best_model_name,
        y_prob_all=y_prob_all,
        y_pred_all=y_pred_all
    )

    # Calcular biomarcadores
    print("\nCalculando biomarcador clínico (Tasa de CAS y Delta CAS)...")
    subjects_metadata = pd.read_csv(METADATA_CSV)
    
    # Mapear de número de sujeto (1-28) a subject_id (P1-P23, C1-C5)
    subject_mapping = {}
    for idx, row in subjects_metadata.iterrows():
        sid = row["subject_id"]
        # build_dataset indexa 1..23 pacientes, 24..28 controles
        if sid.startswith("P"):
            subject_mapping[int(sid[1:])] = (sid, row["bdr_label"], "patient")
        else:
            subject_mapping[int(sid[1:]) + 23] = (sid, row["bdr_label"], "control")

    patient_results = []
    
    # Procesar solo pacientes (1 a 23)
    for subj_num in range(1, 24):
        subj_id, bdr_label, s_type = subject_mapping[subj_num]
        
        # Filtros por sujeto
        mask_subj = (v_subject == subj_num)
        mask_pre = mask_subj & (v_bd == 1)
        mask_post = mask_subj & (v_bd == 2)
        
        # Total de segmentos por fase
        total_pre = np.sum(mask_pre)
        total_post = np.sum(mask_post)
        
        # Predicciones CAS
        cas_pre = np.sum(y_pred_all[mask_pre])
        cas_post = np.sum(y_pred_all[mask_post])
        
        # Tasa de CAS
        rate_pre = (cas_pre / total_pre) if total_pre > 0 else 0.0
        rate_post = (cas_post / total_post) if total_post > 0 else 0.0
        
        # Delta CAS (%)
        delta_cas = ((rate_post - rate_pre) / (rate_pre + eps)) * 100.0 if rate_pre > 0 else 0.0
        
        patient_results.append({
            "subject_id": subj_id,
            "bdr_label": bdr_label,
            "cas_rate_pre": rate_pre,
            "cas_rate_post": rate_post,
            "delta_cas_percent": delta_cas
        })
        
    df_results = pd.DataFrame(patient_results)
    df_results.to_csv(os.path.join(RESULTS_DIR, "clinical_biomarker_results.csv"), index=False)
    print("Resultados de biomarcador guardados.")

    # Prueba estadística Mann-Whitney U
    from scipy.stats import mannwhitneyu
    
    # Filtrar BDR+ y BDR- (excluyendo controles, ya que solo procesamos P1-P23)
    responders = df_results[df_results["bdr_label"] == "BDR+"]["delta_cas_percent"].values
    non_responders = df_results[df_results["bdr_label"] == "BDR-"]["delta_cas_percent"].values
    
    stat, p_val = mannwhitneyu(responders, non_responders, alternative="two-sided")
    print(f"\n--- Prueba Mann-Whitney U para Delta CAS ---")
    print(f"Respondedores (BDR+) Delta CAS: {np.mean(responders):.2f}% ± {np.std(responders):.2f}%")
    print(f"No Respondedores (BDR-) Delta CAS: {np.mean(non_responders):.2f}% ± {np.std(non_responders):.2f}%")
    print(f"U-statistic: {stat} | p-value: {p_val:.5f}")
    
    with open(os.path.join(RESULTS_DIR, "statistical_test.json"), "w", encoding="utf-8") as f:
        json.dump({
            "u_statistic": float(stat),
            "p_value": float(p_val),
            "mean_bdr_positive": float(np.mean(responders)),
            "mean_bdr_negative": float(np.mean(non_responders)),
        }, f, indent=4)

    # Boxplot del Delta CAS por grupo BDR
    plt.figure(figsize=(6, 6))
    sns.boxplot(data=df_results, x="bdr_label", y="delta_cas_percent", palette={"BDR+": "mediumseagreen", "BDR-": "steelblue"})
    sns.stripplot(data=df_results, x="bdr_label", y="delta_cas_percent", color="black", alpha=0.5, size=6, jitter=0.2)
    plt.title(f"Delta CAS (%) según Respuesta Broncodilatadora\np-value = {p_val:.4f}")
    plt.xlabel("Grupo Clínico (BDR)")
    plt.ylabel("Delta CAS (%)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "delta_cas_boxplot.png"), dpi=150)
    plt.close()
    
    print("\nPipeline finalizado con éxito. Archivos generados en:")
    print(f"  Resultados: {RESULTS_DIR}")
    print(f"  Figuras: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
