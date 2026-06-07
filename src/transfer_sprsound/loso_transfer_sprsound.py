import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import scipy.linalg
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

# Localizar la raíz
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from step4_dataset import build_dataset, METADATA_CSV, FS_TARGET
from step8_deep_learning import signal_to_spectrogram, _compute_metrics

# Directorios de salida
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results" / "transfer_sprsound"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "transfer_sprsound"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Cargar XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Cargar PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from step8_deep_learning import CAS_CNN
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Cargar features locales
COMBINED_DIR = PROJECT_ROOT / "outputs" / "results" / "combined"
X_labeled = np.load(COMBINED_DIR / "X_labeled_features.npy")
y_labeled = np.load(COMBINED_DIR / "y_labeled.npy")
groups_labeled = np.load(COMBINED_DIR / "groups_labeled.npy").astype(int)
X_all = np.load(COMBINED_DIR / "X_all_features.npy")

with open(COMBINED_DIR / "feature_names.json", "r", encoding="utf-8") as fh:
    feature_names = json.load(fh)

def run_sprsound_loso():
    loso = LeaveOneGroupOut()
    n_folds = loso.get_n_splits(X_labeled, y_labeled, groups_labeled)
    
    # 1. Cargar modelos y matrices de proyección
    xgb_pretrained = None
    if XGB_AVAILABLE:
        xgb_path = RESULTS_DIR / "xgb_sprsound_pretrained.pkl"
        if xgb_path.exists():
            with open(xgb_path, "rb") as f:
                xgb_pretrained = pickle.load(f)
            print("Modelo XGBoost preentrenado de SPRSound cargado.")
            
    P_subject = np.eye(X_labeled.shape[1])
    proj_path = RESULTS_DIR / "subject_projection.npy"
    if proj_path.exists():
        P_subject = np.load(proj_path)
        print(f"Matriz de proyección del espacio nulo (sujeto) cargada (Shape: {P_subject.shape}).")
        
    # Aplicar proyección + estandarización para mitigar Domain Shift en features locales
    scaler = StandardScaler()
    X_labeled_scaled = scaler.fit_transform(X_labeled)
    X_all_scaled = scaler.transform(X_all)
    X_labeled_proj = X_labeled_scaled @ P_subject
    X_all_proj = X_all_scaled @ P_subject
    
    # Inicializar contenedores de resultados
    results = {
        "XGB_SPRSound": {"y_true": [], "y_prob": [], "y_pred": [], "segs_auc": []},
        "CNN_SPRSound": {"y_true": [], "y_prob": [], "y_pred": [], "segs_auc": []},
        "Ensemble_SPRSound": {"y_true": [], "y_prob": [], "y_pred": [], "segs_auc": []}
    }
    
    X_spectro = None
    if TORCH_AVAILABLE:
        spectro_path = PROJECT_ROOT / "outputs" / "results" / "step8" / "X_spectrograms.npy"
        if spectro_path.exists():
            X_spectro = np.load(spectro_path)
            print(f"Espectrogramas locales cargados: {X_spectro.shape}")
            
    print("\nEjecutando LOSO con Transfer Learning (SPRSound)...")
    
    for fold_i, (train_idx, test_idx) in enumerate(loso.split(X_labeled_proj, y_labeled, groups_labeled), start=1):
        X_train, X_test = X_labeled_proj[train_idx], X_labeled_proj[test_idx]
        y_train, y_test = y_labeled[train_idx], y_labeled[test_idx]
        subj_num = int(groups_labeled[test_idx[0]])
        subj_id = f"P{subj_num}" if subj_num <= 23 else f"C{subj_num - 23}"
        
        # Guardar verdaderas
        for name in results:
            results[name]["y_true"].extend(y_test)
            
        probs_fold = {}
        
        # --- A. XGBoost con transferencia + Mitigación de Domain Shift ---
        if XGB_AVAILABLE and xgb_pretrained is not None:
            xgb_fold = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.01,
                random_state=42, eval_metric="logloss", verbosity=0
            )
            # Entrenar estimadores adicionales con fine-tuning
            xgb_fold.fit(X_train, y_train, xgb_model=xgb_pretrained.get_booster())
            
            y_prob_xgb = xgb_fold.predict_proba(X_test)[:, 1]
            y_pred_xgb = xgb_fold.predict(X_test)
            
            results["XGB_SPRSound"]["y_prob"].extend(y_prob_xgb)
            results["XGB_SPRSound"]["y_pred"].extend(y_pred_xgb)
            probs_fold["XGB_SPRSound"] = y_prob_xgb
            
            auc_seg = float(roc_auc_score(y_test, y_prob_xgb)) if len(np.unique(y_test)) > 1 else 0.5
            results["XGB_SPRSound"]["segs_auc"].append(auc_seg)
            
        # --- B. CNN con transferencia (Fine-tuning) ---
        if TORCH_AVAILABLE and X_spectro is not None:
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cnn_pretrained_path = RESULTS_DIR / "cnn_sprsound_pretrained.pt"
            
            model = CAS_CNN()
            if cnn_pretrained_path.exists():
                model.load_state_dict(torch.load(cnn_pretrained_path, weights_only=True))
            model = model.to(torch_device)
            
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            pos_weight = torch.tensor([np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)], dtype=torch.float32).to(torch_device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            X_tr_spec = torch.tensor(X_spectro[train_idx], dtype=torch.float32).unsqueeze(1)
            y_tr_spec = torch.tensor(y_train, dtype=torch.float32)
            
            ds = TensorDataset(X_tr_spec, y_tr_spec)
            dl = DataLoader(ds, batch_size=32, shuffle=True)
            
            model.train()
            for epoch in range(5):
                for Xb, yb in dl:
                    Xb, yb = Xb.to(torch_device), yb.to(torch_device)
                    optimizer.zero_grad()
                    logits = model(Xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    
            model.eval()
            X_te_spec = torch.tensor(X_spectro[test_idx], dtype=torch.float32).unsqueeze(1).to(torch_device)
            with torch.no_grad():
                logits_te = model(X_te_spec)
                probs_te = torch.sigmoid(logits_te).cpu().numpy()
            preds_te = (probs_te >= 0.5).astype(int)
            
            results["CNN_SPRSound"]["y_prob"].extend(probs_te)
            results["CNN_SPRSound"]["y_pred"].extend(preds_te)
            probs_fold["CNN_SPRSound"] = probs_te
            
            auc_seg = float(roc_auc_score(y_test, probs_te)) if len(np.unique(y_test)) > 1 else 0.5
            results["CNN_SPRSound"]["segs_auc"].append(auc_seg)
            
        # --- C. Ensemble ---
        if len(probs_fold) > 0:
            ens_prob = np.mean(list(probs_fold.values()), axis=0)
            ens_pred = (ens_prob >= 0.5).astype(int)
            results["Ensemble_SPRSound"]["y_prob"].extend(ens_prob)
            results["Ensemble_SPRSound"]["y_pred"].extend(ens_pred)
            
            auc_ens = float(roc_auc_score(y_test, ens_prob)) if len(np.unique(y_test)) > 1 else 0.5
            results["Ensemble_SPRSound"]["segs_auc"].append(auc_ens)
            
        print(f"Fold {fold_i:2d}/{n_folds} — {subj_id:4s} — Segment AUCs: " + 
              " | ".join([f"{name}: {results[name]['segs_auc'][-1]:.3f}" for name in results if len(results[name]['segs_auc']) > 0]))
              
    # Métricas agregadas globales
    print("\n=== RESULTADOS GLOBALES CON SPRSOUND (NIVEL SEGMENTO - LOSO) ===")
    summary_data = []
    for name in results:
        if len(results[name]["y_prob"]) == 0:
            continue
        y_t = np.array(results[name]["y_true"])
        y_pr = np.array(results[name]["y_prob"])
        y_bin = np.array(results[name]["y_pred"])
        
        auc = roc_auc_score(y_t, y_pr)
        acc = accuracy_score(y_t, y_bin)
        f1 = f1_score(y_t, y_bin, zero_division=0)
        sens = recall_score(y_t, y_bin, zero_division=0)
        
        cm = confusion_matrix(y_t, y_bin, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        print(f"{name:18s} -> AUC: {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f}")
        summary_data.append({
            "Modelo": name, "AUC": auc, "Accuracy": acc, "F1-score": f1, "Sensitivity": sens, "Specificity": spec
        })
        
    pd.DataFrame(summary_data).to_csv(RESULTS_DIR / "loso_segment_metrics_sprsound.csv", index=False)
    
    # Inferencia Completa y Biomarcador
    print("\n=== INFERENCIA COMPLETA Y EVALUACIÓN DEL BIOMARCADOR CLÍNICO ===")
    
    npz = np.load(PROJECT_ROOT / "outputs" / "results" / "step4" / "dataset.npz")
    v_subject = npz["v_subject"].astype(int)
    v_bd = npz["v_bd"].astype(int)
    
    subjects_metadata = pd.read_csv(METADATA_CSV)
    subject_mapping = {}
    for idx, row in subjects_metadata.iterrows():
        sid = row["subject_id"]
        if sid.startswith("P"):
            subject_mapping[int(sid[1:])] = (sid, row["bdr_label"])
        else:
            subject_mapping[int(sid[1:]) + 23] = (sid, row["bdr_label"])
            
    # Re-entrenamiento del ensemble final sobre todo el dataset local proyectado
    final_probs = []
    
    if XGB_AVAILABLE and xgb_pretrained is not None:
        xgb_final = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.01,
            random_state=42, eval_metric="logloss", verbosity=0
        )
        xgb_final.fit(X_labeled_proj, y_labeled, xgb_model=xgb_pretrained.get_booster())
        prob_xgb_all = xgb_final.predict_proba(X_all_proj)[:, 1]
        final_probs.append(prob_xgb_all)
        
    if TORCH_AVAILABLE:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn_path = RESULTS_DIR / "cnn_sprsound_pretrained.pt"
        model_final = CAS_CNN()
        if cnn_path.exists():
            model_final.load_state_dict(torch.load(cnn_path, weights_only=True))
        model_final = model_final.to(torch_device)
        
        spectrograms_all_path = PROJECT_ROOT / "outputs" / "results" / "step8" / "X_spectrograms_all14900.npy"
        if spectrograms_all_path.exists():
            X_all_spec = np.load(spectrograms_all_path)
            
            optimizer = optim.Adam(model_final.parameters(), lr=1e-4)
            pos_weight = torch.tensor([np.sum(y_labeled == 0) / max(np.sum(y_labeled == 1), 1)], dtype=torch.float32).to(torch_device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            X_tr_spec = torch.tensor(X_spectro, dtype=torch.float32).unsqueeze(1)
            y_tr_spec = torch.tensor(y_labeled, dtype=torch.float32)
            ds = TensorDataset(X_tr_spec, y_tr_spec)
            dl = DataLoader(ds, batch_size=32, shuffle=True)
            
            model_final.train()
            for epoch in range(5):
                for Xb, yb in dl:
                    Xb, yb = Xb.to(torch_device), yb.to(torch_device)
                    optimizer.zero_grad()
                    logits = model_final(Xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    
            model_final.eval()
            X_all_spec_t = torch.tensor(X_all_spec, dtype=torch.float32).unsqueeze(1).to(torch_device)
            with torch.no_grad():
                logits_all = model_final(X_all_spec_t)
                prob_cnn_all = torch.sigmoid(logits_all).cpu().numpy()
            final_probs.append(prob_cnn_all)
            
    if len(final_probs) > 0:
        all_probs = np.mean(final_probs, axis=0)
        all_preds = (all_probs >= 0.5).astype(int)
    else:
        print("Error: Modelos no disponibles.")
        return
        
    patient_results = []
    for subj_num in range(1, 24):
        sid, bdr_label = subject_mapping[subj_num]
        mask_subj = (v_subject == subj_num)
        mask_pre = mask_subj & (v_bd == 1)
        mask_post = mask_subj & (v_bd == 2)
        
        t_pre = np.sum(mask_pre)
        t_post = np.sum(mask_post)
        
        rate_pre = np.mean(all_preds[mask_pre]) if t_pre > 0 else 0.0
        rate_post = np.mean(all_preds[mask_post]) if t_post > 0 else 0.0
        
        delta_cas = rate_pre - rate_post
        
        patient_results.append({
            "subject_id": sid,
            "bdr_label": bdr_label,
            "cas_rate_pre": rate_pre,
            "cas_rate_post": rate_post,
            "delta_cas": delta_cas
        })
        
    df_res = pd.DataFrame(patient_results)
    df_res.to_csv(RESULTS_DIR / "clinical_biomarker_results_sprsound.csv", index=False)
    
    from scipy.stats import mannwhitneyu
    positives = df_res[df_res["bdr_label"] == "BDR+"]["delta_cas"].values
    negatives = df_res[df_res["bdr_label"] == "BDR-"]["delta_cas"].values
    
    stat, p_val = mannwhitneyu(positives, negatives, alternative="greater")
    print(f"\nPrueba Mann-Whitney U para Delta CAS con SPRSound:")
    print(f"  BDR+ (n={len(positives)}) Delta CAS medio: {np.mean(positives):.4f}")
    print(f"  BDR- (n={len(negatives)}) Delta CAS medio: {np.mean(negatives):.4f}")
    print(f"  U-statistic: {stat} | p-value: {p_val:.5f}")
    
    with open(RESULTS_DIR / "statistical_test_sprsound.json", "w", encoding="utf-8") as f:
        json.dump({"u_statistic": float(stat), "p_value": float(p_val)}, f, indent=4)
        
    # Generar gráficos
    plt.figure(figsize=(6, 6))
    sns.boxplot(data=df_res, x="bdr_label", y="delta_cas", palette={"BDR+": "mediumseagreen", "BDR-": "steelblue"})
    sns.stripplot(data=df_res, x="bdr_label", y="delta_cas", color="black", alpha=0.5, size=6, jitter=0.2)
    plt.title(f"Delta CAS (SPRSound Transfer)\np-value = {p_val:.5f}")
    plt.xlabel("Grupo Clínico (BDR)")
    plt.ylabel("Delta CAS (pre − post)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "delta_cas_boxplot_sprsound.png", dpi=150)
    plt.close()
    
    plt.figure(figsize=(8, 7))
    for name in results:
        if len(results[name]["y_prob"]) == 0:
            continue
        fpr, tpr, _ = roc_curve(results[name]["y_true"], results[name]["y_prob"])
        auc = roc_auc_score(results[name]["y_true"], results[name]["y_prob"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - SPRSound Transfer Learning (LOSO)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves_loso_sprsound.png", dpi=150)
    plt.close()
    
    print("\nEvaluación completada con éxito!")

if __name__ == "__main__":
    run_sprsound_loso()
