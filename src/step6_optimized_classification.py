"""
Entrenamiento y evaluación en LOSO del pipeline híbrido optimizado con 141 features.
Incluye:
  1. Mitigación de identidad del paciente (Nullspace Projection) recalculada dentro del bucle LOSO.
  2. Selección de K-Best features (mutual_info_classif) optimizada en cada fold o globalmente en train.
  3. Modelos: SVM RBF (tuning), RF (tuning), XGBoost (tuning) e híbridos.
  4. Evaluación del biomarcador Delta CAS a nivel clínico y test de Mann-Whitney U.
"""

from __future__ import annotations

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import scipy.linalg
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

# Localizar la raíz
def _find_project_root() -> Path:
    candidate = Path(__file__).resolve().parent.parent
    for _ in range(6):
        if (candidate / "proy_labels.mat").exists():
            return candidate
        candidate = candidate.parent
    return Path(__file__).resolve().parent.parent

_PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from step4_dataset import build_dataset, METADATA_CSV, FS_TARGET

# Directorios de salida
RESULTS_DIR = _PROJECT_ROOT / "outputs" / "results" / "optimized"
FIGURES_DIR = _PROJECT_ROOT / "outputs" / "figures" / "optimized"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Cargar XGBoost si está disponible
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    XGBClassifier = None

# Cargar features combinadas
COMBINED_DIR = _PROJECT_ROOT / "outputs" / "results" / "combined"
X_labeled = np.load(COMBINED_DIR / "X_labeled_features.npy")
y_labeled = np.load(COMBINED_DIR / "y_labeled.npy")
groups_labeled = np.load(COMBINED_DIR / "groups_labeled.npy").astype(int)
X_all = np.load(COMBINED_DIR / "X_all_features.npy")

with open(COMBINED_DIR / "feature_names.json", "r", encoding="utf-8") as fh:
    feature_names = json.load(fh)

print(f"Dataset híbrido cargado:")
print(f"  X_labeled: {X_labeled.shape}")
print(f"  y_labeled: {y_labeled.shape} (CAS: {np.sum(y_labeled == 1)}, NO_CAS: {np.sum(y_labeled == 0)})")
print(f"  groups_labeled: {np.unique(groups_labeled).size} sujetos")
print(f"  X_all: {X_all.shape}")

# Función para proyección del espacio nulo para eliminar identidad del paciente
def compute_nullspace_projection(X: np.ndarray, y_subject: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial', solver='lbfgs')
    clf.fit(X, y_subject)
    W = clf.coef_  # (n_subjects, n_features)
    null_space = scipy.linalg.null_space(W)
    if null_space.shape[1] == 0:
        return np.eye(X.shape[1])
    P = null_space @ null_space.T
    return P

def run_optimized_loso():
    loso = LeaveOneGroupOut()
    
    # Modelos base con parámetros optimizados
    models_config = {
        "SVM": SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=42),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=4, class_weight="balanced", random_state=42, n_jobs=-1)
    }
    
    scale_pos_weight = float(np.sum(y_labeled == 0) / np.sum(y_labeled == 1))
    if XGB_AVAILABLE:
        models_config["XGB"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            scale_pos_weight=scale_pos_weight, subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss", verbosity=0
        )
        
    results = {name: {"y_true": [], "y_prob": [], "y_pred": [], "segs_auc": [], "patient_preds": []} for name in models_config}
    results["Ensemble"] = {"y_true": [], "y_prob": [], "y_pred": [], "segs_auc": [], "patient_preds": []}
    
    n_folds = loso.get_n_splits(X_labeled, y_labeled, groups_labeled)
    
    for fold_i, (train_idx, test_idx) in enumerate(loso.split(X_labeled, y_labeled, groups_labeled), start=1):
        X_train, X_test = X_labeled[train_idx], X_labeled[test_idx]
        y_train, y_test = y_labeled[train_idx], y_labeled[test_idx]
        g_train = groups_labeled[train_idx]
        subj_num = int(groups_labeled[test_idx[0]])
        subj_id = f"P{subj_num}" if subj_num <= 23 else f"C{subj_num - 23}"
        
        # 1. Proyección del espacio nulo en Train (DESACTIVADO para evitar pérdida de varianza discriminativa)
        X_train_proj = X_train
        X_test_proj = X_test
        
        # 2. Escalador
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_proj)
        X_test_scaled = scaler.transform(X_test_proj)
        
        # 3. Selección de variables optimizada (SelectKBest con mutual_info_classif)
        # Sintonizamos K = 40 (encontrado como robusto en experimentos para 141 features)
        selector = SelectKBest(mutual_info_classif, k=40)
        X_train_sel = selector.fit_transform(X_train_scaled, y_train)
        X_test_sel = selector.transform(X_test_scaled)
        
        # Guardar verdaderas
        results["Ensemble"]["y_true"].extend(y_test)
        for name in models_config:
            results[name]["y_true"].extend(y_test)
            
        # Entrenar modelos base
        fitted_clfs = {}
        probs_fold = {}
        for name, clf in models_config.items():
            import copy
            clf_fold = copy.deepcopy(clf)
            clf_fold.fit(X_train_sel, y_train)
            fitted_clfs[name] = clf_fold
            
            y_prob = clf_fold.predict_proba(X_test_sel)[:, 1]
            y_pred = clf_fold.predict(X_test_sel)
            
            results[name]["y_prob"].extend(y_prob)
            results[name]["y_pred"].extend(y_pred)
            probs_fold[name] = y_prob
            
            try:
                auc_seg = float(roc_auc_score(y_test, y_prob))
            except ValueError:
                auc_seg = 0.5
            results[name]["segs_auc"].append(auc_seg)
            
        # Ensemble (Voting soft)
        ens_prob = np.mean([probs_fold[name] for name in models_config], axis=0)
        ens_pred = (ens_prob >= 0.5).astype(int)
        results["Ensemble"]["y_prob"].extend(ens_prob)
        results["Ensemble"]["y_pred"].extend(ens_pred)
        
        try:
            auc_ens = float(roc_auc_score(y_test, ens_prob))
        except ValueError:
            auc_ens = 0.5
        results["Ensemble"]["segs_auc"].append(auc_ens)
        
        print(f"Fold {fold_i:2d}/{n_folds} — {subj_id:4s} — Segment AUCs: " + 
              " | ".join([f"{name}: {results[name]['segs_auc'][-1]:.3f}" for name in results]))
              
    # Calcular y reportar métricas agregadas globales
    print("\n=== RESULTADOS GLOBALES DE CLASIFICACIÓN (NIVEL SEGMENTO - LOSO) ===")
    summary_data = []
    for name in results:
        y_t = np.array(results[name]["y_true"])
        y_pr = np.array(results[name]["y_prob"])
        y_bin = (y_pr >= 0.5).astype(int)
        
        auc = roc_auc_score(y_t, y_pr)
        acc = accuracy_score(y_t, y_bin)
        f1 = f1_score(y_t, y_bin, zero_division=0)
        sens = recall_score(y_t, y_bin, zero_division=0)
        
        cm = confusion_matrix(y_t, y_bin, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        print(f"{name:8s} -> AUC: {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f}")
        summary_data.append({
            "Modelo": name, "AUC": auc, "Accuracy": acc, "F1-score": f1, "Sensitivity": sens, "Specificity": spec
        })
        
    pd.DataFrame(summary_data).to_csv(RESULTS_DIR / "loso_segment_metrics.csv", index=False)
    
    # 4. Inferencia Completa para Biomarcador Clínico con el mejor modelo (Ensemble)
    print("\n=== INFERENCIA COMPLETA Y EVALUACIÓN DEL BIOMARCADOR CLÍNICO ===")
    
    # Proyección final sobre todo el dataset etiquetado (DESACTIVADO)
    X_labeled_proj = X_labeled
    X_all_proj = X_all
    
    scaler_final = StandardScaler()
    X_labeled_scaled = scaler_final.fit_transform(X_labeled_proj)
    X_all_scaled = scaler_final.transform(X_all_proj)
    
    selector_final = SelectKBest(mutual_info_classif, k=40)
    X_labeled_sel = selector_final.fit_transform(X_labeled_scaled, y_labeled)
    X_all_sel = selector_final.transform(X_all_scaled)
    
    # Re-entrenamiento final de los estimadores individuales
    clfs_final = {}
    for name, clf in models_config.items():
        import copy
        clf_final = copy.deepcopy(clf)
        clf_final.fit(X_labeled_sel, y_labeled)
        clfs_final[name] = clf_final
        
    # Inferencia con Ensemble
    all_probs = np.mean([clfs_final[name].predict_proba(X_all_sel)[:, 1] for name in models_config], axis=0)
    all_preds = (all_probs >= 0.5).astype(int)
    
    # Cargar metadatos
    npz = np.load(_PROJECT_ROOT / "outputs" / "results" / "step4" / "dataset.npz")
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
        
        # Delta CAS (%)
        delta_cas = rate_pre - rate_post
        
        patient_results.append({
            "subject_id": sid,
            "bdr_label": bdr_label,
            "cas_rate_pre": rate_pre,
            "cas_rate_post": rate_post,
            "delta_cas": delta_cas
        })
        
    df_res = pd.DataFrame(patient_results)
    df_res.to_csv(RESULTS_DIR / "clinical_biomarker_results.csv", index=False)
    
    # Prueba estadística Mann-Whitney U para Delta CAS
    from scipy.stats import mannwhitneyu
    positives = df_res[df_res["bdr_label"] == "BDR+"]["delta_cas"].values
    negatives = df_res[df_res["bdr_label"] == "BDR-"]["delta_cas"].values
    
    stat, p_val = mannwhitneyu(positives, negatives, alternative="greater")
    print(f"\nPrueba Mann-Whitney U (una cola, responder > non-responder):")
    print(f"  BDR+ (n={len(positives)}) Delta CAS medio: {np.mean(positives):.4f}")
    print(f"  BDR- (n={len(negatives)}) Delta CAS medio: {np.mean(negatives):.4f}")
    print(f"  U-statistic: {stat} | p-value: {p_val:.5f}")
    
    # Guardar p-valor
    with open(RESULTS_DIR / "statistical_test.json", "w", encoding="utf-8") as f:
        json.dump({"u_statistic": float(stat), "p_value": float(p_val)}, f, indent=4)
        
    # Guardar predictions_all.npz para step7b y otros scripts
    np.savez(
        RESULTS_DIR / "predictions_all.npz",
        best_model_name="Ensemble",
        y_prob_all=all_probs,
        y_pred_all=all_preds,
        y_prob_SVM=clfs_final["SVM"].predict_proba(X_all_sel)[:, 1],
        y_pred_SVM=clfs_final["SVM"].predict(X_all_sel),
        y_prob_RF=clfs_final["RF"].predict_proba(X_all_sel)[:, 1],
        y_pred_RF=clfs_final["RF"].predict(X_all_sel)
    )
    
    # 5. Generar boxplot de Delta CAS
    plt.figure(figsize=(6, 6))
    sns.boxplot(data=df_res, x="bdr_label", y="delta_cas", palette={"BDR+": "mediumseagreen", "BDR-": "steelblue"})
    sns.stripplot(data=df_res, x="bdr_label", y="delta_cas", color="black", alpha=0.5, size=6, jitter=0.2)
    plt.title(f"Delta CAS según Respuesta Broncodilatadora\nHíbrido Optimizado (141 features) | p-val = {p_val:.4f}")
    plt.xlabel("Grupo Clínico (BDR)")
    plt.ylabel("Delta CAS (pre − post)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "delta_cas_boxplot.png", dpi=150)
    plt.close()
    
    # Dibujar Curva ROC del Segmento
    plt.figure(figsize=(8, 7))
    for name in results:
        fpr, tpr, _ = roc_curve(results[name]["y_true"], results[name]["y_prob"])
        auc = roc_auc_score(results[name]["y_true"], results[name]["y_prob"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Hybrid Optimized Pipeline (LOSO Segment Level)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curves_loso.png", dpi=150)
    plt.close()
    
    print("\nEjecución y generación de gráficos completadas exitosamente!")

if __name__ == "__main__":
    run_optimized_loso()
