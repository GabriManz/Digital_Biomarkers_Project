"""
Fase 3.1 — Delta_CAS como biomarcador primario de BDR.

Construye un dataset a nivel de PACIENTE (18 filas) usando:
  - Las predicciones del mejor modelo LOSO (step6_classification_loso.py)
  - La estructura pre/post broncodilatador de cada paciente (v_bd)

Biomarcador principal:
    delta_CAS[i] = tasa_CAS_pre[i] - tasa_CAS_post[i]

Un delta positivo indica que el broncodilatador redujo la tasa de CAS,
lo que corresponde a la definición clínica de BDR+ (respuesta positiva).

Flujo:
    1. Cargar predicciones del modelo (predictions_all.npz de step6_loso)
    2. Cargar metadatos de sujetos y estructura pre/post (v_bd de dataset.npz)
    3. Construir dataset de 18 pacientes con features delta_CAS y derivadas
    4. LOSO a nivel de paciente: regresión logística / SVM lineal
    5. Comparar delta_CAS con etiqueta BDR+/BDR-

Uso:
    python src/step7b_delta_cas.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Localización del proyecto
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    candidate = Path(__file__).resolve().parent.parent
    for _ in range(6):
        if (candidate / "proy_labels.mat").exists():
            return candidate
        candidate = candidate.parent
    return Path(__file__).resolve().parent.parent


_PROJECT_ROOT = _find_project_root()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

RANDOM_STATE  = 42
N_PATIENTS    = 23
N_CONTROLS    = 5

STEP6_LOSO_DIR = _PROJECT_ROOT / "outputs" / "results" / "step6_loso"
STEP4_NPZ      = _PROJECT_ROOT / "outputs" / "results" / "step4" / "dataset.npz"
STEP5_DIR      = _PROJECT_ROOT / "outputs" / "results" / "step5"
METADATA_CSV   = _PROJECT_ROOT / "Data" / "database" / "subject_metadata.csv"

RESULTS_DIR    = _PROJECT_ROOT / "outputs" / "results" / "step7b"
FIGURES_DIR    = _PROJECT_ROOT / "outputs" / "figures" / "step7b"

MIN_SEGS_BD    = 3   # mínimo de segmentos por sesión pre/post para incluir paciente


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num_to_subj_id(num: int) -> str:
    if num <= N_PATIENTS:
        return f"P{num}"
    return f"C{num - N_PATIENTS}"


# ===========================================================================
# PARTE 1 — Cargar predicciones y metadatos
# ===========================================================================

def load_data(model_name: str = "best") -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame
]:
    """
    Carga las predicciones del modelo LOSO y los metadatos de sujetos.

    Parámetros
    ----------
    model_name : str
        Nombre del modelo a cargar ('SVM', 'RF', 'XGB', 'Ensemble' o 'best')

    Retorna
    -------
    y_prob_all   : (14900,) probabilidades de CAS por segmento
    y_pred_all   : (14900,) predicciones binarias
    v_subject    : (14900,) ID numérico de sujeto por segmento
    v_bd         : (14900,) sesión: 1=pre-BD, 2=post-BD
    metadata_df  : DataFrame con subject_id, bdr_label, type
    """
    pred_file = STEP6_LOSO_DIR / "predictions_all.npz"
    if not pred_file.exists():
        raise FileNotFoundError(
            f"No se encontró {pred_file}.\n"
            "Ejecuta primero: python src/step6_classification_loso.py"
        )

    preds = np.load(pred_file, allow_pickle=True)
    best_model = str(preds["best_model_name"])

    # Determinar qué claves específicas del modelo cargar
    if model_name == "best":
        y_prob_all = preds["y_prob_all"]
        y_pred_all = preds["y_pred_all"]
        loaded_name = best_model
    else:
        prob_key = f"y_prob_{model_name}"
        pred_key = f"y_pred_{model_name}"
        if prob_key in preds and pred_key in preds:
            y_prob_all = preds[prob_key]
            y_pred_all = preds[pred_key]
            loaded_name = model_name
        else:
            print(f"Advertencia: No se encontraron predicciones específicas para {model_name}. Cargando las del mejor modelo ({best_model}).")
            y_prob_all = preds["y_prob_all"]
            y_pred_all = preds["y_pred_all"]
            loaded_name = best_model

    npz       = np.load(STEP4_NPZ)
    v_subject = npz["v_subject"].astype(int)
    v_bd      = npz["v_bd"].astype(int)   # 1=pre, 2=post

    metadata_df = pd.read_csv(METADATA_CSV)

    print(f"Predicciones cargadas para el modelo: {loaded_name}")
    print(f"  Segmentos totales : {len(y_prob_all)}")
    print(f"  Sesión pre  (v_bd=1): {np.sum(v_bd == 1)}")
    print(f"  Sesión post (v_bd=2): {np.sum(v_bd == 2)}")
    print(f"  CAS predichos : {int(y_pred_all.sum())} / {len(y_pred_all)}")

    return y_prob_all, y_pred_all, v_subject, v_bd, metadata_df


# ===========================================================================
# PARTE 2 — Construir dataset a nivel de paciente
# ===========================================================================

def build_patient_dataset(
    y_prob_all: np.ndarray,
    y_pred_all: np.ndarray,
    v_subject: np.ndarray,
    v_bd: np.ndarray,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el dataset de nivel paciente con Delta_CAS y features derivadas.

    Para cada paciente con suficientes segmentos pre y post:
      - tasa_CAS_pre  = #{CAS pred en sesión pre}  / #{segs pre}
      - tasa_CAS_post = #{CAS pred en sesión post} / #{segs post}
      - delta_CAS     = tasa_CAS_pre - tasa_CAS_post  (>0 → BD reduce CAS → BDR+?)

    También incluye:
      - prob_mean_pre / post  : media de probabilidades (más suave que tasa binaria)
      - prob_delta            : diferencia de medias de probabilidad
      - iqr_prob_pre / post   : IQR de probabilidades (variabilidad intra-sesión)

    Retorna
    -------
    DataFrame con una fila por paciente, columnas:
        subject_id, bdr_label, type, n_pre, n_post,
        cas_rate_pre, cas_rate_post, delta_cas,
        prob_mean_pre, prob_mean_post, prob_delta,
        iqr_prob_pre, iqr_prob_post
    """
    bdr_map  = dict(zip(metadata_df["subject_id"], metadata_df["bdr_label"]))
    type_map = dict(zip(metadata_df["subject_id"], metadata_df["type"]))

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for num in sorted(np.unique(v_subject)):
        sid = _num_to_subj_id(int(num))

        mask_pre  = (v_subject == num) & (v_bd == 1)
        mask_post = (v_subject == num) & (v_bd == 2)

        n_pre  = int(mask_pre.sum())
        n_post = int(mask_post.sum())

        if n_pre < MIN_SEGS_BD or n_post < MIN_SEGS_BD:
            skipped.append(f"{sid} (pre={n_pre}, post={n_post})")
            continue

        prob_pre  = y_prob_all[mask_pre]
        prob_post = y_prob_all[mask_post]
        pred_pre  = y_pred_all[mask_pre]
        pred_post = y_pred_all[mask_post]

        cas_rate_pre  = float(pred_pre.mean())
        cas_rate_post = float(pred_post.mean())
        delta_cas     = cas_rate_pre - cas_rate_post

        prob_mean_pre  = float(prob_pre.mean())
        prob_mean_post = float(prob_post.mean())
        prob_delta     = prob_mean_pre - prob_mean_post

        iqr_prob_pre  = float(np.percentile(prob_pre,  75) - np.percentile(prob_pre,  25))
        iqr_prob_post = float(np.percentile(prob_post, 75) - np.percentile(prob_post, 25))

        rows.append({
            "subject_id":    sid,
            "subject_num":   int(num),
            "bdr_label":     bdr_map.get(sid, "BDR-"),
            "type":          type_map.get(sid, "patient"),
            "n_pre":         n_pre,
            "n_post":        n_post,
            "cas_rate_pre":  cas_rate_pre,
            "cas_rate_post": cas_rate_post,
            "delta_cas":     delta_cas,
            "prob_mean_pre":  prob_mean_pre,
            "prob_mean_post": prob_mean_post,
            "prob_delta":     prob_delta,
            "iqr_prob_pre":   iqr_prob_pre,
            "iqr_prob_post":  iqr_prob_post,
        })

    df = pd.DataFrame(rows)

    if skipped:
        print(f"\nSujetos excluidos por insuficientes segmentos pre/post:")
        for s in skipped:
            print(f"  {s}")

    print(f"\nDataset paciente construido: {len(df)} sujetos")
    print(f"  BDR+: {(df['bdr_label'] == 'BDR+').sum()}")
    print(f"  BDR-: {(df['bdr_label'] == 'BDR-').sum()}")
    print(f"  Controls: {(df['type'] == 'control').sum()}")
    print("\n" + df[["subject_id","bdr_label","n_pre","n_post",
                       "cas_rate_pre","cas_rate_post","delta_cas"]
                    ].to_string(index=False, float_format="%.3f"))

    return df


# ===========================================================================
# PARTE 3 — Análisis univariado de Delta_CAS
# ===========================================================================

def analyze_delta_cas(df: pd.DataFrame) -> None:
    """
    Análisis univariado: ¿separa delta_CAS a los pacientes BDR+ de BDR-?

    Solo pacientes (no controles) con etiqueta BDR conocida.
    """
    df_pat = df[df["type"] == "patient"].copy()

    bdr_pos = df_pat[df_pat["bdr_label"] == "BDR+"]["delta_cas"].values
    bdr_neg = df_pat[df_pat["bdr_label"] == "BDR-"]["delta_cas"].values

    print("\n" + "=" * 50)
    print("Análisis Delta_CAS — BDR+ vs BDR-")
    print("=" * 50)
    print(f"BDR+ (n={len(bdr_pos)}): media={bdr_pos.mean():.3f}, "
          f"std={bdr_pos.std():.3f}, mediana={np.median(bdr_pos):.3f}")
    print(f"BDR- (n={len(bdr_neg)}): media={bdr_neg.mean():.3f}, "
          f"std={bdr_neg.std():.3f}, mediana={np.median(bdr_neg):.3f}")

    if len(bdr_pos) > 0 and len(bdr_neg) > 0:
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(bdr_pos, bdr_neg, alternative="two-sided")
        print(f"Mann-Whitney U: stat={stat:.1f}, p={pval:.4f}")

        # AUC del delta_CAS como predictor univariado
        labels_bin = (df_pat["bdr_label"] == "BDR+").astype(int).values
        scores     = df_pat["delta_cas"].values
        try:
            auc = float(roc_auc_score(labels_bin, scores))
            print(f"AUC univariado (delta_CAS): {auc:.3f}")
        except ValueError:
            print("AUC no calculable (una sola clase)")


# ===========================================================================
# PARTE 4 — LOSO a nivel de paciente
# ===========================================================================

def run_patient_loso(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "bdr_label",
    pos_label: str = "BDR+",
) -> dict[str, Any]:
    """
    LOSO a nivel de paciente sobre el dataset de N filas.

    Con ~18 pacientes, se usan modelos simples (LogReg L2, SVM lineal)
    para evitar overfitting. Árboles y ensembles son inapropiados con
    tan pocas muestras.

    Retorna
    -------
    dict con per_fold, mean_acc, mean_auc, y_true, y_prob
    """
    # Solo pacientes con etiqueta BDR conocida
    df_pat = df[(df["type"] == "patient") & (df[label_col].isin(["BDR+", "BDR-"]))].copy()
    df_pat = df_pat.reset_index(drop=True)

    X      = df_pat[feature_cols].values.astype(float)
    y      = (df_pat[label_col] == pos_label).astype(int).values
    groups = df_pat["subject_num"].values

    loso = LeaveOneGroupOut()

    per_fold: list[dict] = []
    y_true_list: list[int] = []
    y_prob_list: list[float] = []

    models = {
        "LogReg": LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "SVM-Lin": SVC(
            kernel="linear", C=0.1, class_weight="balanced",
            probability=True, random_state=RANDOM_STATE,
        ),
    }

    results: dict[str, dict] = {}

    for model_name, clf in models.items():
        per_fold = []
        y_true_list = []
        y_prob_list = []

        for fold_i, (train_idx, test_idx) in enumerate(
            loso.split(X, y, groups), start=1
        ):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_te_sc = sc.transform(X_te)

            import copy
            clf_fold = copy.deepcopy(clf)
            clf_fold.fit(X_tr_sc, y_tr)

            y_pred_fold = clf_fold.predict(X_te_sc)
            y_prob_fold = clf_fold.predict_proba(X_te_sc)[:, 1]

            sid = df_pat.iloc[test_idx[0]]["subject_id"]
            lbl = int(y_te[0])

            per_fold.append({
                "fold":       fold_i,
                "subject_id": sid,
                "true":       lbl,
                "prob":       float(y_prob_fold[0]),
                "pred":       int(y_pred_fold[0]),
                "correct":    int(y_pred_fold[0] == lbl),
            })
            y_true_list.append(lbl)
            y_prob_list.append(float(y_prob_fold[0]))

            correct_str = "OK" if y_pred_fold[0] == lbl else "FAIL"
            print(
                f"  [{model_name}] Fold {fold_i:2d} — {sid} — "
                f"Prob: {y_prob_fold[0]:.3f} | "
                f"Pred: {'BDR+' if y_pred_fold[0] else 'BDR-':4s} | "
                f"Real: {'BDR+' if lbl else 'BDR-':4s} | {correct_str}"
            )

        acc = float(np.mean([f["correct"] for f in per_fold]))
        try:
            auc = float(roc_auc_score(y_true_list, y_prob_list))
        except ValueError:
            auc = 0.0

        print(f"  {model_name} → Acc: {acc:.3f} | AUC: {auc:.3f}\n")

        results[model_name] = {
            "per_fold": per_fold,
            "acc":      acc,
            "auc":      auc,
            "y_true":   y_true_list,
            "y_prob":   y_prob_list,
        }

    return results


# ===========================================================================
# PARTE 5 — Figuras
# ===========================================================================

def plot_delta_cas(df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Figura 1: Delta_CAS por paciente, coloreado por BDR label.
    Figura 2: Scatter cas_rate_pre vs cas_rate_post por paciente.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_pat = df[df["type"] == "patient"].copy()
    colors = {"BDR+": "green", "BDR-": "steelblue", "unknown": "gray"}

    # ---- Figura 1: Delta_CAS por paciente (barras) ----
    fig, ax = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(len(df_pat))
    bar_colors = [colors.get(row["bdr_label"], "gray") for _, row in df_pat.iterrows()]

    bars = ax.bar(x_pos, df_pat["delta_cas"].values, color=bar_colors, alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_pat["subject_id"].values, rotation=45, ha="right")
    ax.set_ylabel("Delta_CAS (pre − post)")
    ax.set_title("Biomarcador Delta_CAS por paciente\n"
                 "(positivo = más CAS antes del broncodilatador → candidato BDR+)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="green",    alpha=0.8, label="BDR+"),
        Patch(facecolor="steelblue", alpha=0.8, label="BDR-"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    out = figures_dir / "fig1_delta_cas_per_patient.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Figura 1 guardada: {out.name}")

    # ---- Figura 2: Scatter pre vs post ----
    fig, ax = plt.subplots(figsize=(7, 7))
    for _, row in df_pat.iterrows():
        c = colors.get(row["bdr_label"], "gray")
        ax.scatter(row["cas_rate_pre"], row["cas_rate_post"], color=c, s=90, zorder=3)
        ax.annotate(row["subject_id"],
                    (row["cas_rate_pre"], row["cas_rate_post"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    lim = max(df_pat["cas_rate_pre"].max(), df_pat["cas_rate_post"].max()) + 0.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="pre = post")
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.04, color="green",
                    label="pre > post (↓ CAS con BD)")
    ax.fill_between([0, lim], [0, 0], [0, lim], alpha=0.04, color="red",
                    label="post > pre (↑ CAS con BD)")
    ax.set_xlabel("Tasa CAS pre-broncodilatador")
    ax.set_ylabel("Tasa CAS post-broncodilatador")
    ax.set_title("CAS pre vs post por paciente")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    plt.tight_layout()
    out = figures_dir / "fig2_pre_vs_post.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Figura 2 guardada: {out.name}")


def plot_patient_roc(results: dict[str, dict], figures_dir: Path) -> None:
    """Figura 3: Curvas ROC de los modelos LOSO paciente."""
    colors_clf = {"LogReg": "royalblue", "SVM-Lin": "darkorange"}
    fig, ax = plt.subplots(figsize=(6, 6))

    for name, res in results.items():
        if len(set(res["y_true"])) < 2:
            continue
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        ax.plot(fpr, tpr, color=colors_clf.get(name, "black"), lw=2,
                label=f"{name} (AUC={res['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Curvas ROC — LOSO a nivel de paciente (BDR)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = figures_dir / "fig3_patient_roc.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Figura 3 guardada: {out.name}")


# ===========================================================================
# Punto de entrada
# ===========================================================================

def main() -> None:
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    models_to_test = ["SVM", "RF", "XGB", "Ensemble"]
    summary_rows = []

    # Features del dataset paciente
    feature_cols = [
        "delta_cas",          # biomarcador principal
        "cas_rate_pre",       # tasa pre
        "cas_rate_post",      # tasa post
        "prob_mean_pre",      # prob. media pre
        "prob_mean_post",     # prob. media post
        "prob_delta",         # delta de probabilidades (más suave)
        "iqr_prob_pre",       # variabilidad pre
        "iqr_prob_post",      # variabilidad post
    ]

    for model_name in models_to_test:
        print("\n" + "=" * 60)
        print(f"PROCESANDO MODELO SEGMENTO: {model_name}")
        print("=" * 60)
        
        try:
            y_prob_all, y_pred_all, v_subject, v_bd, metadata_df = load_data(model_name)
        except Exception as e:
            print(f"Error cargando datos para {model_name}: {e}")
            continue

        df_patients = build_patient_dataset(
            y_prob_all, y_pred_all, v_subject, v_bd, metadata_df
        )
        
        # Guardar dataset del modelo específico
        df_patients.to_csv(RESULTS_DIR / f"patient_delta_cas_{model_name}.csv",
                           index=False, float_format="%.4f")
        
        # Si es el modelo Ensemble (nuestro modelo con mejor balance de segmento/paciente)
        # guardarlo como el default
        if model_name == "Ensemble":
            df_patients.to_csv(RESULTS_DIR / "patient_delta_cas.csv",
                               index=False, float_format="%.4f")
            # Generar figuras para Ensemble
            print("\nGenerando figuras para el modelo Ensemble...")
            plot_delta_cas(df_patients, FIGURES_DIR)

        analyze_delta_cas(df_patients)

        print(f"\nEjecutando LOSO paciente con predicciones de {model_name}...")
        loso_results = run_patient_loso(df_patients, feature_cols)
        
        if model_name == "Ensemble":
            plot_patient_roc(loso_results, FIGURES_DIR)

        summary_rows.append({
            "model": model_name,
            "lr_acc": loso_results["LogReg"]["acc"],
            "lr_auc": loso_results["LogReg"]["auc"],
            "svm_acc": loso_results["SVM-Lin"]["acc"],
            "svm_auc": loso_results["SVM-Lin"]["auc"],
        })

    # Imprimir la tabla resumen comparativa de BDR
    print("\n" + "=" * 80)
    print("RESUMEN COMPARATIVO — CLASIFICACIÓN BDR PACIENTE")
    print("=" * 80)
    print(f"{'Modelo Segmento':<16} | {'LR Acc':<8} | {'LR AUC':<8} | {'SVM-Lin Acc':<12} | {'SVM-Lin AUC':<12}")
    print("-" * 80)
    for r in summary_rows:
        print(f"{r['model']:<16} | {r['lr_acc']:.3f}   | {r['lr_auc']:.3f}   | {r['svm_acc']:.3f}       | {r['svm_auc']:.3f}")
    print("=" * 80)

    elapsed = (time.time() - t0) / 60
    print(f"\nStep7b completado en {elapsed:.1f} minutos.")


if __name__ == "__main__":
    main()
