"""
Entrenamiento de clasificadores, evaluación Leave-One-Subject-Out (LOSO)
e inferencia sobre las 14 900 señales respiratorias.

Lee las matrices de features generadas por step5_features.py, entrena tres
clasificadores (SVM, Random Forest, XGBoost), selecciona el mejor por AUC
en LOSO, lo re-entrena sobre todos los datos etiquetados y predice en las
14 900 señales para su uso en step7_biomarker_analysis.py.

Uso:
    python src/step6_classification.py
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")  # backend sin pantalla para entornos sin GUI
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    print("WARNING: xgboost no instalado. Ejecuta: pip install xgboost")
    XGBClassifier = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Localización de la raíz del proyecto (igual que en step5_features.py)
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """
    Sube la jerarquía de directorios buscando proy_labels.mat para localizar
    la raíz del proyecto con independencia del directorio de trabajo actual.
    """
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
N_ESTIMATORS  = 300
RESULTS_DIR   = _PROJECT_ROOT / "outputs" / "results" / "step6"
FIGURES_DIR   = _PROJECT_ROOT / "outputs" / "figures" / "step6"

_STEP5_DIR    = _PROJECT_ROOT / "outputs" / "results" / "step5"
_STEP4_NPZ    = _PROJECT_ROOT / "outputs" / "results" / "step4" / "dataset.npz"
_METADATA_CSV = _PROJECT_ROOT / "Data" / "database" / "subject_metadata.csv"

_N_PATIENTS   = 23  # P1–P23
_N_CONTROLS   = 5   # C1–C5

# Colores por clasificador (usados en todas las figuras)
_COLORS: dict[str, str] = {
    "SVM": "blue", "RF": "green", "XGB": "orange", "Ensemble": "purple",
}


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _num_to_subj_id(num: int) -> str:
    """Convierte número de sujeto (1–28) a identificador string (P1–P23, C1–C5)."""
    if num <= _N_PATIENTS:
        return f"P{num}"
    return f"C{num - _N_PATIENTS}"


# ===========================================================================
# PARTE 1 — Carga de datos
# ===========================================================================

def load_data() -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, list[str], pd.DataFrame, np.ndarray,
]:
    """
    Carga las matrices de features de step5 y los metadatos de sujetos.

    Retorna
    -------
    X_labeled     : (1923, 15) features del subconjunto etiquetado
    y_labeled     : (1923,)   etiquetas binarias 1=CAS, 0=NO_CAS
    groups        : (1923,)   IDs numéricos de sujeto para LOSO
    X_all         : (14900, 15) features de todas las señales
    feature_names : lista de 15 strings con los nombres de features
    metadata_df   : DataFrame con columnas subject_id, type, sex, bdr_label
    v_subject_all : (14900,)  IDs numéricos de sujeto para todas las señales
    """
    X_labeled     = np.load(_STEP5_DIR / "X_labeled_features.npy")
    y_labeled     = np.load(_STEP5_DIR / "y_labeled.npy")
    groups        = np.load(_STEP5_DIR / "groups_labeled.npy").astype(int)
    X_all         = np.load(_STEP5_DIR / "X_all_features.npy")

    with open(_STEP5_DIR / "feature_names.json", encoding="utf-8") as fh:
        feature_names: list[str] = json.load(fh)

    metadata_df = pd.read_csv(_METADATA_CSV)

    # IDs de sujeto para las 14 900 señales (necesario para Fig. 5)
    npz           = np.load(_STEP4_NPZ)
    v_subject_all = npz["v_subject"].astype(int)

    n_cas    = int(y_labeled.sum())
    n_nocas  = int((1 - y_labeled).sum())
    n_total  = len(y_labeled)
    n_sujetos = len(np.unique(groups))

    print(f"Señales etiquetadas : {n_total}")
    print(f"  CAS (y=1)         : {n_cas} ({100 * n_cas / n_total:.1f}%)")
    print(f"  NO CAS (y=0)      : {n_nocas} ({100 * n_nocas / n_total:.1f}%)")
    print(f"Sujetos en LOSO     : {n_sujetos}")
    print(f"Señales totales     : {len(X_all)}")

    return X_labeled, y_labeled, groups, X_all, feature_names, metadata_df, v_subject_all


# ===========================================================================
# PARTE 2 — Definición de clasificadores
# ===========================================================================

def build_pipelines(y_labeled: np.ndarray) -> dict[str, Pipeline]:
    """
    Construye los cuatro Pipelines sklearn: SVM, RF, XGBoost y Ensemble.

    Cada pipeline incluye tres pasos:
        1. StandardScaler  — normalización de features
        2. SelectKBest     — selección de las 10 features más informativas
        3. Clasificador    — modelo de aprendizaje

    Para el Ensemble (VotingClassifier soft), los sub-estimadores reciben
    las features ya procesadas por el Pipeline externo, por lo que no
    llevan scaler ni selector propio.

    Los pesos de clase se fijan a {0:1, 1:3} para reforzar la sensibilidad
    al detectar CAS, aceptando un pequeño coste en especificidad. Esto es
    apropiado para un biomarcador de cribado donde los falsos negativos son
    más costosos que los falsos positivos.

    Parámetros
    ----------
    y_labeled : np.ndarray
        Etiquetas del subconjunto etiquetado (para calcular scale_pos_weight
        de XGBoost de forma equivalente al class_weight de sklearn).

    Retorna
    -------
    dict con claves "SVM", "RF", "XGB" (si instalado) y "Ensemble".
    """
    pipelines: dict[str, Pipeline] = {}
    scale_pos_weight = float(np.sum(y_labeled == 0) / np.sum(y_labeled == 1))

    # --- SVM con kernel RBF y peso 3x para clase CAS ---
    pipelines["SVM"] = Pipeline([
        ("scaler",   StandardScaler()),
        ("selector", SelectKBest(f_classif, k=10)),
        ("clf",      SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight={0: 1, 1: 3},
            probability=True,
            random_state=RANDOM_STATE,
        )),
    ])

    # --- Random Forest con peso 3x para clase CAS ---
    pipelines["RF"] = Pipeline([
        ("scaler",   StandardScaler()),
        ("selector", SelectKBest(f_classif, k=10)),
        ("clf",      RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=None,
            class_weight={0: 1, 1: 3},
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    # --- XGBoost con subsampling y compensación de desbalance ---
    if XGBClassifier is not None:
        xgb_params: dict[str, Any] = dict(
            n_estimators=N_ESTIMATORS,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        )
        try:
            # use_label_encoder eliminado en XGBoost 2.0; se ignora si falla
            xgb_clf = XGBClassifier(**xgb_params, use_label_encoder=False)
        except TypeError:
            xgb_clf = XGBClassifier(**xgb_params)

        pipelines["XGB"] = Pipeline([
            ("scaler",   StandardScaler()),
            ("selector", SelectKBest(f_classif, k=10)),
            ("clf",      xgb_clf),
        ])

    # --- Ensemble: VotingClassifier soft con los tres modelos base ---
    # Los sub-estimadores son clasificadores puros (sin scaler/selector propio)
    # porque el Pipeline externo ya aplica escalado y selección de features.
    svm_ens = SVC(
        kernel="rbf", C=1.0, gamma="scale",
        class_weight={0: 1, 1: 3},
        probability=True, random_state=RANDOM_STATE,
    )
    rf_ens = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight={0: 1, 1: 3},
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    estimators_ens: list[tuple[str, Any]] = [("svm", svm_ens), ("rf", rf_ens)]

    if XGBClassifier is not None:
        try:
            xgb_ens = XGBClassifier(**xgb_params, use_label_encoder=False)
        except TypeError:
            xgb_ens = XGBClassifier(**xgb_params)
        estimators_ens.append(("xgb", xgb_ens))

    pipelines["Ensemble"] = Pipeline([
        ("scaler",   StandardScaler()),
        ("selector", SelectKBest(f_classif, k=10)),
        ("clf",      VotingClassifier(estimators=estimators_ens, voting="soft")),
    ])

    return pipelines


# ===========================================================================
# PARTE 3 — Validación LOSO
# ===========================================================================

def run_loso(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf_name: str = "",
) -> dict[str, Any]:
    """
    Ejecuta Leave-One-Subject-Out cross-validation sobre el pipeline indicado.

    En cada fold se hace deepcopy del pipeline para garantizar independencia
    entre iteraciones. Se calculan accuracy, sensibilidad, especificidad,
    precisión, F1 y AUC para el sujeto dejado fuera.

    Parámetros
    ----------
    pipeline : Pipeline
        Pipeline sklearn con scaler + clf (sin entrenar).
    X : np.ndarray
        Matriz de features (1923, 15).
    y : np.ndarray
        Etiquetas binarias (1923,).
    groups : np.ndarray
        IDs numéricos de sujeto (1923,) para la partición LOSO.
    clf_name : str
        Nombre del clasificador para los mensajes de progreso.

    Retorna
    -------
    dict con claves:
        per_fold   — lista de dicts, uno por fold (sujeto)
        mean       — dict con la media de cada métrica entre folds
        std        — dict con la desviación estándar de cada métrica
        y_true_all — (n,) verdaderos etiquetas concatenados por fold
        y_pred_all — (n,) predicciones binarias concatenadas
        y_prob_all — (n,) probabilidades de clase 1 concatenadas
    """
    loso    = LeaveOneGroupOut()
    n_folds = loso.get_n_splits(X, y, groups)

    per_fold: list[dict[str, Any]] = []
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []
    y_prob_list: list[np.ndarray] = []

    for fold_i, (train_idx, test_idx) in enumerate(loso.split(X, y, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        subj_num = int(groups[test_idx[0]])
        subj_id  = _num_to_subj_id(subj_num)

        pipe = copy.deepcopy(pipeline)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc  = float(accuracy_score(y_test, y_pred))
        sens = float(recall_score(y_test, y_pred, zero_division=0))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        f1   = float(f1_score(y_test, y_pred, zero_division=0))

        # Especificidad: tn / (tn + fp) — calculada manualmente
        cm             = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec           = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            # El fold tiene una sola clase presente; AUC no definida
            auc = 0.0

        fold_dict: dict[str, Any] = {
            "fold":        fold_i,
            "subject_id":  subj_id,
            "accuracy":    acc,
            "sensitivity": sens,
            "specificity": spec,
            "precision":   prec,
            "f1":          f1,
            "auc":         auc,
        }
        per_fold.append(fold_dict)
        y_true_list.append(y_test)
        y_pred_list.append(y_pred)
        y_prob_list.append(y_prob)

        print(f"  Fold {fold_i:2d}/{n_folds} — Sujeto {subj_id:4s} — AUC: {auc:.3f}")

    # Estadísticas agregadas entre folds
    metrics   = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]
    mean_dict = {m: float(np.mean([f[m] for f in per_fold])) for m in metrics}
    std_dict  = {m: float(np.std([f[m] for f in per_fold]))  for m in metrics}

    return {
        "per_fold":   per_fold,
        "mean":       mean_dict,
        "std":        std_dict,
        "y_true_all": np.concatenate(y_true_list),
        "y_pred_all": np.concatenate(y_pred_list),
        "y_prob_all": np.concatenate(y_prob_list),
    }


def _save_loso_csv(per_fold: list[dict[str, Any]], clf_name: str) -> None:
    """Persiste los resultados por fold en un CSV en RESULTS_DIR."""
    cols  = ["fold", "subject_id", "accuracy", "sensitivity",
             "specificity", "precision", "f1", "auc"]
    fname = RESULTS_DIR / f"{clf_name.lower()}_loso_results.csv"
    pd.DataFrame(per_fold)[cols].to_csv(fname, index=False, float_format="%.4f")
    print(f"  Resultados LOSO guardados: {fname.name}")


# ===========================================================================
# PARTE 4 — Selección del mejor modelo y re-entrenamiento
# ===========================================================================

def select_and_retrain(
    results: dict[str, dict[str, Any]],
    pipelines: dict[str, Pipeline],
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_all: np.ndarray,
) -> tuple[str, Pipeline, np.ndarray, np.ndarray]:
    """
    Selecciona el clasificador con mayor AUC media en LOSO, lo re-entrena
    sobre los 1923 datos etiquetados y aplica la inferencia a las 14 900 señales.

    Guarda en RESULTS_DIR:
        predictions_all.npz — y_pred_all, y_prob_all, best_model_name
        best_model.pkl      — pipeline re-entrenado

    Retorna
    -------
    best_name     : nombre del mejor clasificador
    best_pipeline : pipeline re-entrenado sobre todos los datos etiquetados
    y_pred_all    : (14900,) predicciones binarias int
    y_prob_all    : (14900,) probabilidades de CAS float
    """
    best_name = max(results, key=lambda k: results[k]["mean"]["auc"])
    best_auc  = results[best_name]["mean"]["auc"]
    print(f"\nMejor modelo: {best_name} (AUC LOSO = {best_auc:.3f})")

    best_pipeline = copy.deepcopy(pipelines[best_name])
    best_pipeline.fit(X_labeled, y_labeled)

    y_pred_all = best_pipeline.predict(X_all).astype(int)
    y_prob_all = best_pipeline.predict_proba(X_all)[:, 1]

    # Guardar predicciones en disco
    np.savez(
        RESULTS_DIR / "predictions_all.npz",
        y_pred_all=y_pred_all,
        y_prob_all=y_prob_all,
        best_model_name=np.array(best_name),
    )
    joblib.dump(best_pipeline, RESULTS_DIR / "best_model.pkl")

    n_cas = int(y_pred_all.sum())
    n_all = len(y_pred_all)
    print(f"Señales clasificadas como CAS : {n_cas} / {n_all} ({100 * n_cas / n_all:.1f}%)")

    return best_name, best_pipeline, y_pred_all, y_prob_all


# ===========================================================================
# PARTE 5 — Figuras
# ===========================================================================

def plot_roc_curves(results: dict[str, dict[str, Any]], figures_dir: Path) -> None:
    """
    Figura 1: Curvas ROC calculadas sobre las predicciones LOSO concatenadas.

    Cada clasificador aporta una curva con su AUC global (no media de folds).
    Se añade la diagonal punteada que representa un clasificador aleatorio.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_true_all"], res["y_prob_all"])
        auc_val      = float(roc_auc_score(res["y_true_all"], res["y_prob_all"]))
        ax.plot(fpr, tpr,
                color=_COLORS.get(name, "black"),
                lw=2,
                label=f"{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatorio")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC — Validación LOSO")
    ax.legend(loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])

    plt.tight_layout()
    out = figures_dir / "fig1_roc_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Figura 1 guardada: {out.name}")


def plot_confusion_matrices(
    results: dict[str, dict[str, Any]],
    figures_dir: Path,
) -> None:
    """
    Figura 2: Matrices de confusión normalizadas por fila (porcentaje de clase real).

    Un subplot por clasificador con escala de color azul (seaborn).
    """
    n_clf  = len(results)
    labels = ["NO CAS", "CAS"]

    fig, axes = plt.subplots(1, n_clf, figsize=(5 * n_clf, 5))
    if n_clf == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm      = confusion_matrix(res["y_true_all"], res["y_pred_all"], labels=[0, 1])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        acc_med = results[name]["mean"]["accuracy"]

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".1%",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(f"{name}\nAcc media = {acc_med:.3f}")

    plt.tight_layout()
    out = figures_dir / "fig2_confusion_matrices.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Figura 2 guardada: {out.name}")


def plot_loso_auc_per_fold(
    results: dict[str, dict[str, Any]],
    metadata_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Figura 3: Evolución del AUC por sujeto (fold) para cada clasificador.

    Los marcadores en el eje X indican el estado BDR de cada sujeto:
      - BDR+ → círculo verde
      - BDR- → cuadrado azul
    Las líneas de trazo discontinuo muestran el AUC medio de cada clasificador.
    """
    bdr_map = dict(zip(metadata_df["subject_id"], metadata_df["bdr_label"]))

    # Usar el primer clasificador disponible para definir el orden de sujetos
    first_name       = next(iter(results))
    subj_ids_ordered = [f["subject_id"] for f in results[first_name]["per_fold"]]
    x_pos            = np.arange(len(subj_ids_ordered))

    fig, ax = plt.subplots(figsize=(14, 6))

    # Líneas por clasificador
    for name, res in results.items():
        aucs  = [f["auc"] for f in res["per_fold"]]
        color = _COLORS.get(name, "black")
        ax.plot(x_pos, aucs, color=color, lw=2, alpha=0.8)
        ax.axhline(
            res["mean"]["auc"],
            color=color, ls="--", lw=1.2, alpha=0.6,
        )

    # Marcadores de BDR debajo del eje X (clip_on=False permite dibujar fuera)
    for i, sid in enumerate(subj_ids_ordered):
        bdr    = bdr_map.get(sid, "BDR-")
        marker = "o" if bdr == "BDR+" else "s"
        mcolor = "green" if bdr == "BDR+" else "steelblue"
        ax.scatter(i, -0.06, marker=marker, color=mcolor,
                   s=80, zorder=5, clip_on=False)

    # Leyenda combinada: clasificadores + estado BDR
    clf_handles = [
        mlines.Line2D(
            [0], [0],
            color=_COLORS.get(n, "black"), lw=2,
            label=f"{n} (media = {results[n]['mean']['auc']:.3f})",
        )
        for n in results
    ]
    bdr_handles = [
        mlines.Line2D(
            [0], [0],
            marker="o", color="w", markerfacecolor="green",
            markersize=10, label="BDR+",
        ),
        mlines.Line2D(
            [0], [0],
            marker="s", color="w", markerfacecolor="steelblue",
            markersize=9, label="BDR-",
        ),
    ]
    ax.legend(handles=clf_handles + bdr_handles, loc="lower right", fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(subj_ids_ordered, rotation=45, ha="right")
    ax.set_xlabel("Sujeto")
    ax.set_ylabel("AUC")
    ax.set_ylim([-0.1, 1.05])
    ax.set_title("AUC por sujeto — Validación LOSO")

    plt.tight_layout()
    out = figures_dir / "fig3_loso_auc_per_fold.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Figura 3 guardada: {out.name}")


def plot_feature_importance(
    pipelines: dict[str, Pipeline],
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    feature_names: list[str],
    figures_dir: Path,
) -> None:
    """
    Figura 4: Importancia de features para RF (MDI) y XGBoost (total gain).

    Ambos modelos se re-entrenan sobre los 1923 datos etiquetados para obtener
    las importancias finales. Los subplots muestran barras horizontales ordenadas
    de mayor a menor importancia.
    """
    # Re-entrenar RF y XGB sobre el conjunto completo etiquetado
    models_to_plot: list[tuple[str, str, np.ndarray]] = []

    for name in ["RF", "XGB"]:
        if name not in pipelines:
            continue

        pipe = copy.deepcopy(pipelines[name])
        pipe.fit(X_labeled, y_labeled)
        clf      = pipe.named_steps["clf"]
        selector = pipe.named_steps["selector"]

        # Recuperar los nombres de las 10 features seleccionadas en este fold
        selected_idx   = selector.get_support(indices=True)
        selected_names = [feature_names[i] for i in selected_idx]

        if name == "RF":
            importances = np.array(clf.feature_importances_, dtype=float)
            models_to_plot.append((name, "green", importances, selected_names))

        elif name == "XGB":
            try:
                # total_gain: dict con claves 'f0', 'f1', ... referidas a las
                # 10 features seleccionadas (no a las 15 originales)
                score_dict  = clf.get_booster().get_score(importance_type="total_gain")
                importances = np.zeros(len(selected_names), dtype=float)
                for key, val in score_dict.items():
                    if key.startswith("f") and key[1:].isdigit():
                        idx = int(key[1:])
                        if idx < len(selected_names):
                            importances[idx] = float(val)
            except Exception:
                importances = np.array(
                    getattr(clf, "feature_importances_",
                            np.zeros(len(selected_names))),
                    dtype=float,
                )
            models_to_plot.append((name, "orange", importances, selected_names))

    if not models_to_plot:
        return

    n_plots = len(models_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 7))
    if n_plots == 1:
        axes = [axes]

    for ax, (name, color, importances, feat_names_sel) in zip(axes, models_to_plot):
        # Ordenar ascendente para que la barra más importante quede arriba en barh
        order     = np.argsort(importances)
        names_ord = [feat_names_sel[i] for i in order]
        vals_ord  = importances[order]

        ax.barh(names_ord, vals_ord, color=color, alpha=0.8)
        ax.set_xlabel("Importancia")
        ax.set_title(f"{name} — {'MDI' if name == 'RF' else 'Total Gain'}")
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle("Importancia de features", fontsize=13)
    plt.tight_layout()
    out = figures_dir / "fig4_feature_importance.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Figura 4 guardada: {out.name}")


def plot_cas_rate_by_group(
    y_pred_all: np.ndarray,
    metadata_df: pd.DataFrame,
    v_subject_all: np.ndarray,
    figures_dir: Path,
) -> None:
    """
    Figura 5: Tasa de CAS predicha agrupada por estado BDR (BDR+, BDR-, Controles).

    Barras de media ± desviación estándar entre sujetos del grupo.
    Puntos individuales con jitter horizontal muestran la variabilidad intra-grupo.
    """
    bdr_map  = dict(zip(metadata_df["subject_id"], metadata_df["bdr_label"]))
    type_map = dict(zip(metadata_df["subject_id"], metadata_df["type"]))

    # Tasa de CAS por sujeto sobre todas sus señales
    subj_cas_rates: dict[str, float] = {}
    for num in np.unique(v_subject_all):
        sid  = _num_to_subj_id(int(num))
        mask = v_subject_all == num
        subj_cas_rates[sid] = float(y_pred_all[mask].mean())

    # Asignación a grupo
    group_rates: dict[str, list[float]] = {
        "BDR+": [], "BDR-": [], "Controls": [],
    }
    for sid, rate in subj_cas_rates.items():
        if type_map.get(sid) == "control":
            group_rates["Controls"].append(rate)
        elif bdr_map.get(sid) == "BDR+":
            group_rates["BDR+"].append(rate)
        else:
            group_rates["BDR-"].append(rate)

    groups_order = ["BDR+", "BDR-", "Controls"]
    colors_grp   = {"BDR+": "green", "BDR-": "steelblue", "Controls": "gray"}

    means = [np.mean(group_rates[g]) if group_rates[g] else 0.0 for g in groups_order]
    stds  = [np.std(group_rates[g])  if group_rates[g] else 0.0 for g in groups_order]
    x_pos = np.arange(len(groups_order))

    rng = np.random.default_rng(RANDOM_STATE)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, g in enumerate(groups_order):
        ax.bar(
            i, means[i],
            color=colors_grp[g], alpha=0.6,
            yerr=stds[i], capsize=6,
            label=g,
        )
        pts = group_rates[g]
        if pts:
            jitter = rng.uniform(-0.15, 0.15, size=len(pts))
            ax.scatter(
                np.full(len(pts), i) + jitter, pts,
                color=colors_grp[g], s=60, alpha=0.85, zorder=4,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups_order)
    ax.set_ylabel("Tasa de CAS predicha")
    ax.set_ylim([0.0, 1.05])
    ax.set_title("Tasa de CAS predicha por grupo")
    ax.legend(loc="upper right")

    plt.tight_layout()
    out = figures_dir / "fig5_cas_rate_by_group.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Figura 5 guardada: {out.name}")


# ===========================================================================
# Punto de entrada
# ===========================================================================

def main() -> None:
    """Orquesta el pipeline completo de clasificación (partes 1–5)."""
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    t_total = time.time()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ P1
    print("=" * 60)
    print("PARTE 1 — Carga de datos")
    print("=" * 60)
    (X_labeled, y_labeled, groups,
     X_all, feature_names,
     metadata_df, v_subject_all) = load_data()

    # ------------------------------------------------------------------ P2
    print("\n" + "=" * 60)
    print("PARTE 2 — Definición de clasificadores")
    print("=" * 60)
    pipelines = build_pipelines(y_labeled)
    print(f"Clasificadores disponibles: {list(pipelines.keys())}")

    # ------------------------------------------------------------------ P3
    print("\n" + "=" * 60)
    print("PARTE 3 — Validación LOSO")
    print("=" * 60)
    results: dict[str, dict[str, Any]] = {}

    for name, pipeline in pipelines.items():
        print(f"\n--- {name} ---")
        t0  = time.time()
        res = run_loso(pipeline, X_labeled, y_labeled, groups, clf_name=name)
        elapsed = (time.time() - t0) / 60
        results[name] = res

        _save_loso_csv(res["per_fold"], name)
        print(f"{name} — LOSO completado en {elapsed:.1f} minutos.")
        print(
            f"{name}  — AUC: {res['mean']['auc']:.3f} ± {res['std']['auc']:.3f}"
            f" | F1: {res['mean']['f1']:.3f} ± {res['std']['f1']:.3f}"
        )

    # ------------------------------------------------------------------ Tabla comparativa LOSO
    ancho_m = 10
    ancho_v = 13
    sep = "+" + "-" * (ancho_m + 2) + ("+" + "-" * (ancho_v + 2)) * 4 + "+"
    print("\n" + sep)
    print(
        f"| {'Modelo':<{ancho_m}} "
        f"| {'Accuracy':^{ancho_v}} "
        f"| {'Sensitivity':^{ancho_v}} "
        f"| {'Specificity':^{ancho_v}} "
        f"| {'AUC':^{ancho_v}} |"
    )
    print(sep)
    for name, res in results.items():
        m, s = res["mean"], res["std"]
        print(
            f"| {name:<{ancho_m}} "
            f"| {m['accuracy']:.2f} +/- {s['accuracy']:.2f}  "
            f"| {m['sensitivity']:.2f} +/- {s['sensitivity']:.2f}  "
            f"| {m['specificity']:.2f} +/- {s['specificity']:.2f}  "
            f"| {m['auc']:.2f} +/- {s['auc']:.2f}  |"
        )
    print(sep)

    # ------------------------------------------------------------------ P4
    print("\n" + "=" * 60)
    print("PARTE 4 — Mejor modelo y predicciones sobre 14 900 señales")
    print("=" * 60)
    best_name, best_pipeline, y_pred_all, y_prob_all = select_and_retrain(
        results, pipelines, X_labeled, y_labeled, X_all,
    )

    # ------------------------------------------------------------------ P5
    print("\n" + "=" * 60)
    print("PARTE 5 — Generación de figuras")
    print("=" * 60)
    plot_roc_curves(results, FIGURES_DIR)
    plot_confusion_matrices(results, FIGURES_DIR)
    plot_loso_auc_per_fold(results, metadata_df, FIGURES_DIR)
    plot_feature_importance(pipelines, X_labeled, y_labeled, feature_names, FIGURES_DIR)
    plot_cas_rate_by_group(y_pred_all, metadata_df, v_subject_all, FIGURES_DIR)

    elapsed_total = (time.time() - t_total) / 60
    print(f"\nClasificación completada en {elapsed_total:.1f} minutos.")


if __name__ == "__main__":
    main()
