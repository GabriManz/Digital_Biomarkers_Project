"""
Step 7: Análisis clínico de biomarcadores basado en CAS.

Lee las predicciones del clasificador (step6) y computa el análisis de
respuesta broncodilatadora (BDR) basado en la tasa de CAS por sujeto,
comparando los grupos BDR+, BDR- y controles sanos.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

# ─────────────────────────────────────────────────────────────────────────────
# Constantes globales
# ─────────────────────────────────────────────────────────────────────────────
BD_PRE      = 1
BD_POST     = 2
CH_LOWER    = 1
CH_UPPER    = 2
PHASE_INSP  = 1
PHASE_ESP   = 2

# Detectar raíz del proyecto anclando en proy_labels.mat
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (p for p in [_HERE.parent, _HERE] if (p / "proy_labels.mat").exists()),
    _HERE.parent,
)

RESULTS_DIR = PROJECT_ROOT / "outputs" / "results" / "step7"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "step7"

COLORS      = {"BDR+": "green", "BDR-": "steelblue", "Controls": "gray"}
GROUP_ORDER = ["BDR+", "BDR-", "Controls"]


# ─────────────────────────────────────────────────────────────────────────────
# PARTE 1 — Carga de datos
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    pd.DataFrame, dict, dict, str,
]:
    """Carga predicciones del clasificador, vectores de metadata y etiquetas clínicas.

    Returns:
        y_pred_all, y_prob_all, v_subject, v_bd, v_channel, v_phase,
        meta_df, subject_num_to_id, subject_meta, best_model_name
    """
    # Predicciones de step6
    pred_path = PROJECT_ROOT / "outputs" / "results" / "step6" / "predictions_all.npz"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"No se encontró {pred_path}. Ejecutar step6_classification.py primero."
        )
    pred = np.load(pred_path, allow_pickle=True)
    y_pred_all = pred["y_pred_all"]
    y_prob_all = pred["y_prob_all"]
    best_model_name = str(pred["best_model_name"])

    # Vectores de metadata del dataset (step4)
    ds_path = PROJECT_ROOT / "outputs" / "results" / "step4" / "dataset.npz"
    ds = np.load(ds_path)
    v_subject = ds["v_subject"]
    v_bd      = ds["v_bd"]
    v_channel = ds["v_channel"]
    v_phase   = ds["v_phase"]

    # Metadatos clínicos por sujeto
    meta_path = PROJECT_ROOT / "database" / "subject_metadata.csv"
    meta_df = pd.read_csv(meta_path)

    # Mapeo: número entero → ID de cadena (ej. 1 → "P1", 24 → "C1")
    subject_num_to_id: dict[int, str] = {
        int(row["subject_num"]): row["subject_id"]
        for _, row in meta_df.iterrows()
    }
    # Información clínica indexada por ID de sujeto
    subject_meta: dict[str, dict] = {
        row["subject_id"]: {"type": row["type"], "bdr_label": row["bdr_label"]}
        for _, row in meta_df.iterrows()
    }

    # Verificación de formas esperadas
    for name, arr in [
        ("y_pred_all", y_pred_all), ("y_prob_all", y_prob_all),
        ("v_subject",  v_subject),  ("v_bd",       v_bd),
        ("v_channel",  v_channel),  ("v_phase",    v_phase),
    ]:
        assert arr.shape == (14900,), (
            f"Shape inesperada para {name}: {arr.shape} (esperado (14900,))"
        )

    print("=" * 55)
    print("CARGA DE DATOS — STEP 7")
    print("=" * 55)
    print(f"  y_pred_all  : {y_pred_all.shape}  (dtype={y_pred_all.dtype})")
    print(f"  y_prob_all  : {y_prob_all.shape}  (dtype={y_prob_all.dtype})")
    print(f"  v_subject   : {v_subject.shape}  (dtype={v_subject.dtype})")
    print(f"  v_bd        : {v_bd.shape}  (dtype={v_bd.dtype})")
    print(f"  v_channel   : {v_channel.shape}  (dtype={v_channel.dtype})")
    print(f"  v_phase     : {v_phase.shape}  (dtype={v_phase.dtype})")
    print(f"  Sujetos     : {len(subject_num_to_id)} (pacientes + controles)")
    print(f"  Mejor modelo: {best_model_name}")
    print("=" * 55)

    return (
        y_pred_all, y_prob_all,
        v_subject, v_bd, v_channel, v_phase,
        meta_df, subject_num_to_id, subject_meta, best_model_name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PARTE 2 — Métricas CAS por sujeto
# ─────────────────────────────────────────────────────────────────────────────

def compute_cas_metrics(
    y_pred: np.ndarray,
    v_subject: np.ndarray,
    v_bd: np.ndarray,
    v_channel: np.ndarray,
    v_phase: np.ndarray,
    subject_num_to_id: dict[int, str],
    subject_meta: dict[str, dict],
    channel: Union[int, str],
    phase: Union[int, str],
) -> pd.DataFrame:
    """Calcula métricas CAS pre/post broncodilatador para cada sujeto.

    Para cada sujeto se calcula:
        n_pre, n_post   : número de señales en cada condición
        cas_pre/post    : suma de predicciones positivas (CAS detectados)
        cas_rate_pre/post: tasa porcentual de CAS respecto al total
        delta_cas       : cambio relativo (%) en CAS tras BD
                          delta_cas = 100 * (cas_pre - cas_post) / cas_pre

    Args:
        channel: CH_LOWER (1), CH_UPPER (2) o "both".
        phase:   PHASE_INSP (1), PHASE_ESP (2) o "both".

    Returns:
        DataFrame con columnas:
        subject_id, bdr_group, n_pre, n_post,
        cas_pre, cas_post, cas_rate_pre, cas_rate_post, delta_cas
    """
    mask_ch: np.ndarray = (
        np.ones(len(v_subject), dtype=bool)
        if channel == "both"
        else (v_channel == channel)
    )
    mask_ph: np.ndarray = (
        np.ones(len(v_subject), dtype=bool)
        if phase == "both"
        else (v_phase == phase)
    )
    mask_base = mask_ch & mask_ph

    filas = []
    for subject_num, subject_id in sorted(subject_num_to_id.items()):
        mask_subj = (v_subject == subject_num) & mask_base

        pre_mask  = mask_subj & (v_bd == BD_PRE)
        post_mask = mask_subj & (v_bd == BD_POST)

        n_pre  = int(pre_mask.sum())
        n_post = int(post_mask.sum())
        cas_pre  = int(y_pred[pre_mask].sum())
        cas_post = int(y_pred[post_mask].sum())

        cas_rate_pre  = (cas_pre  / n_pre  * 100.0) if n_pre  > 0 else 0.0
        cas_rate_post = (cas_post / n_post * 100.0) if n_post > 0 else 0.0
        delta_cas     = (100.0 * (cas_pre - cas_post) / cas_pre) if cas_pre > 0 else 0.0

        meta = subject_meta[subject_id]
        bdr_group = "Controls" if meta["type"] == "control" else meta["bdr_label"]

        filas.append({
            "subject_id":    subject_id,
            "bdr_group":     bdr_group,
            "n_pre":         n_pre,
            "n_post":        n_post,
            "cas_pre":       cas_pre,
            "cas_post":      cas_post,
            "cas_rate_pre":  round(cas_rate_pre, 4),
            "cas_rate_post": round(cas_rate_post, 4),
            "delta_cas":     round(delta_cas, 4),
        })

    return pd.DataFrame(filas)


def run_all_cas_analyses(
    y_pred: np.ndarray,
    v_subject: np.ndarray,
    v_bd: np.ndarray,
    v_channel: np.ndarray,
    v_phase: np.ndarray,
    subject_num_to_id: dict[int, str],
    subject_meta: dict[str, dict],
) -> dict[str, pd.DataFrame]:
    """Ejecuta compute_cas_metrics para las 9 combinaciones de canal × fase y guarda CSVs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # (nombre_archivo, canal, fase)
    condiciones = [
        ("all",      "both",   "both"),
        ("ch1",      CH_LOWER, "both"),
        ("ch2",      CH_UPPER, "both"),
        ("insp",     "both",   PHASE_INSP),
        ("esp",      "both",   PHASE_ESP),
        ("ch1_insp", CH_LOWER, PHASE_INSP),
        ("ch1_esp",  CH_LOWER, PHASE_ESP),
        ("ch2_insp", CH_UPPER, PHASE_INSP),
        ("ch2_esp",  CH_UPPER, PHASE_ESP),
    ]

    resultados: dict[str, pd.DataFrame] = {}
    for nombre, ch, ph in condiciones:
        df = compute_cas_metrics(
            y_pred, v_subject, v_bd, v_channel, v_phase,
            subject_num_to_id, subject_meta, ch, ph,
        )
        csv_path = RESULTS_DIR / f"cas_metrics_{nombre}.csv"
        df.to_csv(csv_path, index=False)
        resultados[nombre] = df
        print(f"  Guardado: {csv_path.name}  ({len(df)} sujetos)")

    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# PARTE 3 — Comparación entre grupos
# ─────────────────────────────────────────────────────────────────────────────

def compute_group_statistics(resultados: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calcula estadísticos descriptivos y pruebas no paramétricas por grupo y condición."""
    filas_stats = []

    for condicion, df in resultados.items():
        grupos = {g: df[df["bdr_group"] == g] for g in GROUP_ORDER}

        # Mann-Whitney U (BDR+ vs BDR-) para delta_cas
        bdr_plus  = grupos["BDR+"]["delta_cas"].values
        bdr_minus = grupos["BDR-"]["delta_cas"].values
        if len(bdr_plus) > 0 and len(bdr_minus) > 0:
            _, mwu_p = stats.mannwhitneyu(bdr_plus, bdr_minus, alternative="two-sided")
        else:
            mwu_p = np.nan

        # Kruskal-Wallis (BDR+ vs BDR- vs Controls) para delta_cas
        all_values = [g["delta_cas"].values for g in grupos.values() if len(g) > 0]
        if len(all_values) >= 2:
            _, kw_p = stats.kruskal(*all_values)
        else:
            kw_p = np.nan

        for grupo, gdf in grupos.items():
            filas_stats.append({
                "condition":          condicion,
                "group":              grupo,
                "n":                  len(gdf),
                "cas_rate_pre_mean":  round(gdf["cas_rate_pre"].mean(),  3),
                "cas_rate_pre_std":   round(gdf["cas_rate_pre"].std(),   3),
                "cas_rate_post_mean": round(gdf["cas_rate_post"].mean(), 3),
                "cas_rate_post_std":  round(gdf["cas_rate_post"].std(),  3),
                "delta_cas_mean":     round(gdf["delta_cas"].mean(),     3),
                "delta_cas_std":      round(gdf["delta_cas"].std(),      3),
                "mwu_pvalue":         round(float(mwu_p), 4) if not np.isnan(mwu_p) else np.nan,
                "kruskal_pvalue":     round(float(kw_p),  4) if not np.isnan(kw_p)  else np.nan,
            })

    stats_df = pd.DataFrame(filas_stats)
    out_path = RESULTS_DIR / "group_statistics.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"  Guardado: {out_path.name}")
    return stats_df


def print_group_table(resultados: dict[str, pd.DataFrame]) -> None:
    """Imprime la tabla de comparación de grupos para el análisis global (todos los canales)."""
    df_all = resultados["all"]

    bdr_plus  = df_all[df_all["bdr_group"] == "BDR+"]["delta_cas"].values
    bdr_minus = df_all[df_all["bdr_group"] == "BDR-"]["delta_cas"].values
    _, mwu_p  = stats.mannwhitneyu(bdr_plus, bdr_minus, alternative="two-sided")
    _, kw_p   = stats.kruskal(
        bdr_plus, bdr_minus,
        df_all[df_all["bdr_group"] == "Controls"]["delta_cas"].values,
    )

    print("\n" + "=" * 72)
    print("ANÁLISIS CAS — TODOS LOS CANALES Y FASES (análisis principal)")
    print("=" * 72)
    header = (
        f"{'Grupo':<10} | {'CAS pre (%)':<16} | {'CAS post (%)':<16} "
        f"| {'Δ CAS (%)':<16} | p-value (MWU)"
    )
    print(header)
    print("-" * 72)

    for grupo in GROUP_ORDER:
        gdf = df_all[df_all["bdr_group"] == grupo]
        pre_s  = f"{gdf['cas_rate_pre'].mean():.1f} ± {gdf['cas_rate_pre'].std():.1f}"
        post_s = f"{gdf['cas_rate_post'].mean():.1f} ± {gdf['cas_rate_post'].std():.1f}"
        dlt_s  = f"{gdf['delta_cas'].mean():.1f} ± {gdf['delta_cas'].std():.1f}"
        p_str  = f"{mwu_p:.4f}" if grupo in ("BDR+", "BDR-") else "—"
        print(f"{grupo:<10} | {pre_s:<16} | {post_s:<16} | {dlt_s:<16} | {p_str}")

    print("=" * 72)
    print(f"  Mann-Whitney U (BDR+ vs BDR-)                : p = {mwu_p:.4f}")
    print(f"  Kruskal-Wallis (BDR+ vs BDR- vs Controls)    : p = {kw_p:.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PARTE 4 — Figuras
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, filename: str) -> None:
    """Guarda la figura con configuración estándar (dpi=150, tight_layout)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {out_path.name}")


def fig1_cas_rate_pre_post_ch1(resultados: dict[str, pd.DataFrame]) -> None:
    """Figura 1 — Tasa de CAS pre-BD y post-BD por sujeto (canal inferior, CH1)."""
    df = resultados["ch1"].sort_values("subject_id").reset_index(drop=True)
    subjects = df["subject_id"].tolist()
    bar_colors = [COLORS[g] for g in df["bdr_group"]]
    x = np.arange(len(subjects))
    width = 0.7

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    for ax, col, ylabel in zip(
        axes,
        ["cas_rate_pre", "cas_rate_post"],
        ["CAS pre-BD (%)", "CAS post-BD (%)"],
    ):
        ax.bar(x, df[col].values, color=bar_colors, width=width,
               edgecolor="white", linewidth=0.5)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0, max(df[col].max() * 1.18, 5.0))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(subjects, rotation=45, ha="right", fontsize=9)

    handles = [mpatches.Patch(color=COLORS[g], label=g) for g in GROUP_ORDER]
    axes[0].legend(handles=handles, loc="upper right", fontsize=9)
    axes[0].set_title(
        "Tasa de CAS pre-BD y post-BD por sujeto (Canal inferior)",
        fontsize=13, fontweight="bold",
    )

    _save_fig(fig, "fig1_cas_rate_pre_post_ch1.png")


def fig2_delta_cas_per_subject(resultados: dict[str, pd.DataFrame]) -> None:
    """Figura 2 — Δ CAS por sujeto, barras horizontales ordenadas de mayor a menor."""
    df = (
        resultados["all"]
        .sort_values("delta_cas", ascending=False)
        .reset_index(drop=True)
    )
    bar_colors = [COLORS[g] for g in df["bdr_group"]]

    fig, ax = plt.subplots(figsize=(10, 12))
    y = np.arange(len(df))
    ax.barh(y, df["delta_cas"].values, color=bar_colors,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(df["subject_id"].tolist(), fontsize=9)
    ax.set_xlabel("Δ CAS (%)", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    handles = [mpatches.Patch(color=COLORS[g], label=g) for g in GROUP_ORDER]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.set_title(
        "Cambio en CAS tras broncodilatador — Δ CAS (%)",
        fontsize=13, fontweight="bold",
    )

    _save_fig(fig, "fig2_delta_cas_per_subject.png")


def fig3_boxplot_delta_cas(resultados: dict[str, pd.DataFrame]) -> None:
    """Figura 3 — Boxplot de Δ CAS por grupo clínico con puntos individuales superpuestos."""
    df_all = resultados["all"]
    bdr_plus  = df_all[df_all["bdr_group"] == "BDR+"]["delta_cas"].values
    bdr_minus = df_all[df_all["bdr_group"] == "BDR-"]["delta_cas"].values
    _, mwu_p  = stats.mannwhitneyu(bdr_plus, bdr_minus, alternative="two-sided")

    data_list = [df_all[df_all["bdr_group"] == g]["delta_cas"].values for g in GROUP_ORDER]

    fig, ax = plt.subplots(figsize=(10, 7))
    bp = ax.boxplot(
        data_list,
        patch_artist=True,
        widths=0.4,
        medianprops={"color": "black", "linewidth": 2},
    )
    for patch, grupo in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor(COLORS[grupo])
        patch.set_alpha(0.6)

    # Puntos individuales con jitter para evitar solapamiento
    rng = np.random.default_rng(42)
    for i, (grupo, vals) in enumerate(zip(GROUP_ORDER, data_list), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(i + jitter, vals, color=COLORS[grupo], s=40, zorder=3,
                   edgecolors="black", linewidths=0.5, alpha=0.85)

    ax.axhline(0, color="gray", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(GROUP_ORDER, fontsize=12)
    ax.set_ylabel("Δ CAS (%)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.annotate(
        f"MWU p = {mwu_p:.4f}\n(BDR+ vs BDR-)",
        xy=(0.98, 0.97), xycoords="axes fraction",
        fontsize=10, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )
    ax.set_title(
        "Distribución de Δ CAS por grupo clínico",
        fontsize=13, fontweight="bold",
    )

    _save_fig(fig, "fig3_boxplot_delta_cas.png")


def fig4_heatmap_delta_cas(resultados: dict[str, pd.DataFrame]) -> None:
    """Figura 4 — Heatmap Δ CAS (%) por canal y fase (28 sujetos × 4 condiciones)."""
    condicion_labels = {
        "ch1_insp": "CH1-Insp",
        "ch1_esp":  "CH1-Esp",
        "ch2_insp": "CH2-Insp",
        "ch2_esp":  "CH2-Esp",
    }
    # Ordenar sujetos: BDR+ primero, luego BDR-, luego Controls
    df_ref = resultados["ch1_insp"].sort_values(
        "bdr_group",
        key=lambda s: s.map({"BDR+": 0, "BDR-": 1, "Controls": 2}),
    ).reset_index(drop=True)
    subject_order = df_ref["subject_id"].tolist()
    group_map     = dict(zip(df_ref["subject_id"], df_ref["bdr_group"]))

    # Construir matriz (28 × 4)
    matrix = np.zeros((len(subject_order), 4))
    for col_idx, cond in enumerate(condicion_labels.keys()):
        df_cond = resultados[cond].set_index("subject_id")
        for row_idx, sid in enumerate(subject_order):
            if sid in df_cond.index:
                matrix[row_idx, col_idx] = df_cond.loc[sid, "delta_cas"]

    abs_max = max(float(np.abs(matrix).max()), 1.0)

    fig, ax = plt.subplots(figsize=(12, 16))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-abs_max, vmax=abs_max, aspect="auto")
    plt.colorbar(im, ax=ax, label="Δ CAS (%)", fraction=0.03, pad=0.04)

    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(list(condicion_labels.values()), fontsize=11)
    ax.set_yticks(np.arange(len(subject_order)))
    ax.set_yticklabels(subject_order, fontsize=9)

    # Colorear etiquetas de fila según grupo clínico
    for tick_label, sid in zip(ax.get_yticklabels(), subject_order):
        tick_label.set_color(COLORS[group_map[sid]])

    # Anotaciones numéricas en cada celda
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            val = matrix[row_idx, col_idx]
            txt_color = "black" if abs(val) < abs_max * 0.6 else "white"
            ax.text(col_idx, row_idx, f"{val:.1f}",
                    ha="center", va="center", fontsize=7, color=txt_color)

    handles = [mpatches.Patch(color=COLORS[g], label=g) for g in GROUP_ORDER]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              bbox_to_anchor=(1.22, 0.0), title="Grupo")
    ax.set_title(
        "Δ CAS (%) por canal y fase respiratoria",
        fontsize=13, fontweight="bold",
    )

    _save_fig(fig, "fig4_heatmap_delta_cas.png")


def fig5_group_comparison(resultados: dict[str, pd.DataFrame]) -> None:
    """Figura 5 — CAS pre vs post por grupo, canal y fase (2×2 subplots)."""
    subplot_config = [
        ("ch1_insp", "Canal inferior — Inspiración"),
        ("ch1_esp",  "Canal inferior — Espiración"),
        ("ch2_insp", "Canal superior — Inspiración"),
        ("ch2_esp",  "Canal superior — Espiración"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()

    width = 0.3
    x = np.arange(len(GROUP_ORDER))

    for ax, (cond, subtitle) in zip(axes_flat, subplot_config):
        df = resultados[cond]

        for i, grupo in enumerate(GROUP_ORDER):
            gdf = df[df["bdr_group"] == grupo]
            pre_mean  = gdf["cas_rate_pre"].mean()
            post_mean = gdf["cas_rate_post"].mean()
            pre_std   = gdf["cas_rate_pre"].std()
            post_std  = gdf["cas_rate_post"].std()
            color = COLORS[grupo]

            # Barra pre-BD (sólida)
            ax.bar(i - width / 2, pre_mean, width, color=color, alpha=0.85,
                   yerr=pre_std, capsize=4,
                   label=f"{grupo} pre-BD" if cond == "ch1_insp" else "_nolegend_")
            # Barra post-BD (tramado)
            ax.bar(i + width / 2, post_mean, width, color=color, alpha=0.5,
                   hatch="///", yerr=post_std, capsize=4,
                   label=f"{grupo} post-BD" if cond == "ch1_insp" else "_nolegend_")

        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_ORDER, fontsize=10)
        ax.set_ylabel("Tasa CAS (%)", fontsize=10)
        ax.set_title(subtitle, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0)

    # Leyenda compartida en la parte inferior de la figura
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "CAS pre-BD vs post-BD por grupo, canal y fase",
        fontsize=14, fontweight="bold", y=1.01,
    )

    _save_fig(fig, "fig5_group_comparison.png")


def fig6_roc_delta_cas(resultados: dict[str, pd.DataFrame]) -> None:
    """Figura 6 — Curva ROC de Δ CAS como biomarcador de respuesta broncodilatadora."""
    df_all = resultados["all"]
    # Solo pacientes (excluir controles sanos)
    df_patients = df_all[df_all["bdr_group"].isin(["BDR+", "BDR-"])].copy()
    y_true  = (df_patients["bdr_group"] == "BDR+").astype(int).values
    scores  = df_patients["delta_cas"].values

    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"Δ CAS  (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Clasificador aleatorio")
    ax.set_xlabel("Tasa de falsos positivos", fontsize=12)
    ax.set_ylabel("Tasa de verdaderos positivos", fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_title(
        "ROC — Δ CAS como biomarcador de respuesta broncodilatadora",
        fontsize=13, fontweight="bold",
    )

    _save_fig(fig, "fig6_roc_delta_cas_biomarker.png")


# ─────────────────────────────────────────────────────────────────────────────
# PARTE 5 — Informe de limitaciones
# ─────────────────────────────────────────────────────────────────────────────

def print_limitations() -> None:
    """Imprime el informe estructurado de limitaciones del estudio."""
    print("""
=== LIMITACIONES DEL ESTUDIO ===

LIMITACIONES TÉCNICAS:
1. El clasificador fue entrenado únicamente con señales del canal
   inferior (CH1). Las predicciones sobre el canal superior (CH2)
   son extrapolaciones no validadas con ground truth.
2. Las etiquetas de entrenamiento cubren solo 18 de los 23 pacientes
   y ningún control sano, lo que limita la representatividad del
   clasificador.
3. Las features extraídas son handcrafted — un enfoque de
   aprendizaje de representaciones (CNN sobre espectrogramas)
   podría capturar patrones más complejos.
4. La validación LOSO con 18 sujetos produce estimaciones de
   rendimiento con alta varianza entre folds.

LIMITACIONES CLÍNICAS:
1. El análisis delta_CAS asume que el número de CAS detectados es
   proporcional al grado de obstrucción bronquial, lo cual no está
   universalmente validado.
2. La respuesta broncodilatadora se evalúa a los 10-15 minutos
   post-broncodilatador. Efectos más tardíos no están capturados.
3. El dataset es pequeño (23 pacientes) para extraer conclusiones
   estadísticas robustas sobre el biomarcador.
4. Los controles sanos no tienen etiquetas CAS de referencia,
   por lo que su análisis de delta_CAS se basa únicamente en
   predicciones del modelo.
""")


# ─────────────────────────────────────────────────────────────────────────────
# PARTE 6 — Verificación del pipeline completo (end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

def _check(condition: bool, step: str, description: str) -> None:
    """Valida una condición; imprime el fallo y termina si no se cumple."""
    if not condition:
        print(f"VERIFICACIÓN FALLIDA — {step} — {description}")
        sys.exit(1)


def verify_pipeline() -> None:
    """Ejecuta el pipeline completo (step5 → step6 → step7) y verifica cada salida."""
    python = sys.executable

    print("\n" + "=" * 60)
    print("INICIANDO VERIFICACIÓN DEL PIPELINE COMPLETO")
    print("=" * 60)

    # ── Step 5 ──────────────────────────────────────────────────────────────
    print("\n[1/3] Ejecutando step5_features.py ...")
    result = subprocess.run(
        [python, str(PROJECT_ROOT / "src" / "step5_features.py")],
        cwd=str(PROJECT_ROOT),
    )
    _check(result.returncode == 0, "step5", "El script terminó con error")

    s5_res = PROJECT_ROOT / "outputs" / "results" / "step5"
    _check((s5_res / "X_all_features.npy").exists(),     "step5", "X_all_features.npy no encontrado")
    _check((s5_res / "X_labeled_features.npy").exists(), "step5", "X_labeled_features.npy no encontrado")
    _check((s5_res / "y_labeled.npy").exists(),          "step5", "y_labeled.npy no encontrado")

    shape_all = np.load(s5_res / "X_all_features.npy").shape
    _check(shape_all == (14900, 15), "step5", f"X_all_features shape {shape_all} != (14900, 15)")
    shape_lab = np.load(s5_res / "X_labeled_features.npy").shape
    _check(shape_lab == (1923, 15),  "step5", f"X_labeled_features shape {shape_lab} != (1923, 15)")
    shape_y   = np.load(s5_res / "y_labeled.npy").shape
    _check(shape_y == (1923,),       "step5", f"y_labeled shape {shape_y} != (1923,)")

    figs5 = list((PROJECT_ROOT / "outputs" / "figures" / "step5").glob("*.png"))
    _check(len(figs5) >= 3, "step5", f"Solo {len(figs5)} figura(s) en figures/step5/ (mínimo 3)")
    print(f"  OK Step5 — features (14900, 15) extraidas, {len(figs5)} figuras")

    # ── Step 6 ──────────────────────────────────────────────────────────────
    print("\n[2/3] Ejecutando step6_classification.py ...")
    result = subprocess.run(
        [python, str(PROJECT_ROOT / "src" / "step6_classification.py")],
        cwd=str(PROJECT_ROOT),
    )
    _check(result.returncode == 0, "step6", "El script terminó con error")

    s6_res    = PROJECT_ROOT / "outputs" / "results" / "step6"
    pred_path = s6_res / "predictions_all.npz"
    _check(pred_path.exists(), "step6", "predictions_all.npz no encontrado")

    pred = np.load(pred_path, allow_pickle=True)
    _check("y_pred_all" in pred.files, "step6", "y_pred_all no está en predictions_all.npz")
    _check(
        pred["y_pred_all"].shape == (14900,),
        "step6",
        f"y_pred_all shape {pred['y_pred_all'].shape} != (14900,)",
    )

    csvs6 = list(s6_res.glob("*_loso_results.csv"))
    _check(len(csvs6) >= 2, "step6", f"Solo {len(csvs6)} CSV(s) de resultados LOSO (mínimo 2)")

    figs6 = list((PROJECT_ROOT / "outputs" / "figures" / "step6").glob("*.png"))
    _check(len(figs6) >= 5, "step6", f"Solo {len(figs6)} figura(s) en figures/step6/ (mínimo 5)")

    best_model = str(pred["best_model_name"])
    # Leer AUC promedio del mejor modelo desde su CSV de resultados
    loso_auc = float("nan")
    best_csv = s6_res / f"{best_model.lower()}_loso_results.csv"
    if best_csv.exists():
        loso_df = pd.read_csv(best_csv)
        if "auc" in loso_df.columns:
            loso_auc = float(loso_df["auc"].mean())

    print(f"  OK Step6 — mejor modelo = {best_model}, AUC LOSO = {loso_auc:.3f}, {len(figs6)} figuras")

    # ── Step 7 ──────────────────────────────────────────────────────────────
    print("\n[3/3] Ejecutando step7_biomarker_analysis.py --no-verify ...")
    result = subprocess.run(
        [python, str(PROJECT_ROOT / "src" / "step7_biomarker_analysis.py"), "--no-verify"],
        cwd=str(PROJECT_ROOT),
    )
    _check(result.returncode == 0, "step7", "El script terminó con error")

    s7_res = PROJECT_ROOT / "outputs" / "results" / "step7"
    _check((s7_res / "cas_metrics_all.csv").exists(),  "step7", "cas_metrics_all.csv no encontrado")
    _check((s7_res / "group_statistics.csv").exists(), "step7", "group_statistics.csv no encontrado")

    figs7 = list((PROJECT_ROOT / "outputs" / "figures" / "step7").glob("*.png"))
    _check(len(figs7) >= 6, "step7", f"Solo {len(figs7)} figura(s) en figures/step7/ (mínimo 6)")

    n_subjects = len(pd.read_csv(s7_res / "cas_metrics_all.csv"))
    print(f"  OK Step7 — {n_subjects} sujetos analizados, {len(figs7)} figuras")

    # ── Resumen final ────────────────────────────────────────────────────────
    print("\n" + "=" * 48)
    print("PIPELINE COMPLETO — TODAS LAS VERIFICACIONES OK")
    print("=" * 48)
    print(f"Step 5: features extraídas para 14900 señales")
    print(f"Step 6: mejor modelo = {best_model}, AUC LOSO = {loso_auc:.3f}")
    print(f"Step 7: análisis completado para {n_subjects} sujetos")
    print(f"Figuras generadas: outputs/figures/")
    print(f"Resultados guardados: outputs/results/")
    print("=" * 48)


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Ejecuta el análisis de biomarcadores CAS (Partes 1–5)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Parte 1 — Carga de datos
    (y_pred_all, y_prob_all,
     v_subject, v_bd, v_channel, v_phase,
     meta_df, subject_num_to_id, subject_meta, best_model_name) = load_data()

    # Parte 2 — Métricas CAS por sujeto para todas las condiciones
    print("\n--- PARTE 2: Métricas CAS por sujeto ---")
    resultados = run_all_cas_analyses(
        y_pred_all, v_subject, v_bd, v_channel, v_phase,
        subject_num_to_id, subject_meta,
    )

    # Parte 3 — Estadísticos por grupo y pruebas estadísticas
    print("\n--- PARTE 3: Estadísticos por grupo ---")
    stats_df = compute_group_statistics(resultados)
    print_group_table(resultados)

    # Parte 4 — Generación de figuras
    print("\n--- PARTE 4: Generando figuras ---")
    fig1_cas_rate_pre_post_ch1(resultados)
    fig2_delta_cas_per_subject(resultados)
    fig3_boxplot_delta_cas(resultados)
    fig4_heatmap_delta_cas(resultados)
    fig5_group_comparison(resultados)
    fig6_roc_delta_cas(resultados)

    # Parte 5 — Limitaciones
    print_limitations()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 7: análisis clínico de biomarcadores CAS"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Ejecutar solo el análisis (sin lanzar la verificación del pipeline completo)",
    )
    args = parser.parse_args()

    if args.no_verify:
        main()
    else:
        verify_pipeline()
