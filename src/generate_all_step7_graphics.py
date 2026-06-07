import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score

# Ensure stdout uses UTF-8 to avoid encoding errors in windows consoles/logs
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

# Constants
BD_PRE      = 1
BD_POST     = 2
CH_LOWER    = 1
CH_UPPER    = 2
PHASE_INSP  = 1
PHASE_ESP   = 2

COLORS      = {"BDR+": "green", "BDR-": "steelblue", "Controls": "gray"}
GROUP_ORDER = ["BDR+", "BDR-", "Controls"]

# Paths
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (p for p in [_HERE.parent, _HERE] if (p / "proy_labels.mat").exists()),
    _HERE.parent,
)

def load_base_metadata():
    # Load dataset.npz for the segment-level metadata
    ds_path = PROJECT_ROOT / "outputs" / "results" / "step4" / "dataset.npz"
    ds = np.load(ds_path)
    v_subject = ds["v_subject"]
    v_bd      = ds["v_bd"]
    v_channel = ds["v_channel"]
    v_phase   = ds["v_phase"]

    # Load subject metadata
    meta_path = PROJECT_ROOT / "Data" / "database" / "subject_metadata.csv"
    meta_df = pd.read_csv(meta_path)

    # Subject mappings
    subject_num_to_id = {
        int(row["subject_num"]): row["subject_id"]
        for _, row in meta_df.iterrows()
    }
    subject_meta = {
        row["subject_id"]: {"type": row["type"], "bdr_label": row["bdr_label"]}
        for _, row in meta_df.iterrows()
    }

    return v_subject, v_bd, v_channel, v_phase, subject_num_to_id, subject_meta

def compute_cas_metrics(y_pred, v_subject, v_bd, v_channel, v_phase, subject_num_to_id, subject_meta, channel, phase):
    mask_ch = np.ones(len(v_subject), dtype=bool) if channel == "both" else (v_channel == channel)
    mask_ph = np.ones(len(v_subject), dtype=bool) if phase == "both" else (v_phase == phase)
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
        
        delta_cas = cas_rate_pre - cas_rate_post

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

def run_all_cas_analyses(y_pred, v_subject, v_bd, v_channel, v_phase, subject_num_to_id, subject_meta, block_results_dir):
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

    resultados = {}
    for nombre, ch, ph in condiciones:
        df = compute_cas_metrics(y_pred, v_subject, v_bd, v_channel, v_phase, subject_num_to_id, subject_meta, ch, ph)
        csv_path = block_results_dir / f"cas_metrics_{nombre}.csv"
        df.to_csv(csv_path, index=False)
        resultados[nombre] = df
    return resultados

def compute_group_statistics(resultados, block_results_dir):
    filas_stats = []
    for condicion, df in resultados.items():
        grupos = {g: df[df["bdr_group"] == g] for g in GROUP_ORDER}
        bdr_plus  = grupos["BDR+"]["delta_cas"].values
        bdr_minus = grupos["BDR-"]["delta_cas"].values
        
        if len(bdr_plus) > 0 and len(bdr_minus) > 0:
            _, mwu_p = stats.mannwhitneyu(bdr_plus, bdr_minus, alternative="greater")
        else:
            mwu_p = np.nan

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
    out_path = block_results_dir / "group_statistics.csv"
    stats_df.to_csv(out_path, index=False)
    return stats_df

def save_fig(fig, block_figs_dir, filename):
    fig.tight_layout()
    out_path = block_figs_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def generate_block_plots(resultados, block_figs_dir, block_name, best_model_name):
    # Figure 1: CAS rate pre-BD and post-BD per subject (CH1)
    df_ch1 = resultados["ch1"].sort_values("subject_id").reset_index(drop=True)
    subjects = df_ch1["subject_id"].tolist()
    bar_colors = [COLORS[g] for g in df_ch1["bdr_group"]]
    x = np.arange(len(subjects))
    width = 0.7

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    for ax, col, ylabel in zip(axes, ["cas_rate_pre", "cas_rate_post"], ["CAS pre-BD (%)", "CAS post-BD (%)"]):
        ax.bar(x, df_ch1[col].values, color=bar_colors, width=width, edgecolor="white", linewidth=0.5)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0, max(df_ch1[col].max() * 1.18, 5.0))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(subjects, rotation=45, ha="right", fontsize=9)
    handles = [mpatches.Patch(color=COLORS[g], label=g) for g in GROUP_ORDER]
    axes[0].legend(handles=handles, loc="upper right", fontsize=9)
    axes[0].set_title(f"Tasa de CAS pre-BD y post-BD por sujeto (Canal inferior) - Bloque: {block_name.upper()} ({best_model_name})", fontsize=13, fontweight="bold")
    save_fig(fig, block_figs_dir, "fig1_cas_rate_pre_post_ch1.png")

    # Figure 2: Delta CAS per subject (ordered)
    df_all = resultados["all"].sort_values("delta_cas", ascending=False).reset_index(drop=True)
    bar_colors = [COLORS[g] for g in df_all["bdr_group"]]
    fig, ax = plt.subplots(figsize=(10, 12))
    y = np.arange(len(df_all))
    ax.barh(y, df_all["delta_cas"].values, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(df_all["subject_id"].tolist(), fontsize=9)
    ax.set_xlabel("Δ CAS (%)", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.set_title(f"Cambio en CAS tras broncodilatador — Δ CAS (%) - Bloque: {block_name.upper()} ({best_model_name})", fontsize=13, fontweight="bold")
    save_fig(fig, block_figs_dir, "fig2_delta_cas_per_subject.png")

    # Figure 3: Boxplot of Delta CAS by group
    bdr_plus  = df_all[df_all["bdr_group"] == "BDR+"]["delta_cas"].values
    bdr_minus = df_all[df_all["bdr_group"] == "BDR-"]["delta_cas"].values
    if len(bdr_plus) > 0 and len(bdr_minus) > 0:
        _, mwu_p = stats.mannwhitneyu(bdr_plus, bdr_minus, alternative="greater")
    else:
        mwu_p = np.nan
    data_list = [df_all[df_all["bdr_group"] == g]["delta_cas"].values for g in GROUP_ORDER]

    fig, ax = plt.subplots(figsize=(10, 7))
    bp = ax.boxplot(data_list, patch_artist=True, widths=0.4, medianprops={"color": "black", "linewidth": 2})
    for patch, grupo in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor(COLORS[grupo])
        patch.set_alpha(0.6)
    
    rng = np.random.default_rng(42)
    for i, (grupo, vals) in enumerate(zip(GROUP_ORDER, data_list), start=1):
        if len(vals) > 0:
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(i + jitter, vals, color=COLORS[grupo], s=40, zorder=3, edgecolors="black", linewidths=0.5, alpha=0.85)

    ax.axhline(0, color="gray", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(GROUP_ORDER, fontsize=12)
    ax.set_ylabel("Δ CAS (%)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if not np.isnan(mwu_p):
        ax.annotate(f"MWU p = {mwu_p:.4f}\n(BDR+ vs BDR-)", xy=(0.98, 0.97), xycoords="axes fraction", fontsize=10, ha="right", va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    ax.set_title(f"Distribución de Δ CAS por grupo clínico - Bloque: {block_name.upper()} ({best_model_name})", fontsize=13, fontweight="bold")
    save_fig(fig, block_figs_dir, "fig3_boxplot_delta_cas.png")

    # Figure 4: Heatmap of Delta CAS (%) by channel and phase
    condicion_labels = {
        "ch1_insp": "CH1-Insp",
        "ch1_esp":  "CH1-Esp",
        "ch2_insp": "CH2-Insp",
        "ch2_esp":  "CH2-Esp",
    }
    df_ref = resultados["ch1_insp"].sort_values("bdr_group", key=lambda s: s.map({"BDR+": 0, "BDR-": 1, "Controls": 2})).reset_index(drop=True)
    subject_order = df_ref["subject_id"].tolist()
    group_map = dict(zip(df_ref["subject_id"], df_ref["bdr_group"]))

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

    for tick_label, sid in zip(ax.get_yticklabels(), subject_order):
        tick_label.set_color(COLORS[group_map[sid]])

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            val = matrix[row_idx, col_idx]
            txt_color = "black" if abs(val) < abs_max * 0.6 else "white"
            ax.text(col_idx, row_idx, f"{val:.1f}", ha="center", va="center", fontsize=7, color=txt_color)

    ax.legend(handles=handles, loc="lower right", fontsize=9, bbox_to_anchor=(1.22, 0.0), title="Grupo")
    ax.set_title(f"Δ CAS (%) por canal y fase respiratoria - Bloque: {block_name.upper()} ({best_model_name})", fontsize=13, fontweight="bold")
    save_fig(fig, block_figs_dir, "fig4_heatmap_delta_cas.png")

    # Figure 5: Group comparison (2x2 subplots)
    subplot_config = [
        ("ch1_insp", "Canal inferior — Inspiración"),
        ("ch1_esp",  "Canal inferior — Espiración"),
        ("ch2_insp", "Canal superior — Inspiración"),
        ("ch2_esp",  "Canal superior — Espiración"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()
    width_bar = 0.3
    x_bar = np.arange(len(GROUP_ORDER))

    for ax, (cond, subtitle) in zip(axes_flat, subplot_config):
        df = resultados[cond]
        for i, grupo in enumerate(GROUP_ORDER):
            gdf = df[df["bdr_group"] == grupo]
            pre_mean  = gdf["cas_rate_pre"].mean()
            post_mean = gdf["cas_rate_post"].mean()
            pre_std   = gdf["cas_rate_pre"].std()
            post_std  = gdf["cas_rate_post"].std()
            color = COLORS[grupo]

            ax.bar(i - width_bar / 2, pre_mean, width_bar, color=color, alpha=0.85, yerr=pre_std, capsize=4,
                   label=f"{grupo} pre-BD" if cond == "ch1_insp" else "_nolegend_")
            ax.bar(i + width_bar / 2, post_mean, width_bar, color=color, alpha=0.5, hatch="///", yerr=post_std, capsize=4,
                   label=f"{grupo} post-BD" if cond == "ch1_insp" else "_nolegend_")

        ax.set_xticks(x_bar)
        ax.set_xticklabels(GROUP_ORDER, fontsize=10)
        ax.set_ylabel("Tasa CAS (%)", fontsize=10)
        ax.set_title(subtitle, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0)

    handles_legend, labels_legend = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles_legend, labels_legend, loc="lower center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"CAS pre-BD vs post-BD por grupo, canal y fase - Bloque: {block_name.upper()} ({best_model_name})", fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, block_figs_dir, "fig5_group_comparison.png")

    # Figure 6: ROC Curve of Delta CAS as BDR Biomarker
    df_patients = df_all[df_all["bdr_group"].isin(["BDR+", "BDR-"])].copy()
    if len(df_patients) > 0 and len(np.unique(df_patients["bdr_group"])) > 1:
        y_true = (df_patients["bdr_group"] == "BDR+").astype(int).values
        scores = df_patients["delta_cas"].values
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
    else:
        fpr, tpr = [0, 1], [0, 1]
        auc = 0.5

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"Δ CAS (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Clasificador aleatorio")
    ax.set_xlabel("Tasa de falsos positivos", fontsize=12)
    ax.set_ylabel("Tasa de verdaderos positivos", fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_title(f"ROC — Δ CAS como biomarcador de BDR - Bloque: {block_name.upper()} ({best_model_name})", fontsize=13, fontweight="bold")
    save_fig(fig, block_figs_dir, "fig6_roc_delta_cas_biomarker.png")

def main():
    blocks = {
        "sota": PROJECT_ROOT / "outputs" / "results" / "sota" / "predictions_all.npz",
        "clasico": PROJECT_ROOT / "outputs" / "results" / "step6" / "predictions_all.npz",
        "hibrido": PROJECT_ROOT / "outputs" / "results" / "optimized" / "predictions_all.npz",
        "adria": PROJECT_ROOT / "outputs" / "results" / "adria" / "predictions_all.npz",
    }

    print("Cargando metadatos base...")
    v_subject, v_bd, v_channel, v_phase, subject_num_to_id, subject_meta = load_base_metadata()

    for block_name, pred_path in blocks.items():
        print(f"\n==================================================")
        print(f"PROCESANDO BLOQUE: {block_name.upper()}")
        print(f"==================================================")
        
        if not pred_path.exists():
            print(f"ADVERTENCIA: No se encontró {pred_path}. Saltando bloque...")
            continue
            
        pred = np.load(pred_path, allow_pickle=True)
        y_pred_all = pred["y_pred_all"]
        best_model_name = str(pred["best_model_name"]) if "best_model_name" in pred.files else "Modelo"
        
        block_results_dir = PROJECT_ROOT / "outputs" / "results" / "step7" / block_name
        block_figs_dir = PROJECT_ROOT / "outputs" / "figures" / "presentation" / block_name
        
        block_results_dir.mkdir(parents=True, exist_ok=True)
        block_figs_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"1. Calculando métricas de CAS por sujeto...")
        resultados = run_all_cas_analyses(y_pred_all, v_subject, v_bd, v_channel, v_phase, subject_num_to_id, subject_meta, block_results_dir)
        
        print(f"2. Calculando estadísticas de grupo...")
        compute_group_statistics(resultados, block_results_dir)
        
        print(f"3. Generando las 6 figuras del bloque...")
        generate_block_plots(resultados, block_figs_dir, block_name, best_model_name)
        
        print(f"¡Bloque {block_name.upper()} completado!")

    # Organizar todos los archivos sueltos existentes en presentation/ en sus respectivas subcarpetas
    pres_dir = PROJECT_ROOT / "outputs" / "figures" / "presentation"
    print("\nOrganizando archivos sueltos en outputs/figures/presentation/...")
    for file_path in pres_dir.glob("*.*"):
        if file_path.is_file():
            name = file_path.name.lower()
            target_subfolder = None
            if name.startswith("sota_"):
                target_subfolder = pres_dir / "sota"
            elif name.startswith("clasico_") or name.startswith("classico_"):
                target_subfolder = pres_dir / "clasico"
            elif name.startswith("hibrido_"):
                target_subfolder = pres_dir / "hibrido"
            elif name.startswith("adria_"):
                target_subfolder = pres_dir / "adria"

            if target_subfolder:
                target_subfolder.mkdir(parents=True, exist_ok=True)
                new_path = target_subfolder / file_path.name
                file_path.rename(new_path)
                print(f"  Movido: {file_path.name} -> {target_subfolder.name}/")

if __name__ == "__main__":
    main()
