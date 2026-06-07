"""
Genera figuras combinadas de comparación de métricas (barra por modelo, en 2x3 subplots)
y matrices de confusión combinadas en una sola imagen (2x3 subplots) sobre fondo blanco.
Diseñado para coincidir estéticamente con el estilo solicitado por el usuario.
"""

from __future__ import annotations

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Raíz del proyecto
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "outputs", "results", "sota")
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "outputs", "figures", "sota")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Mapeo de nombres estéticos y colores para que combine con el estilo de la imagen del usuario
_MODEL_DISPLAY_NAMES = {
    "Logistic Regression L1": "LR-L1",
    "Logistic Regression L2": "LR-L2",
    "SVM Linear": "SVM-Lin",
    "SVM RBF": "SVM-RBF",
    "Random Forest": "RF"
}

# Paleta de colores atractiva de tipo HSL
_COLORS = ["#4682B4", "#D2691E", "#CD5C5C", "#5F9EA0", "#8FBC8F"]

def generate_combined_plots():
    metrics_file = os.path.join(RESULTS_DIR, "metrics_summary.json")
    if not os.path.exists(metrics_file):
        print(f"Error: No se encontró el archivo de métricas: {metrics_file}")
        return
        
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    models = [m["model"] for m in metrics]
    display_names = [_MODEL_DISPLAY_NAMES.get(m, m) for m in models]
    
    # ------------------------------------------------------------------
    # FIGURA 1: COMPARATIVA DE MÉTRICAS (2x3 subplots, fondo blanco)
    # ------------------------------------------------------------------
    metric_keys = ["accuracy", "precision", "sensitivity", "specificity", "f1", "auc"]
    metric_titles = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"]
    
    # Configurar estilo blanco limpio y tipografías
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "grid.color": "#f0f0f0",
        "font.family": "sans-serif",
        "text.color": "#333333",
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333"
    })
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    
    for idx, (key, title) in enumerate(zip(metric_keys, metric_titles)):
        ax = axes_flat[idx]
        values = [m[key] for m in metrics]
        
        # Dibujar barras con bordes elegantes
        bars = ax.bar(display_names, values, color=_COLORS[:len(models)], edgecolor="#666666", width=0.5, linewidth=0.8)
        
        # Añadir valores numéricos encima de cada barra
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # offset vertical de 3 puntos
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="semibold")
            
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_ylabel("Score", fontsize=9.5)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(display_names, rotation=15, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle="-", linewidth=0.5)
        
        # Activar y colorear los ejes en los 4 bordes en negro sólido
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("black")
            ax.spines[spine].set_linewidth(1.0)
        
    fig.suptitle("Comparativa de Métricas (Pipeline SOTA - 4 Marcadores)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_comparison = os.path.join(FIGURES_DIR, "sota_combined_metrics.png")
    plt.savefig(out_comparison, dpi=180)
    plt.close()
    print(f"Nueva figura combinada de metricas guardada en: {out_comparison}")
    
    # ------------------------------------------------------------------
    # FIGURA 2: MATRICES DE CONFUSIÓN COMBINADAS (2x3 subplots, fondo blanco)
    # ------------------------------------------------------------------
    total_cas = 590
    total_nocas = 1333
    
    fig_cm, axes_cm = plt.subplots(2, 3, figsize=(16, 10))
    axes_cm_flat = axes_cm.flatten()
    
    for idx, m in enumerate(metrics):
        ax = axes_cm_flat[idx]
        name = m["model"]
        display_name = _MODEL_DISPLAY_NAMES.get(name, name)
        sens = m["sensitivity"]
        spec = m["specificity"]
        
        tp = int(round(sens * total_cas))
        fn = total_cas - tp
        tn = int(round(spec * total_nocas))
        fp = total_nocas - tn
        
        cm = np.array([[tn, fp], [fn, tp]])
        total_samples = tn + fp + fn + tp
        
        labels_arr = np.array([
            [f"{tn}\n({tn/total_samples*100:.1f}%)", f"{fp}\n({fp/total_samples*100:.1f}%)"],
            [f"{fn}\n({fn/total_samples*100:.1f}%)", f"{tp}\n({tp/total_samples*100:.1f}%)"]
        ])
        
        # Usar mapa de color Blues sobre fondo blanco limpio
        sns.heatmap(cm, annot=labels_arr, fmt="", cmap="Blues", cbar=False,
                    xticklabels=["NO CAS", "CAS"], yticklabels=["NO CAS", "CAS"],
                    ax=ax, linewidths=1.5, linecolor="white", annot_kws={"fontsize": 11, "fontweight": "semibold"})
        
        # Habilitar ejes negros en los 4 costados de las matrices de confusión
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("black")
            ax.spines[spine].set_linewidth(1.0)
            
        ax.set_title(f"{display_name}", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel("Clase Real", fontsize=9.5)
        ax.set_xlabel("Clase Predicha", fontsize=9.5)
        
    # Ocultar el último subplot (el 6º) ya que son 5 modelos
    if len(metrics) < 6:
        axes_cm_flat[-1].set_visible(False)
        
    fig_cm.suptitle("Matrices de Confusión de Modelos (Pipeline SOTA)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_cm = os.path.join(FIGURES_DIR, "sota_combined_confusion_matrices.png")
    plt.savefig(out_cm, dpi=180)
    plt.close()
    print(f"Nueva figura combinada de matrices de confusion guardada en: {out_cm}")

if __name__ == "__main__":
    generate_combined_plots()
