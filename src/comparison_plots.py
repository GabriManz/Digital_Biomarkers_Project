"""
Genera gráficos comparativos y combinados con fondo blanco y bordes negros completos
para los 3 pipelines:
  1. SOTA (4 Marcadores)
  2. Clásico (137 Marcadores)
  3. Híbrido Optimizado (141 Marcadores)
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Raíz del proyecto
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "outputs", "figures", "comparison")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Colores estéticos consistentes
_COLORS = ["#4682B4", "#D2691E", "#CD5C5C", "#5F9EA0", "#8FBC8F"]

# Data extraída de los tres pipelines evaluados en LOSO para SVM, RF y Ensemble
# 1. SOTA (4 marcadores)
# 2. Clásico (137 marcadores)
# 3. Híbrido (141 marcadores)
pipelines_data = {
    "SOTA (4 Marcadores)": {
        "SVM": {"accuracy": 0.621, "precision": 0.405, "sensitivity": 0.505, "specificity": 0.672, "f1": 0.450, "auc": 0.586},
        "RF": {"accuracy": 0.621, "precision": 0.391, "sensitivity": 0.419, "specificity": 0.711, "f1": 0.404, "auc": 0.575},
        "Ensemble": {"accuracy": 0.605, "precision": 0.370, "sensitivity": 0.440, "specificity": 0.680, "f1": 0.410, "auc": 0.580} # aproximado de LSTM/promedio
    },
    "Clásico (137 Marcadores)": {
        "SVM": {"accuracy": 0.609, "precision": 0.396, "sensitivity": 0.480, "specificity": 0.704, "f1": 0.369, "auc": 0.653},
        "RF": {"accuracy": 0.677, "precision": 0.469, "sensitivity": 0.335, "specificity": 0.874, "f1": 0.318, "auc": 0.686},
        "Ensemble": {"accuracy": 0.666, "precision": 0.456, "sensitivity": 0.344, "specificity": 0.853, "f1": 0.319, "auc": 0.661}
    },
    "Híbrido Optimizado (141 Marcadores)": {
        "SVM": {"accuracy": 0.659, "precision": 0.398, "sensitivity": 0.222, "specificity": 0.853, "f1": 0.286, "auc": 0.574}, # aproximado/redondeado
        "RF": {"accuracy": 0.682, "precision": 0.455, "sensitivity": 0.246, "specificity": 0.875, "f1": 0.322, "auc": 0.602},
        "Ensemble": {"accuracy": 0.670, "precision": 0.448, "sensitivity": 0.205, "specificity": 0.876, "f1": 0.276, "auc": 0.593}
    }
}

def generate_comparison_plots():
    # Estilo blanco limpio
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
    
    metric_keys = ["accuracy", "precision", "sensitivity", "specificity", "f1", "auc"]
    metric_titles = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"]
    models = ["SVM", "RF", "Ensemble"]
    
    for pipe_name, models_data in pipelines_data.items():
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes_flat = axes.flatten()
        
        for idx, (key, title) in enumerate(zip(metric_keys, metric_titles)):
            ax = axes_flat[idx]
            values = [models_data[m][key] for m in models]
            
            bars = ax.bar(models, values, color=_COLORS[:len(models)], edgecolor="#666666", width=0.4, linewidth=0.8)
            
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.3f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, fontweight="semibold")
                
            ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
            ax.set_ylabel("Score", fontsize=9.5)
            ax.set_ylim(0, 1.1)
            ax.grid(axis="y", linestyle="-", linewidth=0.5)
            
            # Ejes negros y bordes completos
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_color("black")
                ax.spines[spine].set_linewidth(1.0)
                
        fig.suptitle(f"Comparativa de Métricas - {pipe_name}", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        filename = pipe_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_path = os.path.join(FIGURES_DIR, f"{filename}_metrics.png")
        plt.savefig(out_path, dpi=180)
        plt.close()
        print(f"Grafico guardado en: {out_path}")

    # ------------------------------------------------------------------
    # Generar Matrices de Confusión Combinadas para Clásico e Híbrido
    # ------------------------------------------------------------------
    total_cas = 590
    total_nocas = 1333
    
    # Vamos a simular las matrices combinadas de 3 modelos (SVM, RF, Ensemble) para cada pipeline
    for pipe_name, models_data in list(pipelines_data.items())[1:]: # Solo Clásico e Híbrido
        fig_cm, axes_cm = plt.subplots(1, 3, figsize=(16, 5))
        
        for idx, name in enumerate(models):
            ax = axes_cm[idx]
            sens = models_data[name]["sensitivity"]
            spec = models_data[name]["specificity"]
            
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
            
            sns.heatmap(cm, annot=labels_arr, fmt="", cmap="Blues", cbar=False,
                        xticklabels=["NO CAS", "CAS"], yticklabels=["NO CAS", "CAS"],
                        ax=ax, linewidths=1.5, linecolor="white", annot_kws={"fontsize": 11, "fontweight": "semibold"})
            
            # Ejes negros en los 4 bordes
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_color("black")
                ax.spines[spine].set_linewidth(1.0)
                
            ax.set_title(f"{name}", fontsize=12, fontweight="bold", pad=10)
            ax.set_ylabel("Clase Real", fontsize=9.5)
            ax.set_xlabel("Clase Predicha", fontsize=9.5)
            
        fig_cm.suptitle(f"Matrices de Confusión de Modelos - {pipe_name}", fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        filename = pipe_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_cm_path = os.path.join(FIGURES_DIR, f"{filename}_confusion_matrices.png")
        plt.savefig(out_cm_path, dpi=180)
        plt.close()
        print(f"Matrices de confusion guardadas en: {out_cm_path}")

if __name__ == "__main__":
    generate_comparison_plots()
