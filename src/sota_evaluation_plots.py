"""
Script auxiliar para generar gráficos de evaluación para el pipeline SOTA minimalista (4 marcadores).
Genera:
  1. Gráficas de barras comparativas de Accuracy, Recall (Sensibilidad), Precisión y F1-score por modelo.
  2. Matrices de confusión para cada uno de los modelos evaluados en LOSO.
"""

from __future__ import annotations

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Raíz del proyecto
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "outputs", "results", "sota")
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "outputs", "figures", "sota")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Cargamos el archivo de métricas de sota
metrics_file = os.path.join(RESULTS_DIR, "metrics_summary.json")

def generate_evaluation_plots():
    if not os.path.exists(metrics_file):
        print(f"Error: No se encontró el archivo de métricas: {metrics_file}")
        print("Por favor, ejecuta primero: python src/sota_pipeline.py")
        return
        
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    models = [m["model"] for m in metrics]
    accuracy = [m["accuracy"] for m in metrics]
    recall = [m["sensitivity"] for m in metrics]
    precision = [m["precision"] for m in metrics]
    f1 = [m["f1"] for m in metrics]
    auc = [m["auc"] for m in metrics]
    
    # ------------------------------------------------------------------
    # 1. Gráfico de barras comparativo de métricas
    # ------------------------------------------------------------------
    x = np.arange(len(models))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - 2*width, accuracy, width, label="Accuracy", color="#3498db")
    ax.bar(x - width, recall, width, label="Recall (Sens)", color="#2ecc71")
    ax.bar(x, precision, width, label="Precision", color="#e67e22")
    ax.bar(x + width, f1, width, label="F1-score", color="#e74c3c")
    ax.bar(x + 2*width, auc, width, label="AUC-ROC", color="#9b59b6")
    
    ax.set_ylabel("Valor (0.0 - 1.0)")
    ax.set_title("Comparación de Métricas por Modelo - Pipeline SOTA (4 Marcadores)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "sota_metrics_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Grafico de comparacion de metricas guardado en: {out_path}")
    
    # ------------------------------------------------------------------
    # 2. Re-calcular matrices de confusión simuladas a partir de métricas
    # ------------------------------------------------------------------
    # El conjunto etiquetado tiene N = 1923 segmentos: CAS = 590, NO_CAS = 1333
    total_cas = 590
    total_nocas = 1333
    
    for m in metrics:
        name = m["model"]
        sens = m["sensitivity"]
        spec = m["specificity"]
        
        # Calcular TP, FN, TN, FP a partir de sensibilidad y especificidad
        tp = int(round(sens * total_cas))
        fn = total_cas - tp
        tn = int(round(spec * total_nocas))
        fp = total_nocas - tn
        
        cm = np.array([[tn, fp], [fn, tp]])
        total_samples = tn + fp + fn + tp
        
        # Crear etiquetas personalizadas que incluyan la cantidad absoluta y el porcentaje
        labels_arr = np.array([
            [f"{tn}\n({tn/total_samples*100:.1f}%)", f"{fp}\n({fp/total_samples*100:.1f}%)"],
            [f"{fn}\n({fn/total_samples*100:.1f}%)", f"{tp}\n({tp/total_samples*100:.1f}%)"]
        ])
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=labels_arr, fmt="", cmap="Blues", cbar=False,
                    xticklabels=["NO CAS", "CAS"], yticklabels=["NO CAS", "CAS"])
        plt.title(f"Matriz de Confusión: {name}\nPipeline SOTA (4 Marcadores)")
        plt.ylabel("Clase Real")
        plt.xlabel("Clase Predicha")
        plt.tight_layout()
        
        model_filename = name.lower().replace(" ", "_")
        cm_path = os.path.join(FIGURES_DIR, f"confusion_matrix_{model_filename}.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"Matriz de confusion para {name} guardada en: {cm_path}")

if __name__ == "__main__":
    generate_evaluation_plots()
