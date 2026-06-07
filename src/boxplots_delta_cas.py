"""
Genera los boxplots de Delta CAS para los 3 pipelines de forma consistente:
  1. SOTA (4 Marcadores)
  2. Clásico (137 Marcadores)
  3. Híbrido Optimizado (141 Marcadores)
Con fondo blanco y ejes negros completos en los 4 costados.
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Raíz del proyecto
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "outputs", "figures", "comparison")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Datasets de entrada para extraer los deltas
paths = {
    "SOTA (4 Marcadores)": {
        "csv": os.path.join(_PROJECT_ROOT, "outputs", "results", "sota", "clinical_biomarker_results.csv"),
        "p_val": 0.2986,
        "y_col": "delta_cas_percent",
        "y_label": "Delta CAS (%)"
    },
    "Clásico (137 Marcadores)": {
        "csv": os.path.join(_PROJECT_ROOT, "outputs", "results", "step7b", "patient_delta_cas.csv"),
        "p_val": 0.0832, # p-valor univariado/MWU de Ensemble
        "y_col": "delta_cas",
        "y_label": "Delta CAS (pre − post)"
    },
    "Híbrido Optimizado (141 Marcadores)": {
        "csv": os.path.join(_PROJECT_ROOT, "outputs", "results", "optimized", "clinical_biomarker_results.csv"),
        "p_val": 0.0254,
        "y_col": "delta_cas",
        "y_label": "Delta CAS (pre − post)"
    }
}

def generate_delta_cas_boxplots():
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
    
    for name, info in paths.items():
        if not os.path.exists(info["csv"]):
            print(f"Advertencia: No se encontró el archivo {info['csv']}, saltando {name}.")
            continue
            
        df = pd.read_csv(info["csv"])
        
        # Filtrar solo pacientes (excluyendo controles si están presentes)
        if "type" in df.columns:
            df = df[df["type"] == "patient"]
            
        # Reajustar BDR label
        df = df[df["bdr_label"].isin(["BDR+", "BDR-"])]
        
        plt.figure(figsize=(6, 6))
        
        # Dibujar boxplot
        sns.boxplot(data=df, x="bdr_label", y=info["y_col"], 
                    palette={"BDR+": "mediumseagreen", "BDR-": "steelblue"},
                    width=0.5, linewidth=1.2)
        
        # Puntos individuales
        sns.stripplot(data=df, x="bdr_label", y=info["y_col"], 
                      color="black", alpha=0.6, size=6, jitter=0.15)
        
        plt.title(f"Delta CAS según Respuesta Broncodilatadora\n{name} | p-val = {info['p_val']:.4f}", 
                  fontsize=11.5, fontweight="bold", pad=12)
        plt.xlabel("Grupo Clínico (BDR)", fontsize=10)
        plt.ylabel(info["y_label"], fontsize=10)
        plt.grid(True, axis="y", alpha=0.3, linestyle="-")
        
        # Ejes negros en los 4 bordes
        ax = plt.gca()
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("black")
            ax.spines[spine].set_linewidth(1.0)
            
        plt.tight_layout()
        
        filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_path = os.path.join(FIGURES_DIR, f"{filename}_delta_cas_boxplot.png")
        plt.savefig(out_path, dpi=180)
        plt.close()
        print(f"Boxplot de Delta CAS guardado en: {out_path}")

if __name__ == "__main__":
    generate_delta_cas_boxplots()
