import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configurar estilo premium limpio (fondo blanco, bordes negros)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "grid.color": "#e2e2e2",
    "font.family": "sans-serif",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222"
})

def plot_scatter_pipeline(df, title, out_path, include_controls=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Separar BDR+ y BDR-
    df_pos = df[df["bdr_label"] == "BDR+"]
    df_neg = df[df["bdr_label"] == "BDR-"]
    
    # Graficar BDR-
    ax.scatter(
        df_neg["cas_rate_pre"] * 100, 
        df_neg["cas_rate_post"] * 100, 
        color="#4f81bd", 
        edgecolor="#222222", 
        s=90, 
        alpha=0.9, 
        label=f"BDR- (n={len(df_neg)})",
        zorder=3
    )
    
    # Graficar BDR+
    ax.scatter(
        df_pos["cas_rate_pre"] * 100, 
        df_pos["cas_rate_post"] * 100, 
        color="#76933c", 
        edgecolor="#222222", 
        s=90, 
        alpha=0.9, 
        label=f"BDR+ (n={len(df_pos)})",
        zorder=3
    )
    
    # Graficar Controles si están disponibles
    if include_controls and "type" in df.columns:
        df_ctrl = df[df["type"] == "control"]
        ax.scatter(
            df_ctrl["cas_rate_pre"] * 100, 
            df_ctrl["cas_rate_post"] * 100, 
            color="#a6a6a6", 
            edgecolor="#222222", 
            s=90, 
            alpha=0.8, 
            label=f"Controles (n={len(df_ctrl)})",
            zorder=3
        )
        
    # Diagonal y = x (línea de no cambio)
    max_val = max(df["cas_rate_pre"].max() * 100, df["cas_rate_post"].max() * 100) * 1.15
    ax.plot([0, max_val], [0, max_val], color="#222222", linestyle="--", linewidth=1.2, label="Línea de No Cambio", zorder=2)
    
    # Configuración de límites y etiquetas
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Tasa de CAS Pre-BD (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Tasa de CAS Post-BD (%)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    
    # Estilo de rejilla y caja
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_color("#222222")
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_linewidth(1.0)
    
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Scatter plot guardado en: {out_path}")

def main():
    root_dir = r"c:\DATA\01_Proyectos\Master\Digital_Biomarkers\Project"
    out_dir = os.path.join(root_dir, "outputs", "figures", "presentation")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Cargar y graficar SOTA
    sota_csv = os.path.join(root_dir, "outputs", "results", "sota", "clinical_biomarker_results.csv")
    if os.path.exists(sota_csv):
        df_sota = pd.read_csv(sota_csv)
        plot_scatter_pipeline(
            df_sota, 
            "Pre vs Post CAS Rate - SOTA (4 Features)", 
            os.path.join(out_dir, "sota_scatter_pre_post.png")
        )
        
    # 2. Cargar y graficar Clásico
    classic_csv = os.path.join(root_dir, "outputs", "results", "step7b", "patient_delta_cas.csv")
    if os.path.exists(classic_csv):
        df_classic = pd.read_csv(classic_csv)
        plot_scatter_pipeline(
            df_classic, 
            "Pre vs Post CAS Rate - Clásico (137 Features)", 
            os.path.join(out_dir, "clasico_scatter_pre_post.png"),
            include_controls=True
        )
        
    # 3. Cargar y graficar Híbrido
    hybrid_csv = os.path.join(root_dir, "outputs", "results", "optimized", "clinical_biomarker_results.csv")
    if os.path.exists(hybrid_csv):
        df_hybrid = pd.read_csv(hybrid_csv)
        # Vamos a intentar añadir los controles del Híbrido si podemos. 
        # Cargar predictions_all.npz para extraer tasas de controles
        hybrid_npz = os.path.join(root_dir, "outputs", "results", "optimized", "predictions_all.npz")
        step4_npz = os.path.join(root_dir, "outputs", "results", "step4", "dataset.npz")
        
        if os.path.exists(hybrid_npz) and os.path.exists(step4_npz):
            try:
                preds = np.load(hybrid_npz)
                y_pred = preds["y_pred_all"]
                
                s4 = np.load(step4_npz)
                v_subject = s4["v_subject"].astype(int)
                v_bd = s4["v_bd"].astype(int)
                
                # Para los 5 controles (sujetos 24 a 28)
                control_rows = []
                for c_num in range(24, 29):
                    c_id = f"C{c_num - 23}"
                    # Filtros
                    idx_pre = (v_subject == c_num) & (v_bd == 1)
                    idx_post = (v_subject == c_num) & (v_bd == 2)
                    
                    rate_pre = y_pred[idx_pre].mean() if idx_pre.sum() > 0 else 0.0
                    rate_post = y_pred[idx_post].mean() if idx_post.sum() > 0 else 0.0
                    
                    control_rows.append({
                        "subject_id": c_id,
                        "bdr_label": "BDR-", # para el scatter plot
                        "type": "control",
                        "cas_rate_pre": rate_pre,
                        "cas_rate_post": rate_post,
                        "delta_cas": rate_pre - rate_post
                    })
                
                df_ctrl = pd.DataFrame(control_rows)
                df_hybrid["type"] = "patient"
                df_hybrid_extended = pd.concat([df_hybrid, df_ctrl], ignore_index=True)
                
                plot_scatter_pipeline(
                    df_hybrid_extended, 
                    "Pre vs Post CAS Rate - Híbrido (141 Features)", 
                    os.path.join(out_dir, "hibrido_scatter_pre_post.png"),
                    include_controls=True
                )
            except Exception as e:
                print(f"Error procesando controles de Híbrido: {e}")
                plot_scatter_pipeline(
                    df_hybrid, 
                    "Pre vs Post CAS Rate - Híbrido (141 Features)", 
                    os.path.join(out_dir, "hibrido_scatter_pre_post.png")
                )
        else:
            plot_scatter_pipeline(
                df_hybrid, 
                "Pre vs Post CAS Rate - Híbrido (141 Features)", 
                os.path.join(out_dir, "hibrido_scatter_pre_post.png")
            )

if __name__ == "__main__":
    main()
