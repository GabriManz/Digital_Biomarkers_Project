import shutil
from pathlib import Path

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    print("=== COPIANDO IMÁGENES SELECCIONADAS PARA LAS DIAPOSITIVAS ===")
    
    # Directorio de salida
    target_dir = PROJECT_ROOT / "outputs" / "figures" / "presentation" / "slide_images"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapeo de nombre de destino a ruta de origen
    images_to_copy = {
        "slide1_1_senal_cruda_vs_preprocesada.png": PROJECT_ROOT / "outputs" / "figures" / "step2" / "fig1_senal_cruda_vs_preprocesada.png",
        "slide1_2_psd_antes_despues.png": PROJECT_ROOT / "outputs" / "figures" / "step2" / "fig4_psd_antes_despues.png",
        "slide2_segmentacion_maniobra_BDRpos.png": PROJECT_ROOT / "outputs" / "figures" / "step3" / "fig1_segmentacion_maniobra_BDRpos.png",
        "slide3_feature_means_cas_vs_nocas.png": PROJECT_ROOT / "outputs" / "figures" / "step5" / "fig3_feature_means_cas_vs_nocas.png",
        "slide4_feature_correlation.png": PROJECT_ROOT / "outputs" / "figures" / "step5" / "fig2_feature_correlation.png",
        "slide6_hibrido_presentation_metrics.png": PROJECT_ROOT / "outputs" / "figures" / "presentation" / "hibrido" / "hibrido_presentation_metrics.png",
        "slide7_espectrograma_antes_despues.png": PROJECT_ROOT / "outputs" / "figures" / "step2" / "fig5_espectrograma_antes_despues.png",
        "slide8_adria_presentation_metrics.png": PROJECT_ROOT / "outputs" / "figures" / "presentation" / "adria" / "adria_presentation_metrics.png",
        "slide9_boxplot_delta_cas.png": PROJECT_ROOT / "outputs" / "figures" / "presentation" / "hibrido" / "hibrido_presentation_delta_cas.png",
        "slide10_heatmap_delta_cas.png": PROJECT_ROOT / "outputs" / "figures" / "presentation" / "hibrido" / "fig4_heatmap_delta_cas.png",
    }
    
    for dest_name, src_path in images_to_copy.items():
        if src_path.exists():
            dest_path = target_dir / dest_name
            shutil.copy(src_path, dest_path)
            print(f"  Copiado: {src_path.name} -> slide_images/{dest_name}")
        else:
            print(f"  ADVERTENCIA: No se encontró {src_path}")
            
    print(f"\n¡Proceso completado! Todas las imágenes están agrupadas en: {target_dir}")

if __name__ == "__main__":
    main()
