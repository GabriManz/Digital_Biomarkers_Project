"""
Extracción de features combinadas: 137 clásicos + 4 SOTA = 141 features.
Carga los clásicos pre-calculados para acelerar el proceso y calcula los 4 SOTA
para los 14 900 segmentos.
"""

from __future__ import annotations

import json
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Localizar la raíz
def _find_project_root() -> Path:
    candidate = Path(__file__).resolve().parent.parent
    for _ in range(6):
        if (candidate / "proy_labels.mat").exists():
            return candidate
        candidate = candidate.parent
    return Path(__file__).resolve().parent.parent

_PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from step4_dataset import build_dataset, METADATA_CSV, FS_TARGET
from sota_pipeline import extract_sota_features_global

# Paths
CACHE_DIR = _PROJECT_ROOT / "outputs" / "results" / "step5"
COMBINED_DIR = _PROJECT_ROOT / "outputs" / "results" / "combined"
COMBINED_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=== EXTRACCIÓN Y FUSIÓN DE FEATURES (137 CLÁSICOS + 4 SOTA) ===")
    
    # 1. Cargar clásicos
    if not (CACHE_DIR / "X_all_features.npy").exists():
        print("Error: No se encontraron los features clásicos del step5. Ejecuta primero step5_features.py")
        sys.exit(1)
        
    print("Cargando features clásicos...")
    X_all_classic = np.load(CACHE_DIR / "X_all_features.npy")
    X_labeled_classic = np.load(CACHE_DIR / "X_labeled_features.npy")
    y_labeled = np.load(CACHE_DIR / "y_labeled.npy")
    groups_labeled = np.load(CACHE_DIR / "groups_labeled.npy")
    
    with open(CACHE_DIR / "feature_names.json", "r", encoding="utf-8") as fh:
        classic_names = json.load(fh)
        
    print(f"  Clásicos cargados: X_all={X_all_classic.shape}, X_labeled={X_labeled_classic.shape}")
    
    # 2. Cargar señales para extraer las 4 SOTA
    # El caché de step8 tiene solo las 1923 señales etiquetadas. Necesitamos las 14 900 señales totales.
    # Por lo tanto, cargamos usando build_dataset.
    print("Cargando las 14 900 señales reconstruyendo el dataset...")
    from step4_dataset import _load_metadata, _build_subject_list
    metadata = _load_metadata(str(_PROJECT_ROOT / "Data" / "database" / "subject_metadata.csv"))
    subjects = _build_subject_list(metadata)
    all_signals, _, _, _, _, _ = build_dataset(subjects)
        
    print(f"Cargadas {len(all_signals)} señales.")
    
    # 3. Calcular las 4 SOTA para todas las señales
    print("Calculando las 4 features SOTA por segmento...")
    t0 = time.time()
    X_all_sota = []
    n_total = len(all_signals)
    for idx, sig in enumerate(all_signals):
        if (idx + 1) % 1000 == 0 or idx == 0:
            print(f"  Procesando señales: {idx+1}/{n_total} ({(idx+1)/n_total*100:.1f}%)")
        X_sota_i = extract_sota_features_global(sig, FS_TARGET)
        X_all_sota.append(X_sota_i)
    X_all_sota = np.array(X_all_sota)
    print(f"Features SOTA extraídos en {time.time() - t0:.1f} segundos. Forma: {X_all_sota.shape}")
    
    # 4. Fusionar matrices
    # Las señales están en el mismo orden que en el dataset.npz / step4
    X_all_combined = np.hstack([X_all_classic, X_all_sota])
    
    # Para el subconjunto etiquetado
    labels_file = _PROJECT_ROOT / "proy_labels.mat"
    import scipy.io
    mat = scipy.io.loadmat(str(labels_file), squeeze_me=True)
    labels = np.asarray(mat["labels"]).ravel().astype(int)
    mask = (labels == 2) | (labels == 3)
    
    X_labeled_combined = X_all_combined[mask]
    
    sota_names = ["sota_tonal_index", "sota_peak_entropy", "sota_kurtosis", "sota_f50_f90"]
    combined_names = classic_names + sota_names
    
    # 5. Guardar resultados
    np.save(COMBINED_DIR / "X_all_features.npy", X_all_combined)
    np.save(COMBINED_DIR / "X_labeled_features.npy", X_labeled_combined)
    np.save(COMBINED_DIR / "y_labeled.npy", y_labeled)
    np.save(COMBINED_DIR / "groups_labeled.npy", groups_labeled)
    with open(COMBINED_DIR / "feature_names.json", "w", encoding="utf-8") as fh:
        json.dump(combined_names, fh, indent=2)
        
    print("\nFusión de matrices completada exitosamente!")
    print(f"  Directorio: {COMBINED_DIR}")
    print(f"  Matriz total   X_all     : {X_all_combined.shape}")
    print(f"  Matriz labeled X_labeled : {X_labeled_combined.shape}")
    print(f"  Nombres total            : {len(combined_names)}")

if __name__ == "__main__":
    main()
