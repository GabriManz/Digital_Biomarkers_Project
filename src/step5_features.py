"""
Extracción de features de los segmentos respiratorios.

Procesa las 14 900 señales construidas en el paso 4, extrae 15 features
por segmento y guarda las matrices resultantes en disco para su uso en
step6_classification.py. Si el caché existe, lo carga directamente sin
recomputar.

Uso:
    python src/step5_features.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla para entornos sin GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import scipy.stats
import seaborn as sns

# ---------------------------------------------------------------------------
# Permite importar módulos vecinos desde src/
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from step4_dataset import (
    N_PATIENTS,
    N_CONTROLS,
    build_dataset,
)
import csv as _csv

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """
    Busca la raíz del proyecto subiendo en la jerarquía de directorios
    hasta encontrar proy_labels.mat. Necesario para compatibilidad con
    worktrees de git donde __file__ no apunta a la raíz del proyecto.
    """
    candidate = Path(__file__).resolve().parent.parent
    for _ in range(6):
        if (candidate / "proy_labels.mat").exists():
            return candidate
        candidate = candidate.parent
    # Fallback: raíz del directorio src/
    return Path(__file__).resolve().parent.parent


_PROJECT_ROOT = _find_project_root()

FS_TARGET    = 4000
N_FEATURES   = 15
LABEL_CAS    = 2
LABEL_NO_CAS = 3

LABELS_FILE  = _PROJECT_ROOT / "proy_labels.mat"
METADATA_CSV = str(_PROJECT_ROOT / "Data" / "database" / "subject_metadata.csv")
DATASET_NPZ  = _PROJECT_ROOT / "outputs" / "results" / "step4" / "dataset.npz"
CACHE_DIR    = _PROJECT_ROOT / "outputs" / "results" / "step5"
OUTPUT_FIGS = _PROJECT_ROOT / "outputs" / "figures" / "step5"

FEATURE_NAMES: list[str] = [
    "rms",
    "duration_s",
    "zcr",
    "kurtosis",
    "skewness",
    "tkeo_mean",
    "freq_dominant",
    "freq_mean",
    "band_power_100_1000",
    "band_power_70_200",
    "band_power_200_600",
    "band_power_600_1000",
    "spectral_entropy",
    "harmonic_ratio",
    "sample_entropy",
]
# ---------------------------------------------------------------------------


# ===========================================================================
# Helpers locales para carga de metadatos y lista de sujetos
# (replican la lógica de step4 con los paths correctos de este proyecto)
# ===========================================================================

def _load_metadata_local(csv_path: str) -> dict[str, str]:
    """Carga el CSV de metadatos y devuelve {subject_id: bdr_label}."""
    metadata: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            metadata[row["subject_id"]] = row["bdr_label"]
    return metadata


def _build_subject_list_local(
    metadata: dict[str, str],
    data_dir: Path,
) -> list[tuple]:
    """
    Construye la lista de los 28 sujetos usando los paths de data_dir.
    Retorna lista de (subj_num, subj_id, sig_file, mkr_file, bdr_label, type).
    """
    subjects: list[tuple] = []
    for i in range(1, N_PATIENTS + 1):
        sid = f"P{i}"
        subjects.append((
            i, sid,
            str(data_dir / f"{sid}.mat"),
            str(data_dir / f"t{sid}.mat"),
            metadata[sid], "patient",
        ))
    for i in range(1, N_CONTROLS + 1):
        sid = f"C{i}"
        subjects.append((
            N_PATIENTS + i, sid,
            str(data_dir / f"{sid}.mat"),
            str(data_dir / f"t{sid}.mat"),
            metadata[sid], "control",
        ))
    return subjects


# ===========================================================================
# Implementación de sample entropy (sin librerías externas)
# ===========================================================================

def _sampen(sig: np.ndarray, m: int, r: float) -> float:
    """
    Calcula la sample entropy de una señal discreta.

    Parámetros
    ----------
    sig : np.ndarray
        Señal 1D (ya submuestreada si es necesario).
    m : int
        Longitud del template.
    r : float
        Tolerancia (umbral de similitud).

    Retorna
    -------
    float
        Valor de sample entropy. Devuelve 0 si no hay coincidencias.
    """
    N = len(sig)

    def _count_matches(template_len: int) -> int:
        count = 0
        for i in range(N - template_len):
            template = sig[i : i + template_len]
            for j in range(i + 1, N - template_len):
                if np.max(np.abs(sig[j : j + template_len] - template)) <= r:
                    count += 1
        return count

    B = _count_matches(m)
    A = _count_matches(m + 1)
    if B == 0:
        return 0.0
    return -np.log(A / B) if A > 0 else 0.0


# ===========================================================================
# Extracción de features
# ===========================================================================

def extract_features(signal: np.ndarray, fs: int = FS_TARGET) -> np.ndarray:
    """
    Extrae exactamente 15 features de un segmento de señal respiratoria.

    Parámetros
    ----------
    signal : np.ndarray
        Segmento 1D a frecuencia de muestreo fs.
    fs : int
        Frecuencia de muestreo (Hz).

    Retorna
    -------
    np.ndarray
        Array float64 de forma (15,) con los valores de cada feature en el
        orden definido por FEATURE_NAMES. NaN e Inf se sustituyen por 0.
    """
    feats = np.zeros(N_FEATURES, dtype=np.float64)
    n = len(signal)

    # ------------------------------------------------------------------
    # 1. RMS
    # ------------------------------------------------------------------
    feats[0] = np.sqrt(np.mean(signal ** 2))

    # ------------------------------------------------------------------
    # 2. Duración en segundos
    # ------------------------------------------------------------------
    feats[1] = n / fs

    # ------------------------------------------------------------------
    # 3. Zero-crossing rate
    # ------------------------------------------------------------------
    feats[2] = np.sum(np.abs(np.diff(np.sign(signal + 1e-10)))) / (2 * n)

    # ------------------------------------------------------------------
    # 4. Curtosis (Fisher, media cero para distribución normal)
    # ------------------------------------------------------------------
    feats[3] = float(scipy.stats.kurtosis(signal, fisher=True))

    # ------------------------------------------------------------------
    # 5. Asimetría (skewness)
    # ------------------------------------------------------------------
    feats[4] = float(scipy.stats.skew(signal))

    # ------------------------------------------------------------------
    # 6. TKEO medio
    # ------------------------------------------------------------------
    if n >= 3:
        tkeo = signal[1:-1] ** 2 - signal[:-2] * signal[2:]
        feats[5] = np.mean(np.abs(tkeo))
    else:
        feats[5] = 0.0

    # ------------------------------------------------------------------
    # Welch PSD — se calcula una única vez y se reutiliza en features 7–14
    # ------------------------------------------------------------------
    nperseg = min(256, n)
    f, psd = scipy.signal.welch(signal, fs, nperseg=nperseg)
    total_power = np.sum(psd) + 1e-12   # evitar división por cero

    # ------------------------------------------------------------------
    # 7. Frecuencia dominante (restringida a 70–2000 Hz para evitar
    #    que el componente DC o derivas de baja frecuencia dominen)
    # ------------------------------------------------------------------
    mask_valid = (f >= 70) & (f <= 2000)
    f_dom = float(f[mask_valid][np.argmax(psd[mask_valid])]) if np.any(mask_valid) else 0.0
    feats[6] = f_dom

    # ------------------------------------------------------------------
    # 8. Frecuencia media (banda 70–2000 Hz)
    # ------------------------------------------------------------------
    mask_band = (f >= 70) & (f <= 2000)
    f_band  = f[mask_band]
    psd_band = psd[mask_band]
    band_sum = np.sum(psd_band) + 1e-12
    feats[7] = float(np.sum(f_band * psd_band) / band_sum)

    # ------------------------------------------------------------------
    # 9–12. Potencias de banda (fracción sobre potencia total)
    # ------------------------------------------------------------------
    feats[8]  = np.sum(psd[(f >= 100) & (f <= 1000)]) / total_power
    feats[9]  = np.sum(psd[(f >= 70)  & (f <= 200)])  / total_power
    feats[10] = np.sum(psd[(f >= 200) & (f <= 600)])  / total_power
    feats[11] = np.sum(psd[(f >= 600) & (f <= 1000)]) / total_power

    # ------------------------------------------------------------------
    # 13. Entropía espectral (normalizada a [0, 1])
    # ------------------------------------------------------------------
    psd_norm = psd / total_power
    H = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    feats[12] = H / np.log2(len(psd)) if len(psd) > 1 else 0.0

    # ------------------------------------------------------------------
    # 14. Harmonic ratio — energía en armónicos 1, 2, 3 de f_dom
    # ------------------------------------------------------------------
    if f_dom > 0:
        harmonic_energy = 0.0
        valid = True
        for k in range(1, 4):
            f_k = k * f_dom
            mask_k = (f >= f_k - 5) & (f <= f_k + 5)
            if not np.any(mask_k):
                valid = False
                break
            harmonic_energy += np.sum(psd[mask_k])
        feats[13] = (harmonic_energy / total_power) if valid else 0.0
    else:
        feats[13] = 0.0

    # ------------------------------------------------------------------
    # 15. Sample entropy (submuestro a máx. 400 puntos para velocidad)
    #     La tolerancia r se calcula sobre la señal submuestreada, no
    #     sobre la original, para calibrarla respecto a los datos reales
    #     que _sampen va a comparar.
    # ------------------------------------------------------------------
    if np.std(signal) == 0.0 or n < 50:
        feats[14] = 0.0
    else:
        step = max(1, n // 400)
        sig_sub = signal[::step]
        r_tol = 0.2 * np.std(sig_sub)
        feats[14] = _sampen(sig_sub, m=2, r=r_tol)

    # ------------------------------------------------------------------
    # Sanear NaN e Inf
    # ------------------------------------------------------------------
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


# ===========================================================================
# Construcción y guardado de las matrices de features
# ===========================================================================

def build_all_feature_matrices() -> dict:
    """
    Construye las matrices de features para los 14 900 segmentos.

    Si el caché existe en CACHE_DIR, lo carga sin recomputar.
    Si no, ejecuta el pipeline completo (step1–step4), extrae features
    y guarda los resultados en disco.

    Retorna
    -------
    dict con claves:
        X_all          — np.ndarray (14900, 15) todas las features
        X_labeled      — np.ndarray (1923, 15) solo segmentos con etiqueta 2 o 3
        y_labeled      — np.ndarray (1923,) binario: 1=CAS, 0=NO_CAS
        groups         — np.ndarray (1923,) IDs de sujeto para LOSO
        feature_names  — list[str] de 15 nombres
    """
    cache_file = CACHE_DIR / "X_all_features.npy"

    if cache_file.exists():
        print("Cache encontrado — cargando features desde disco.")
        X_all     = np.load(CACHE_DIR / "X_all_features.npy")
        X_labeled = np.load(CACHE_DIR / "X_labeled_features.npy")
        y_labeled = np.load(CACHE_DIR / "y_labeled.npy")
        groups    = np.load(CACHE_DIR / "groups_labeled.npy")
        with open(CACHE_DIR / "feature_names.json", encoding="utf-8") as fh:
            feat_names = json.load(fh)
        return {
            "X_all": X_all,
            "X_labeled": X_labeled,
            "y_labeled": y_labeled,
            "groups": groups,
            "feature_names": feat_names,
        }

    # ------------------------------------------------------------------
    # 1. Reconstruir las 14 900 señales mediante el pipeline completo
    # ------------------------------------------------------------------
    print("Reconstruyendo señales mediante el pipeline completo (step1–step4)...")
    data_dir = _PROJECT_ROOT / "Data"
    metadata = _load_metadata_local(str(METADATA_CSV))
    subjects = _build_subject_list_local(metadata, data_dir)
    all_signals, v_subject, v_bd, v_channel, v_phase, _ = build_dataset(subjects)

    # ------------------------------------------------------------------
    # 2. Cargar etiquetas del archivo del profesor
    # ------------------------------------------------------------------
    mat    = scipy.io.loadmat(str(LABELS_FILE), squeeze_me=True)
    labels = np.asarray(mat["labels"]).ravel().astype(int)

    # ------------------------------------------------------------------
    # 3. Extraer features de los 14 900 segmentos
    # ------------------------------------------------------------------
    n_total = len(all_signals)
    X_all = np.zeros((n_total, N_FEATURES), dtype=np.float64)

    print(f"\nExtrayendo {N_FEATURES} features por segmento ({n_total} segmentos total)...")
    for i, sig in enumerate(all_signals):
        if (i + 1) % 500 == 0:
            pct = 100 * (i + 1) / n_total
            print(f"  Extrayendo features: {i + 1}/{n_total} ({pct:.1f}%)")
        X_all[i] = extract_features(sig, fs=FS_TARGET)

    # ------------------------------------------------------------------
    # 4. Construir subconjunto etiquetado (etiquetas 2 y 3 del profesor)
    # ------------------------------------------------------------------
    mask      = (labels == LABEL_CAS) | (labels == LABEL_NO_CAS)
    X_labeled = X_all[mask]
    y_labeled = (labels[mask] == LABEL_CAS).astype(int)   # 1=CAS, 0=NO_CAS
    groups    = v_subject[mask]

    # ------------------------------------------------------------------
    # 5. Guardar en disco
    # ------------------------------------------------------------------
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_DIR / "X_all_features.npy",     X_all)
    np.save(CACHE_DIR / "X_labeled_features.npy", X_labeled)
    np.save(CACHE_DIR / "y_labeled.npy",           y_labeled)
    np.save(CACHE_DIR / "groups_labeled.npy",      groups)
    with open(CACHE_DIR / "feature_names.json", "w", encoding="utf-8") as fh:
        json.dump(FEATURE_NAMES, fh, ensure_ascii=False, indent=2)

    print(f"\nFeatures guardados en {CACHE_DIR}")
    print(f"  X_all_features.npy     : {X_all.shape}")
    print(f"  X_labeled_features.npy : {X_labeled.shape}")
    print(f"  y_labeled.npy          : {y_labeled.shape}  (CAS={y_labeled.sum()}, NO_CAS={(1-y_labeled).sum()})")
    print(f"  groups_labeled.npy     : {groups.shape}")

    return {
        "X_all": X_all,
        "X_labeled": X_labeled,
        "y_labeled": y_labeled,
        "groups": groups,
        "feature_names": FEATURE_NAMES,
    }


# ===========================================================================
# Figuras de análisis exploratorio
# ===========================================================================

def _generate_figures(X_labeled: np.ndarray, y_labeled: np.ndarray) -> None:
    """
    Genera las tres figuras de análisis exploratorio y las guarda en
    outputs/figures/step5/.

    Parámetros
    ----------
    X_labeled : np.ndarray
        Matriz (1923, 15) de features del subconjunto etiquetado.
    y_labeled : np.ndarray
        Vector (1923,) binario: 1=CAS, 0=NO_CAS.
    """
    OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)

    mask_cas  = y_labeled == 1
    mask_ncas = y_labeled == 0

    df = pd.DataFrame(X_labeled, columns=FEATURE_NAMES)
    df["clase"] = np.where(mask_cas, "CAS", "NO CAS")

    # ------------------------------------------------------------------
    # Figura 1 — Distribuciones KDE: CAS vs NO CAS (3×5 subplots)
    # ------------------------------------------------------------------
    print("\nGenerando figura 1: distribuciones de features...")
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes_flat = axes.flatten()

    for idx, feat in enumerate(FEATURE_NAMES):
        ax = axes_flat[idx]
        sns.kdeplot(
            data=df, x=feat, hue="clase",
            palette={"CAS": "red", "NO CAS": "blue"},
            fill=True, alpha=0.6, ax=ax,
            legend=(idx == 0),
        )
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("")

    fig.suptitle("Distribucion de features — CAS vs NO CAS", fontsize=14, y=1.01)
    plt.tight_layout()
    out1 = OUTPUT_FIGS / "fig1_feature_distributions.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {out1}")

    # ------------------------------------------------------------------
    # Figura 2 — Matriz de correlación 15×15
    # ------------------------------------------------------------------
    print("Generando figura 2: matriz de correlacion...")
    corr = pd.DataFrame(X_labeled, columns=FEATURE_NAMES).corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, annot=False, cmap="coolwarm",
        vmin=-1, vmax=1,
        xticklabels=FEATURE_NAMES, yticklabels=FEATURE_NAMES,
        ax=ax,
    )
    ax.set_title("Matriz de correlacion de features", fontsize=13)
    plt.tight_layout()
    out2 = OUTPUT_FIGS / "fig2_feature_correlation.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {out2}")

    # ------------------------------------------------------------------
    # Figura 3 — Medias CAS vs NO CAS (barras horizontales, normalizadas)
    # ------------------------------------------------------------------
    print("Generando figura 3: medias CAS vs NO CAS...")

    X_cas  = X_labeled[mask_cas]
    X_ncas = X_labeled[mask_ncas]

    # Normalización min-max por feature sobre el subconjunto completo
    feat_min = X_labeled.min(axis=0)
    feat_max = X_labeled.max(axis=0)
    feat_range = feat_max - feat_min + 1e-12

    mean_cas_norm  = (X_cas.mean(axis=0)  - feat_min) / feat_range
    mean_ncas_norm = (X_ncas.mean(axis=0) - feat_min) / feat_range

    y_pos = np.arange(N_FEATURES)
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(y_pos + bar_height / 2, mean_cas_norm,  bar_height,
            color="red",  alpha=0.75, label="CAS")
    ax.barh(y_pos - bar_height / 2, mean_ncas_norm, bar_height,
            color="blue", alpha=0.75, label="NO CAS")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=9)
    ax.set_xlabel("Media normalizada [0, 1]")
    ax.set_title("Media de cada feature — CAS vs NO CAS")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    out3 = OUTPUT_FIGS / "fig3_feature_means_cas_vs_nocas.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {out3}")


# ===========================================================================
# Punto de entrada
# ===========================================================================

if __name__ == "__main__":
    t0 = time.time()
    result = build_all_feature_matrices()
    _generate_figures(result["X_labeled"], result["y_labeled"])
    elapsed = (time.time() - t0) / 60
    print(f"\nExtraccion completada en {elapsed:.1f} minutos.")
