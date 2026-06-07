"""
Extracción de features de los segmentos respiratorios.

Procesa las 14 900 señales construidas en el paso 4, extrae features
por segmento y guarda las matrices resultantes en disco para su uso en
step6_classification.py. Si el caché existe, lo carga directamente sin
recomputar.

Grupos de features (137 total — Fase 2.1):
  - Temporales        : 16  (RMS, std, varianza, energía, skewness, kurtosis…)
  - Espectrales       : 13  (centroide, spread, rolloff, flatness, entropía + 5 bandas)
  - MFCC dinámico    : 80  (20 coefs × std_temporal + |delta|_mean + delta_std + |delta²|_mean)
  - Ratios espectrales: 9   (proporciones de banda + ratios alta/baja, invariantes a ganancia)
  - Modulación AM    : 4   (índice de modulación, frec. dominante, energía lenta/media)
  - Wavelet (db4)    : 15  (5 niveles × energía + entropía + std)

Cambio respecto a versión anterior:
  - MFCCs absolutos (120 features, media+std de main/delta/delta²) eliminados.
    Codificaban identidad del paciente (timbre), no patología.
  - Sustituidos por MFCCs dinámicos (80): sólo variación temporal intra-paciente.
  - Añadidos ratios espectrales (9) y modulación AM (4) del plan Fase 2.

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
from scipy.signal import hilbert
import scipy.stats
import seaborn as sns
import librosa
import pywt

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
N_MFCC       = 20
LABEL_CAS    = 2
LABEL_NO_CAS = 3

# --- Nuevos hiperparámetros de preprocesamiento ---
APPLY_BANDPASS_SEGMENT = False  # Si aplicar filtro bandpass al segmento
APPLY_PREEMPHASIS     = False   # Si aplicar pre-énfasis para potenciar frecuencias altas (sibilancias)
PREEMPHASIS_ALPHA      = 0.97

# Fase 2.1: MFCCs dinámicos (std + |delta| + delta_std + |delta²|) × 20 coef = 80
# Ratios espectrales: 9 | Modulación AM: 4
# Total: 16 + 13 + 80 + 9 + 4 + 15 = 137
N_FEATURES   = 137

LABELS_FILE  = _PROJECT_ROOT / "proy_labels.mat"
METADATA_CSV = str(_PROJECT_ROOT / "Data" / "database" / "subject_metadata.csv")
DATASET_NPZ  = _PROJECT_ROOT / "outputs" / "results" / "step4" / "dataset.npz"
CACHE_DIR    = _PROJECT_ROOT / "outputs" / "results" / "step5"
OUTPUT_FIGS = _PROJECT_ROOT / "outputs" / "figures" / "step5"

_MFCC_DYN_NAMES: list[str] = (
    [f"mfcc{i}_std"     for i in range(N_MFCC)] +   # variabilidad temporal (20)
    [f"dmfcc{i}_absm"   for i in range(N_MFCC)] +   # |delta| medio          (20)
    [f"dmfcc{i}_std"    for i in range(N_MFCC)] +   # delta std              (20)
    [f"d2mfcc{i}_absm"  for i in range(N_MFCC)]     # |delta²| medio         (20)
)   # 80 features

_SPECTRAL_RATIO_NAMES: list[str] = [
    "sr_prop_70_250", "sr_prop_250_500", "sr_prop_500_1000",
    "sr_prop_1000_1500", "sr_prop_1500_1900",
    "sr_hi_lo",      # (500–1900) / (70–500)
    "sr_mihi_milo",  # (1000–1500) / (250–500)
    "sr_hihi_mid",   # (1500–1900) / (500–1000)
    "sr_global",     # (1000–1900) / (70–500)
]   # 9 features

_AM_NAMES: list[str] = [
    "am_mod_idx", "am_dom_freq", "am_energy_slow", "am_energy_mid",
]   # 4 features

_WV_NAMES: list[str] = [
    f"wv_l{lvl}_{stat}"
    for lvl in range(1, 6)
    for stat in ("energy", "entropy", "std")
]
FEATURE_NAMES: list[str] = (
    # --- temporal (16) ---
    ["t_mean", "t_std", "t_var", "t_rms", "t_maxabs", "t_range",
     "t_skew", "t_kurt", "t_zcr", "t_crest",
     "t_entropy", "t_energy", "t_log_energy", "t_var2",
     "t_hig_mob", "t_hig_comp"] +
    # --- spectral (13) ---
    ["s_centroid", "s_spread", "s_rolloff85", "s_flatness", "s_entropy",
     "s_dom_freq", "s_centroid2", "s_median_freq",
     "s_bp_70_250", "s_bp_250_500", "s_bp_500_1000",
     "s_bp_1000_1500", "s_bp_1500_1900"] +
    # --- MFCC dinámico (80) ---
    _MFCC_DYN_NAMES +
    # --- ratios espectrales (9) ---
    _SPECTRAL_RATIO_NAMES +
    # --- modulación AM (4) ---
    _AM_NAMES +
    # --- wavelet (15) ---
    _WV_NAMES
)
assert len(FEATURE_NAMES) == N_FEATURES, (
    f"FEATURE_NAMES tiene {len(FEATURE_NAMES)} entradas, se esperan {N_FEATURES}"
)
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
# Normalización robusta por segmento (MAD z-score)
# ===========================================================================

def _pre_emphasis(sig: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """Aplica pre-énfasis a la señal: y[n] = x[n] - alpha * x[n-1]."""
    if len(sig) == 0:
        return sig
    return np.append(sig[0], sig[1:] - alpha * sig[:-1])


def _bandpass_segment(sig: np.ndarray, fs: int = FS_TARGET, low: float = 70.0, high: float = 1900.0) -> np.ndarray:
    """Filtro paso-banda Butterworth de fase cero sobre el segmento."""
    if len(sig) < 15:  # butterworth necesita un tamaño mínimo de señal
        return sig
    sos = scipy.signal.butter(4, [low, high], btype="band", fs=fs, output="sos")
    return scipy.signal.sosfiltfilt(sos, sig)


def _mad_normalize(sig: np.ndarray) -> np.ndarray:
    """Normalización robusta: z = (x - mediana) / (1.4826 * MAD)."""
    med = np.median(sig)
    mad = np.median(np.abs(sig - med))
    if mad < 1e-12:
        return sig - med
    return (sig - med) / (1.4826 * mad)


def _safe(seg: np.ndarray, mn: int = 2048) -> np.ndarray:
    """Rellena con ceros hasta longitud mínima y devuelve float64."""
    if len(seg) < mn:
        seg = np.pad(seg, (0, mn - len(seg)))
    return seg.astype(np.float64)


# ===========================================================================
# Grupos de features (igual que Adria/classification.py)
# ===========================================================================

def _feat_temporal(s: np.ndarray) -> list[float]:
    """16 features temporales: estadísticos, energía, complejidad de Higuchi."""
    s   = _safe(s)
    rms = np.sqrt(np.mean(s ** 2))
    d1  = np.diff(s); d2 = np.diff(d1)
    v   = np.var(s)
    hm  = np.sqrt(np.var(d1) / (v + 1e-12))
    prob = s ** 2 / (np.sum(s ** 2) + 1e-12)
    return [
        float(np.mean(s)), float(np.std(s)), float(v), float(rms),
        float(np.max(np.abs(s))), float(np.max(s) - np.min(s)),
        float(pd.Series(s).skew()), float(pd.Series(s).kurt()),
        float(np.sum(np.abs(np.diff(np.sign(s))) > 0) / len(s)),
        float(np.max(np.abs(s)) / (rms + 1e-12)),
        float(-np.sum(prob * np.log(prob + 1e-12))),
        float(np.sum(s ** 2)), float(np.log(np.sum(s ** 2) + 1e-12)),
        float(v), float(hm),
        float(np.sqrt(np.var(d2) / (np.var(d1) + 1e-12)) / (hm + 1e-12)),
    ]


def _feat_spectral(s: np.ndarray, fs: int = FS_TARGET) -> list[float]:
    """13 features espectrales: centroide, spread, rolloff, bandas de potencia."""
    s  = _safe(s)
    f, p = scipy.signal.welch(s, fs=fs, nperseg=min(512, len(s)))
    tp = float(np.sum(p)) + 1e-12
    pn = p / tp
    sc = float(np.sum(f * pn))
    feats = [
        sc,
        float(np.sqrt(np.sum(((f - sc) ** 2) * pn))),
        float(f[np.searchsorted(np.cumsum(pn), 0.85)]),
        float(np.exp(np.mean(np.log(p + 1e-12))) / (np.mean(p) + 1e-12)),
        float(-np.sum(pn * np.log(pn + 1e-12))),
        float(f[np.argmax(p)]),
        sc,
        float(f[np.searchsorted(np.cumsum(pn), 0.50)]),
    ]
    for lo, hi in [(70, 250), (250, 500), (500, 1000), (1000, 1500), (1500, 1900)]:
        feats.append(float(np.sum(p[(f >= lo) & (f < hi)]) / tp))
    return feats   # 13


def _feat_mfcc_dynamic(s: np.ndarray, fs: int = FS_TARGET) -> list[float]:
    """
    80 features MFCC dinámicas: invariantes a la identidad del paciente.

    En lugar de la media de cada MFCC (que codifica el timbre absoluto del
    tracto vocal — identidad del paciente), se extraen:
      - std temporal     (20): variabilidad intra-segmento de cada coef.
      - |delta| medio    (20): magnitud media de la velocidad espectral.
      - delta std        (20): variabilidad de esa velocidad.
      - |delta²| medio  (20): magnitud media de la aceleración espectral.

    Estas medidas capturan la DINÁMICA temporal del sonido respiratorio,
    que difiere entre CAS (modulaciones periódicas) y NO-CAS, sin depender
    del nivel absoluto del MFCC (que es constante por paciente).
    """
    s  = _safe(s, 2048).astype(np.float32)
    m  = librosa.feature.mfcc(y=s, sr=fs, n_mfcc=N_MFCC)
    nf = m.shape[1]
    w  = min(9, nf if nf % 2 == 1 else max(nf - 1, 1))
    w  = max(w, 3)
    mo = "interp" if nf >= w else "nearest"
    d  = librosa.feature.delta(m, width=w, mode=mo)
    d2 = librosa.feature.delta(m, width=w, mode=mo, order=2)
    return (
        list(np.std(m,  axis=1).astype(float))          +   # std temporal  (20)
        list(np.mean(np.abs(d),  axis=1).astype(float)) +   # |delta| medio (20)
        list(np.std(d,  axis=1).astype(float))          +   # delta std     (20)
        list(np.mean(np.abs(d2), axis=1).astype(float))     # |d2| medio    (20)
    )   # 80


def _feat_spectral_ratios(s: np.ndarray, fs: int = FS_TARGET) -> list[float]:
    """
    9 features de ratios espectrales invariantes a ganancia.

    Las potencias absolutas de banda dependen del volumen de respiración
    y la distancia al micrófono (identidad del contexto, no patología).
    Los ratios entre bandas son invariantes a esa ganancia global.

    CAS (sibilancias, roncus) concentra energía en bandas altas (>500 Hz)
    mientras que la respiración normal tiene más energía en frecuencias bajas.
    """
    s = _safe(s)
    f, p = scipy.signal.welch(s, fs=fs, nperseg=min(512, len(s)))
    bp = lambda lo, hi: float(np.sum(p[(f >= lo) & (f < hi)])) + 1e-12

    b1 = bp(70,   250)    # baja
    b2 = bp(250,  500)    # media-baja
    b3 = bp(500,  1000)   # media
    b4 = bp(1000, 1500)   # media-alta
    b5 = bp(1500, 1900)   # alta
    total = b1 + b2 + b3 + b4 + b5

    return [
        b1/total, b2/total, b3/total, b4/total, b5/total,   # proporciones (5)
        (b3+b4+b5) / (b1+b2),     # ratio global alta/baja
        b4 / b2,                   # media-alta / media-baja
        b5 / b3,                   # alta / media
        (b4+b5) / (b1+b2),         # similar al global, filtro ms agresivo
    ]   # 9


def _feat_amplitude_modulation(s: np.ndarray, fs: int = FS_TARGET) -> list[float]:
    """
    4 features de modulación de amplitud (AM) basadas en la envolvente Hilbert.

    CAS como sibilancias tienen modulaciones AM periódicas en la envolvente
    (rango 5–100 Hz). La respiración normal tiene envolvente más uniforme.
    Estas features capturan la estructura temporal de la amplitud sin depender
    del nivel absoluto.
    """
    s = _safe(s)
    envelope = np.abs(hilbert(s))
    mean_env = float(np.mean(envelope)) + 1e-12

    # Espectro de la envolvente (modulación AM)
    f_env, p_env = scipy.signal.welch(
        envelope, fs=fs, nperseg=min(256, len(envelope))
    )
    total_env = float(np.sum(p_env)) + 1e-12

    return [
        float(np.std(envelope)) / mean_env,                           # índice de modulación
        float(f_env[np.argmax(p_env)]),                                # frec. dominante AM
        float(np.sum(p_env[f_env <= 20])) / total_env,                # energía AM lenta (<20 Hz)
        float(np.sum(p_env[(f_env > 20) & (f_env <= 100)])) / total_env,  # AM media (20-100 Hz)
    ]   # 4


def _feat_wavelet(s: np.ndarray) -> list[float]:
    """15 features wavelet: db4 nivel 5, energía + entropía + std por nivel."""
    s      = _safe(s)
    coeffs = pywt.wavedec(s, "db4", level=5)
    out: list[float] = []
    for c in coeffs[1:]:
        e    = float(np.sum(c ** 2))
        prob = c ** 2 / (e + 1e-12)
        out.extend([e, float(-np.sum(prob * np.log(prob + 1e-12))), float(np.std(c))])
    return out   # 15


# ===========================================================================
# Extracción de features (función pública)
# ===========================================================================

def extract_features(signal: np.ndarray, fs: int = FS_TARGET) -> np.ndarray:
    """
    Extrae 137 features de un segmento de señal respiratoria.

    Grupos:
      - 16 features temporales
      - 13 features espectrales (potencias de banda absolutas)
      - 80 MFCC dinámicos (std + |delta| + delta_std + |delta²|)
      - 9 ratios espectrales (invariantes a ganancia)
      - 4 modulación AM (envolvente Hilbert)
      - 15 wavelet (db4, 5 niveles)

    Aplica preprocesamiento (bandpass y pre-énfasis opcionales) y normalización MAD robusta antes de la extracción.
    NaN e Inf se sustituyen por 0.
    """
    sig = np.asarray(signal, dtype=np.float64)

    if APPLY_BANDPASS_SEGMENT:
        sig = _bandpass_segment(sig, fs=fs)

    if APPLY_PREEMPHASIS:
        sig = _pre_emphasis(sig, alpha=PREEMPHASIS_ALPHA)

    sig = _mad_normalize(sig)

    feats = (
        _feat_temporal(sig) +               # 16
        _feat_spectral(sig, fs) +           # 13
        _feat_mfcc_dynamic(sig, fs) +       # 80 (dinámicos, Fase 2.1)
        _feat_spectral_ratios(sig, fs) +    # 9  (invariantes a ganancia, Fase 2.2)
        _feat_amplitude_modulation(sig, fs) +  # 4 (modulación AM, Fase 2.3)
        _feat_wavelet(sig)                  # 15
    )                                       # = 137

    arr = np.array(feats, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


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
        # Verificar que el caché tiene el número correcto de features
        X_test = np.load(cache_file, mmap_mode="r")
        n_cached_features = X_test.shape[1]
        del X_test  # Liberar el mmap antes de posible sobreescritura (Windows errno 22)
        if n_cached_features == N_FEATURES:
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
        else:
            print(f"Cache obsoleto ({n_cached_features} features vs {N_FEATURES} esperadas) — regenerando.")

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
    print(f"  feature_names.json     : {len(FEATURE_NAMES)} features")

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
    # Figura 1 — Distribuciones KDE: CAS vs NO CAS (4×5 subplots, primeras 20)
    # ------------------------------------------------------------------
    print("\nGenerando figura 1: distribuciones de features (primeras 20)...")
    feat_show = FEATURE_NAMES[:20]
    fig, axes = plt.subplots(4, 5, figsize=(20, 14))
    axes_flat = axes.flatten()

    for idx, feat in enumerate(feat_show):
        ax = axes_flat[idx]
        sns.kdeplot(
            data=df, x=feat, hue="clase",
            palette={"CAS": "red", "NO CAS": "blue"},
            fill=True, alpha=0.6, ax=ax,
            legend=(idx == 0),
        )
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("")

    for ax in axes_flat[len(feat_show):]:
        ax.set_visible(False)

    fig.suptitle("Distribucion de features (primeras 20 de 164) — CAS vs NO CAS",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out1 = OUTPUT_FIGS / "fig1_feature_distributions.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardada: {out1}")

    # ------------------------------------------------------------------
    # Figura 2 — Matriz de correlación (primeras 30 features)
    # ------------------------------------------------------------------
    print("Generando figura 2: matriz de correlacion (primeras 30 features)...")
    feat_corr = FEATURE_NAMES[:30]
    corr = pd.DataFrame(X_labeled[:, :30], columns=feat_corr).corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr, annot=False, cmap="coolwarm",
        vmin=-1, vmax=1,
        xticklabels=feat_corr, yticklabels=feat_corr,
        ax=ax,
    )
    ax.set_title("Matriz de correlacion (primeras 30 de 164 features)", fontsize=12)
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

    # Seleccionar las 20 características con mayor diferencia de medias para evitar solapamientos
    abs_diff = np.abs(mean_cas_norm - mean_ncas_norm)
    top_20_idx = np.argsort(abs_diff)[-20:] # Las 20 mejores
    
    mean_cas_top = mean_cas_norm[top_20_idx]
    mean_ncas_top = mean_ncas_norm[top_20_idx]
    names_top = [FEATURE_NAMES[idx] for idx in top_20_idx]

    y_pos = np.arange(len(top_20_idx))
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y_pos + bar_height / 2, mean_cas_top,  bar_height,
            color="red",  alpha=0.75, label="CAS")
    ax.barh(y_pos - bar_height / 2, mean_ncas_top, bar_height,
            color="blue", alpha=0.75, label="NO CAS")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_top, fontsize=9.5)
    ax.set_xlabel("Media normalizada [0, 1]")
    ax.set_title("Media de las 20 features más discriminativas — CAS vs NO CAS")
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
