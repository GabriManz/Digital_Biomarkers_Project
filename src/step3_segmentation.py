"""
Segmentación de ciclos respiratorios a partir de marcadores temporales.

Carga los ficheros de marcadores tPX.mat / tCX.mat y extrae, para cada
maniobra, los segmentos individuales de inspiración y espiración de la
señal preprocesada.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

# Permite importar módulos vecinos desde el mismo directorio src/
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from step1_read_signals import read_signals
from step2_preprocessing import preprocess_signal

# ---------------------------------------------------------------------------
# Parámetros configurables
# ---------------------------------------------------------------------------
FS_TARGET = 4000   # Hz — frecuencia de muestreo de la señal preprocesada

# Rutas de ejemplo
FILE_BDR_POS    = r"C:\DATA\01_Proyectos\Master\Digital_Biomarkers\Project\Data\P2.mat"
FILE_BDR_NEG    = r"C:\DATA\01_Proyectos\Master\Digital_Biomarkers\Project\Data\P16.mat"
MARKERS_BDR_POS = r"C:\DATA\01_Proyectos\Master\Digital_Biomarkers\Project\Data\tP2.mat"
MARKERS_BDR_NEG = r"C:\DATA\01_Proyectos\Master\Digital_Biomarkers\Project\Data\tP16.mat"

_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
OUTPUT_DIR    = os.path.join(_PROJECT_ROOT, "outputs", "figures", "step3")
# ---------------------------------------------------------------------------


# ===========================================================================
# API pública — importable por step4_dataset.py
# ===========================================================================

def load_markers(pth: str) -> list[np.ndarray]:
    """
    Carga un fichero de marcadores temporales tPX.mat o tCX.mat.

    Parámetros
    ----------
    pth : str
        Ruta absoluta al fichero .mat.

    Retorna
    -------
    list[np.ndarray]
        Lista de 6 arrays, uno por maniobra. Cada array tiene forma
        (n_ciclos, 4) con columnas:
          0 → t_inicio_inspiracion (s)
          1 → t_fin_inspiracion    (s)
          2 → t_inicio_espiracion  (s)
          3 → t_fin_espiracion     (s)

    Lanza
    -----
    FileNotFoundError
        Si el fichero no existe.
    """
    if not os.path.isfile(pth):
        raise FileNotFoundError(
            f"No se encontró el fichero de marcadores: {pth}"
        )

    mat = scipy.io.loadmat(pth, squeeze_me=False)
    seg_t = mat["seg_t"]   # array de objetos de forma (1, 6)

    markers: list[np.ndarray] = []
    for j in range(seg_t.shape[1]):
        elem = seg_t[0, j]
        # loadmat puede envolver el array en capas adicionales de objetos
        while isinstance(elem, np.ndarray) and elem.dtype == object and elem.size == 1:
            elem = elem.flat[0]
        markers.append(elem.astype(np.float64))   # (n_ciclos, 4)

    return markers


def segment_signal(
    signal: np.ndarray,
    markers: np.ndarray,
    fs: int = FS_TARGET,
) -> dict:
    """
    Segmenta una señal preprocesada en ciclos individuales de inspiración
    y espiración usando los marcadores temporales de una maniobra.

    Parámetros
    ----------
    signal : np.ndarray
        Señal preprocesada 1D a frecuencia de muestreo fs.
    markers : np.ndarray
        Array de forma (n_ciclos, 4) con tiempos en segundos:
          col 0 → t_inicio_inspiracion
          col 1 → t_fin_inspiracion
          col 2 → t_inicio_espiracion
          col 3 → t_fin_espiracion
    fs : int
        Frecuencia de muestreo de la señal (Hz).

    Retorna
    -------
    dict con claves:
        "inspiracion" : list[np.ndarray] — n_ciclos segmentos de inspiración
        "espiracion"  : list[np.ndarray] — n_ciclos segmentos de espiración
    """
    n = len(signal)

    def _idx(t: float) -> int:
        """Convierte tiempo en segundos a índice de muestra con pinzado."""
        return max(0, min(int(round(t * fs)), n - 1))

    inspiraciones: list[np.ndarray] = []
    espiraciones: list[np.ndarray] = []

    for row in markers:
        t_i_ini, t_i_fin, t_e_ini, t_e_fin = row
        inspiraciones.append(signal[_idx(t_i_ini) : _idx(t_i_fin) + 1])
        espiraciones.append(signal[_idx(t_e_ini)  : _idx(t_e_fin) + 1])

    return {"inspiracion": inspiraciones, "espiracion": espiraciones}


# ===========================================================================
# Auxiliares de figura
# ===========================================================================

def _shade_manoeuvre(
    ax: plt.Axes,
    signal: np.ndarray,
    markers: np.ndarray,
    fs: int = FS_TARGET,
) -> None:
    """
    Dibuja la señal con sombreado verde (inspiración) y naranja (espiración)
    para cada ciclo, más líneas de frontera y etiquetas en el primer ciclo.
    """
    t_axis = np.arange(len(signal)) / fs
    ax.plot(t_axis, signal, linewidth=0.5, color="steelblue", zorder=2)

    y_label = float(np.max(np.abs(signal))) * 0.85 * np.sign(np.max(signal))

    for k, row in enumerate(markers):
        t_i_ini, t_i_fin, t_e_ini, t_e_fin = row
        primer_ciclo = k == 0

        ax.axvspan(t_i_ini, t_i_fin, alpha=0.3, color="green",
                   label="Inspiración" if primer_ciclo else None, zorder=1)
        ax.axvspan(t_e_ini, t_e_fin, alpha=0.3, color="orange",
                   label="Espiración" if primer_ciclo else None, zorder=1)

        # Líneas de frontera de ciclo
        for t_b in (t_i_ini, t_e_fin):
            ax.axvline(t_b, color="gray", linestyle="--", linewidth=0.5,
                       alpha=0.5, zorder=3)

        if primer_ciclo:
            ax.text((t_i_ini + t_i_fin) / 2, y_label, "Insp",
                    ha="center", va="center", fontsize=9,
                    color="darkgreen", fontweight="bold")
            ax.text((t_e_ini + t_e_fin) / 2, y_label, "Esp",
                    ha="center", va="center", fontsize=9,
                    color="darkorange", fontweight="bold")

    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud (V)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)


def _guardar(fig: plt.Figure, out_dir: str, filename: str) -> None:
    """Guarda la figura y libera memoria."""
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {path}")


# ===========================================================================
# Figuras
# ===========================================================================

def _fig_maniobra_segmentada(
    signal: np.ndarray,
    markers: np.ndarray,
    sujeto: str,
    maniobra_idx: int,
    filename: str,
    out_dir: str,
) -> None:
    """Figura 1/2: señal completa de una maniobra con ciclos sombreados."""
    fig, ax = plt.subplots(figsize=(14, 6))
    _shade_manoeuvre(ax, signal, markers)
    ax.set_title(
        f"Segmentación — {sujeto}, Canal inferior, "
        f"Maniobra {maniobra_idx + 1} (Pre-BD)",
        fontsize=11,
    )
    fig.tight_layout()
    _guardar(fig, out_dir, filename)


def _fig_segmentos_individuales(
    segs: dict,
    sujeto: str,
    filename: str,
    out_dir: str,
) -> None:
    """
    Figura 3/4: rejilla 2×4 con los primeros 4 segmentos de inspiración
    (fila superior, verde) y espiración (fila inferior, naranja).
    """
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    insp_list = segs["inspiracion"][:4]
    esp_list  = segs["espiracion"][:4]

    for col, seg in enumerate(insp_list):
        ax = axes[0, col]
        t_ms = np.arange(len(seg)) / FS_TARGET * 1000.0
        dur_ms = len(seg) / FS_TARGET * 1000.0
        ax.plot(t_ms, seg, color="green", linewidth=0.8)
        ax.set_title(f"Insp {col + 1} ({dur_ms:.0f} ms)", fontsize=9)
        ax.set_xlabel("Tiempo (ms)")
        ax.set_ylabel("Amplitud (V)")
        ax.grid(True, alpha=0.3)

    for col, seg in enumerate(esp_list):
        ax = axes[1, col]
        t_ms = np.arange(len(seg)) / FS_TARGET * 1000.0
        dur_ms = len(seg) / FS_TARGET * 1000.0
        ax.plot(t_ms, seg, color="orange", linewidth=0.8)
        ax.set_title(f"Esp {col + 1} ({dur_ms:.0f} ms)", fontsize=9)
        ax.set_xlabel("Tiempo (ms)")
        ax.set_ylabel("Amplitud (V)")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Segmentos individuales — {sujeto}, Maniobra 1", fontsize=13)
    fig.tight_layout()
    _guardar(fig, out_dir, filename)


def _fig_prebd_vs_postbd(
    signals_all: list[np.ndarray],
    markers_all: list[np.ndarray],
    sujeto: str,
    filename: str,
    out_dir: str,
) -> None:
    """
    Figura 5/6: comparativa de maniobra 2 (pre-BD) frente a maniobra 5
    (post-BD) para un mismo sujeto.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    configs = [
        (axes[0], 1, "Maniobra 2 (Pre-BD)"),
        (axes[1], 4, "Maniobra 5 (Post-BD)"),
    ]
    for ax, idx, subtitulo in configs:
        _shade_manoeuvre(ax, signals_all[idx], markers_all[idx])
        ax.set_title(subtitulo, fontsize=10)

    fig.suptitle(f"Comparativa Pre-BD vs Post-BD — {sujeto}", fontsize=13)
    fig.tight_layout()
    _guardar(fig, out_dir, filename)


# ===========================================================================
# Resumen de consola
# ===========================================================================

def _imprimir_resumen(
    sujeto: str,
    segs_all: list[dict],
) -> None:
    """Imprime una tabla de resumen con ciclos y duraciones medias por maniobra."""
    print(f"\nResumen — {sujeto}")
    cabecera = (
        f"  {'Maniobra':>9} | {'Ciclos Insp':>11} | {'Ciclos Esp':>10} | "
        f"{'Dur. Insp media (ms)':>21} | {'Dur. Esp media (ms)':>20}"
    )
    print(cabecera)
    print("  " + "-" * (len(cabecera) - 2))
    for j, segs in enumerate(segs_all):
        insp = segs["inspiracion"]
        esp  = segs["espiracion"]
        dur_i = np.mean([len(s) / FS_TARGET * 1000.0 for s in insp]) if insp else 0.0
        dur_e = np.mean([len(s) / FS_TARGET * 1000.0 for s in esp])  if esp  else 0.0
        print(
            f"  {j + 1:>9} | {len(insp):>11} | {len(esp):>10} | "
            f"{dur_i:>21.1f} | {dur_e:>20.1f}"
        )


# ===========================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Cargando señales...")
    sdata_pos = read_signals(FILE_BDR_POS)
    sdata_neg = read_signals(FILE_BDR_NEG)

    print("Cargando marcadores...")
    markers_pos = load_markers(MARKERS_BDR_POS)
    markers_neg = load_markers(MARKERS_BDR_NEG)

    print("Preprocesando señales (6 maniobras × 2 sujetos)...")
    signals_pos = [
        preprocess_signal(sdata_pos["signals"][0][j].astype(np.float64))
        for j in range(6)
    ]
    signals_neg = [
        preprocess_signal(sdata_neg["signals"][0][j].astype(np.float64))
        for j in range(6)
    ]

    print("Segmentando ciclos...")
    segs_pos = [segment_signal(signals_pos[j], markers_pos[j]) for j in range(6)]
    segs_neg = [segment_signal(signals_neg[j], markers_neg[j]) for j in range(6)]

    _imprimir_resumen("BDR+ (P2)",  segs_pos)
    _imprimir_resumen("BDR- (P16)", segs_neg)

    print("\nGenerando figuras...")

    _fig_maniobra_segmentada(
        signals_pos[0], markers_pos[0],
        sujeto="BDR+ (P2)", maniobra_idx=0,
        filename="fig1_segmentacion_maniobra_BDRpos.png",
        out_dir=OUTPUT_DIR,
    )
    _fig_maniobra_segmentada(
        signals_neg[0], markers_neg[0],
        sujeto="BDR- (P16)", maniobra_idx=0,
        filename="fig2_segmentacion_maniobra_BDRneg.png",
        out_dir=OUTPUT_DIR,
    )
    _fig_segmentos_individuales(
        segs_pos[0], sujeto="BDR+ (P2)",
        filename="fig3_segmentos_individuales_BDRpos.png",
        out_dir=OUTPUT_DIR,
    )
    _fig_segmentos_individuales(
        segs_neg[0], sujeto="BDR- (P16)",
        filename="fig4_segmentos_individuales_BDRneg.png",
        out_dir=OUTPUT_DIR,
    )
    _fig_prebd_vs_postbd(
        signals_pos, markers_pos,
        sujeto="BDR+ (P2)",
        filename="fig5_prebd_vs_postbd_BDRpos.png",
        out_dir=OUTPUT_DIR,
    )
    _fig_prebd_vs_postbd(
        signals_neg, markers_neg,
        sujeto="BDR- (P16)",
        filename="fig6_prebd_vs_postbd_BDRneg.png",
        out_dir=OUTPUT_DIR,
    )

    print("\nSegmentación completada.")
