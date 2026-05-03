"""
Preprocesado de señales de sonido respiratorio.

Implementa la cadena de preprocesado completa en tres pasos:
  1. Remuestreo de FS_ORIGINAL Hz a FS_TARGET Hz mediante resample_poly
  2. Filtro paso-banda Butterworth de fase cero (BP_LOW–BP_HIGH Hz, orden BP_ORDER)
  3. Filtro notch en comb sobre NOTCH_FUND Hz y sus armónicos hasta Nyquist,
     con ancho de banda fijo de NOTCH_BW Hz por componente
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

# Permite importar step1_read_signals desde el mismo directorio src/
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from step1_read_signals import read_signals

# ---------------------------------------------------------------------------
# Parámetros configurables
# ---------------------------------------------------------------------------
FS_ORIGINAL = 12500       # Hz — frecuencia de muestreo original
FS_TARGET   = 4000        # Hz — frecuencia de muestreo objetivo
BP_LOW      = 70.0        # Hz — límite inferior del paso banda
BP_HIGH     = 1900.0      # Hz — límite superior del paso banda
BP_ORDER    = 8           # Orden del filtro Butterworth
NOTCH_FUND  = 50.0        # Hz — frecuencia fundamental del comb notch
NOTCH_BW    = 1.0         # Hz — ancho de banda de cada notch (Q = f0 / BW)

DISPLAY_SECS = 5.0        # Segundos a visualizar en las figuras de tiempo

# Rutas de ejemplo — modificar si es necesario
FILE_BDR_POS = r"C:\DATA\01_Proyectos\Master\Digital_Biomarkers\Project\Data\P1.mat"
FILE_BDR_NEG = r"C:\DATA\01_Proyectos\Master\Digital_Biomarkers\Project\Data\P19.mat"

_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
OUTPUT_DIR    = os.path.join(_PROJECT_ROOT, "outputs", "figures", "step2")
# ---------------------------------------------------------------------------


# ===========================================================================
# Lógica de preprocesado
# ===========================================================================

def _preprocess_steps(
    signal: np.ndarray,
    fs_in: int = FS_ORIGINAL,
    fs_out: int = FS_TARGET,
) -> list[tuple[np.ndarray, int, str]]:
    """
    Aplica la cadena de preprocesado y devuelve la señal tras cada etapa.

    Parámetros
    ----------
    signal : np.ndarray
        Señal cruda de entrada en float64.
    fs_in : int
        Frecuencia de muestreo de entrada (Hz).
    fs_out : int
        Frecuencia de muestreo objetivo (Hz).

    Retorna
    -------
    list[tuple[np.ndarray, int, str]]
        Lista de (señal, fs, etiqueta) para cada estado del preprocesado.
    """
    steps: list[tuple[np.ndarray, int, str]] = [
        (signal.copy(), fs_in, f"Señal original ({fs_in} Hz)")
    ]

    # Paso 1: Remuestreo mediante razón de enteros irreducible
    g = math.gcd(fs_out, fs_in)
    up, down = fs_out // g, fs_in // g        # up=8, down=25 para 4000/12500
    s = scipy.signal.resample_poly(signal, up, down).astype(np.float64)
    steps.append((s, fs_out, f"Tras remuestreo a {fs_out} Hz"))

    # Paso 2: Filtro paso-banda Butterworth de fase cero (sosfiltfilt)
    sos_bp = scipy.signal.butter(
        BP_ORDER, [BP_LOW, BP_HIGH], btype="band", fs=fs_out, output="sos"
    )
    s = scipy.signal.sosfiltfilt(sos_bp, s)
    steps.append((s, fs_out,
                  f"Tras filtro Butterworth paso banda ({BP_LOW:.0f}–{BP_HIGH:.0f} Hz)"))

    # Paso 3: Comb notch — aplicar secuencialmente cada armónico
    f0 = NOTCH_FUND
    while f0 < fs_out / 2:
        Q = f0 / NOTCH_BW          # BW constante de 1 Hz → Q proporcional a f0
        b, a = scipy.signal.iirnotch(f0, Q, fs=fs_out)
        sos_n = scipy.signal.tf2sos(b, a)
        s = scipy.signal.sosfiltfilt(sos_n, s)
        f0 += NOTCH_FUND
    steps.append((s, fs_out, "Tras filtro notch comb (50 Hz y armónicos)"))

    return steps


def preprocess_signal(
    signal: np.ndarray,
    fs_in: int = FS_ORIGINAL,
    fs_out: int = FS_TARGET,
) -> np.ndarray:
    """
    Aplica la cadena completa de preprocesado a una señal de sonido respiratorio.

    Parámetros
    ----------
    signal : np.ndarray
        Señal cruda de entrada con frecuencia de muestreo fs_in.
    fs_in : int
        Frecuencia de muestreo de la señal de entrada (Hz).
    fs_out : int
        Frecuencia de muestreo objetivo tras el remuestreo (Hz).

    Retorna
    -------
    np.ndarray
        Señal preprocesada a fs_out Hz.
    """
    return _preprocess_steps(signal, fs_in, fs_out)[-1][0]


# ===========================================================================
# Funciones auxiliares para figuras
# ===========================================================================

def _primeros_n_secs(
    signal: np.ndarray, fs: float, secs: float = DISPLAY_SECS
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve los primeros `secs` segundos de la señal y su eje temporal."""
    n = min(int(secs * fs), len(signal))
    return np.arange(n) / fs, signal[:n]


def _psd_db(
    signal: np.ndarray, fs: float, nperseg: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula la PSD en dB mediante el método de Welch."""
    f, pxx = scipy.signal.welch(signal, fs=fs, nperseg=min(len(signal), nperseg))
    return f, 10.0 * np.log10(np.maximum(pxx, 1e-12))


def _espectrograma_db(
    signal: np.ndarray, fs: float, nperseg: int = 256, noverlap: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula el espectrograma en dB."""
    f, t, Sxx = scipy.signal.spectrogram(
        signal, fs=fs, nperseg=nperseg, noverlap=noverlap
    )
    return f, t, 10.0 * np.log10(np.maximum(Sxx, 1e-12))


# ===========================================================================
# Generación de figuras
# ===========================================================================

def _fig1_cruda_vs_preprocesada(
    raw_pos: np.ndarray, proc_pos: np.ndarray,
    raw_neg: np.ndarray, proc_neg: np.ndarray,
    out_dir: str,
) -> None:
    """Figura 1: señal cruda vs preprocesada para BDR+ y BDR- (2×2)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    configs = [
        (axes[0, 0], raw_pos,  FS_ORIGINAL, "BDR+ (P1)  —  Señal cruda"),
        (axes[0, 1], proc_pos, FS_TARGET,   "BDR+ (P1)  —  Señal preprocesada"),
        (axes[1, 0], raw_neg,  FS_ORIGINAL, "BDR- (P19) —  Señal cruda"),
        (axes[1, 1], proc_neg, FS_TARGET,   "BDR- (P19) —  Señal preprocesada"),
    ]
    for ax, sig, fs, title in configs:
        t, s = _primeros_n_secs(sig, fs)
        ax.plot(t, s, linewidth=0.5, color="steelblue")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Amplitud (V)")
        ax.set_xlim(0, DISPLAY_SECS)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Señal cruda vs preprocesada — primeros {DISPLAY_SECS:.0f} s", fontsize=13)
    fig.tight_layout()
    _guardar(fig, out_dir, "fig1_senal_cruda_vs_preprocesada.png")


def _fig_pasos(
    steps: list[tuple[np.ndarray, int, str]],
    sujeto: str,
    filename: str,
    out_dir: str,
) -> None:
    """Figura 2/3: señal tras cada etapa del preprocesado, apilada verticalmente."""
    n = len(steps)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(8, 2 * n)), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (sig, fs, etiqueta) in zip(axes, steps):
        t, s = _primeros_n_secs(sig, fs)
        ax.plot(t, s, linewidth=0.5, color="steelblue")
        ax.set_title(etiqueta, fontsize=10)
        ax.set_ylabel("Amplitud (V)")
        ax.set_xlim(0, DISPLAY_SECS)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Tiempo (s)")

    fig.suptitle(
        f"Efecto de cada filtro — {sujeto} (primeros {DISPLAY_SECS:.0f} s)", fontsize=13
    )
    fig.tight_layout()
    _guardar(fig, out_dir, filename)


def _fig4_psd(
    raw_pos: np.ndarray, proc_pos: np.ndarray,
    raw_neg: np.ndarray, proc_neg: np.ndarray,
    out_dir: str,
) -> None:
    """Figura 4: PSD antes y después del preprocesado para BDR+ y BDR- (2×2)."""
    F_MAX = 2000.0
    notch_freqs = np.arange(NOTCH_FUND, FS_TARGET / 2, NOTCH_FUND)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    configs = [
        (axes[0, 0], raw_pos,  FS_ORIGINAL, "BDR+ (P1)  —  PSD señal cruda"),
        (axes[0, 1], proc_pos, FS_TARGET,   "BDR+ (P1)  —  PSD señal preprocesada"),
        (axes[1, 0], raw_neg,  FS_ORIGINAL, "BDR- (P19) —  PSD señal cruda"),
        (axes[1, 1], proc_neg, FS_TARGET,   "BDR- (P19) —  PSD señal preprocesada"),
    ]
    for ax, sig, fs, title in configs:
        f, pdb = _psd_db(sig, fs)
        mask = f <= F_MAX
        ax.plot(f[mask], pdb[mask], linewidth=0.8, color="steelblue")

        # Límites del paso banda
        ax.axvline(BP_LOW,  color="green", linestyle="--", linewidth=1.0,
                   label=f"Paso banda ({BP_LOW:.0f}–{BP_HIGH:.0f} Hz)")
        ax.axvline(BP_HIGH, color="green", linestyle="--", linewidth=1.0)

        # Armónicos del comb notch
        for i, fn in enumerate(notch_freqs):
            lbl = "Armónicos notch (50 Hz)" if i == 0 else None
            ax.axvline(fn, color="tomato", linestyle=":", linewidth=0.6,
                       alpha=0.7, label=lbl)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Potencia (dB)")
        ax.set_xlim(0, F_MAX)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Densidad espectral de potencia — antes y después del preprocesado", fontsize=13)
    fig.tight_layout()
    _guardar(fig, out_dir, "fig4_psd_antes_despues.png")


def _fig5_espectrograma(
    raw_pos: np.ndarray, proc_pos: np.ndarray,
    raw_neg: np.ndarray, proc_neg: np.ndarray,
    out_dir: str,
) -> None:
    """Figura 5: espectrograma antes y después del preprocesado para BDR+ y BDR- (2×2)."""
    F_MAX = 2000.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    configs = [
        (axes[0, 0], raw_pos,  FS_ORIGINAL, "BDR+ (P1)  —  Espectrograma crudo"),
        (axes[0, 1], proc_pos, FS_TARGET,   "BDR+ (P1)  —  Espectrograma preprocesado"),
        (axes[1, 0], raw_neg,  FS_ORIGINAL, "BDR- (P19) —  Espectrograma crudo"),
        (axes[1, 1], proc_neg, FS_TARGET,   "BDR- (P19) —  Espectrograma preprocesado"),
    ]
    for ax, sig, fs, title in configs:
        f, t, Sxx_db = _espectrograma_db(sig, fs)
        mask = f <= F_MAX
        im = ax.pcolormesh(
            t, f[mask], Sxx_db[mask, :], cmap="inferno", shading="gouraud"
        )
        fig.colorbar(im, ax=ax, label="dB")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Frecuencia (Hz)")
        ax.set_ylim(0, F_MAX)

    fig.suptitle("Espectrograma — antes y después del preprocesado", fontsize=13)
    fig.tight_layout()
    _guardar(fig, out_dir, "fig5_espectrograma_antes_despues.png")


def _guardar(fig: plt.Figure, out_dir: str, filename: str) -> None:
    """Guarda una figura en disco y cierra el objeto para liberar memoria."""
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {path}")


# ===========================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Cargando señales...")
    sdata_pos = read_signals(FILE_BDR_POS)
    sdata_neg = read_signals(FILE_BDR_NEG)

    raw_pos: np.ndarray = sdata_pos["signals"][0][0].astype(np.float64)
    raw_neg: np.ndarray = sdata_neg["signals"][0][0].astype(np.float64)

    print("Preprocesando BDR+ (P1)...")
    steps_pos = _preprocess_steps(raw_pos)
    proc_pos  = steps_pos[-1][0]

    print("Preprocesando BDR- (P19)...")
    steps_neg = _preprocess_steps(raw_neg)
    proc_neg  = steps_neg[-1][0]

    print("\nGenerando figuras...")

    _fig1_cruda_vs_preprocesada(raw_pos, proc_pos, raw_neg, proc_neg, OUTPUT_DIR)
    _fig_pasos(steps_pos, "BDR+ (P1)",  "fig2_pasos_preprocesado_BDRpos.png", OUTPUT_DIR)
    _fig_pasos(steps_neg, "BDR- (P19)", "fig3_pasos_preprocesado_BDRneg.png", OUTPUT_DIR)
    _fig4_psd(raw_pos, proc_pos, raw_neg, proc_neg, OUTPUT_DIR)
    _fig5_espectrograma(raw_pos, proc_pos, raw_neg, proc_neg, OUTPUT_DIR)

    print("\nPreprocesado completado.")
    print(f"  Señal BDR+ original : {len(raw_pos):>8} muestras @ {FS_ORIGINAL} Hz")
    print(f"  Señal BDR+ procesada: {len(proc_pos):>8} muestras @ {FS_TARGET} Hz")
    print(f"  Señal BDR- original : {len(raw_neg):>8} muestras @ {FS_ORIGINAL} Hz")
    print(f"  Señal BDR- procesada: {len(proc_neg):>8} muestras @ {FS_TARGET} Hz")
