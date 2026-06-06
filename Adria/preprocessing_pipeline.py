"""
PIPELINE DE PREPROCESSAMENT DE SENYALS RESPIRATÒRIES
=====================================================
Processa els fitxers P*.mat i C*.mat seguint exactament l'especificació:
  1. Lectura de la senyal bruta (LabChart / BIOPAC format)
  2. Resample 12500 → 4000 Hz  (resample_poly, sense interpolació lineal)
  3. Filtre Butterworth pas-banda 70–1900 Hz, ordre 8, zero-phase
  4. Filtre Notch Comb  50 Hz + harmònics, BW=1 Hz, zero-phase
  5. Normalització per segment (z-score robusta)
  6. Segmentació inspiració / expiració amb fitxers tP*.mat / tC*.mat
  7. Visualitzacions comparatives pas a pas
  8. Guardat de totes les senyals + metadades per a ML/DL posterior

Ús:
  python preprocessing_pipeline.py

Sortides (carpeta ./preprocessed/):
  signals.pkl          → llista de segments numpy (N~14900)
  metadata.csv         → participant, prepost, channel, phase per segment
  plots/               → figures PNG de cada pas de preprocessament
"""

import os, sys, time, pickle
import numpy as np
import scipy.io
from scipy.signal import (resample_poly, butter, filtfilt,
                           iirnotch, welch, sosfilt, sosfiltfilt)
from scipy.stats import zscore
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── Paràmetres globals ────────────────────────────────────────────────────────
FS_ORIG   = 12500      # Hz als fitxers .mat
FS_TARGET = 4000       # Hz objectiu
UP, DOWN  = 8, 25      # 4000/12500 = 8/25
BP_LOW    = 70         # Hz  pas-banda inferior
BP_HIGH   = 1900       # Hz  pas-banda superior
BP_ORDER  = 8          # ordre Butterworth
NOTCH_F0  = 50         # Hz  fonamental notch
NOTCH_BW  = 1.0        # Hz  amplada de banda notch
PLOT_DUR  = 3.0        # s   de senyal que es mostra als gràfics
PLOT_DEMO_CH  = 0      # canal per al gràfic demostratiu
PLOT_DEMO_BLK = 0      # bloc (maniobra) per al gràfic demostratiu

# Paleta de colors coherent
C = {
    'raw':      '#e63946',   # vermell: senyal bruta
    'resamp':   '#f4a261',   # taronja: remostreig
    'butter':   '#2a9d8f',   # verd mar: Butterworth
    'notch':    '#457b9d',   # blau: notch
    'norm':     '#6a4c93',   # violeta: normalitzada
    'insp':     '#2196F3',   # blau: inspiració
    'exp':      '#FF5722',   # taronja: espiració
    'bg':       '#0f1117',
    'grid':     '#2a2d35',
    'text':     '#e8eaf0',
}

plt.rcParams.update({
    'figure.facecolor':  C['bg'],
    'axes.facecolor':    C['bg'],
    'axes.edgecolor':    C['grid'],
    'axes.labelcolor':   C['text'],
    'xtick.color':       C['text'],
    'ytick.color':       C['text'],
    'text.color':        C['text'],
    'grid.color':        C['grid'],
    'grid.linewidth':    0.5,
    'font.family':       'monospace',
    'axes.titlesize':    11,
    'axes.labelsize':    9,
})


# =============================================================================
# 1. LECTURA DEL FITXER .MAT
# =============================================================================

def read_mat(path):
    """
    Llegeix un fitxer .mat (format LabChart/BIOPAC) i retorna:
      signals[ch][blk] = array numpy 1-D float64
      samplerate, titles
    """
    mat       = scipy.io.loadmat(path)
    data      = mat['data'].ravel().astype(np.float64)
    datastart = mat['datastart']   # (nch, nblk) base-1
    dataend   = mat['dataend']     # (nch, nblk) base-1
    samplerate= float(mat['samplerate'].ravel()[0])
    titles    = list(mat['titles']) if 'titles' in mat else ['CH0','CH1']
    nch, nblk = datastart.shape
    signals   = {}
    for ch in range(nch):
        signals[ch] = {}
        for blk in range(nblk):
            s = int(datastart[ch, blk]) - 1
            e = int(dataend[ch, blk])
            signals[ch][blk] = data[s:e]
    return signals, samplerate, titles, nch, nblk


# =============================================================================
# 2. CADENA DE PREPROCESSAMENT
# =============================================================================

def step_resample(sig, fs_orig=FS_ORIG, fs_target=FS_TARGET,
                  up=UP, down=DOWN):
    """Resample sense interpolació lineal (resample_poly usa filtre anti-aliàsing)."""
    return resample_poly(sig, up, down)


def step_butterworth(sig, fs=FS_TARGET,
                     low=BP_LOW, high=BP_HIGH, order=BP_ORDER):
    """Filtre pas-banda Butterworth zero-phase (filtfilt)."""
    nyq  = fs / 2.0
    b, a = butter(order, [low/nyq, high/nyq], btype='bandpass')
    return filtfilt(b, a, sig)


def step_notch_comb(sig, fs=FS_TARGET, f0=NOTCH_F0, bw=NOTCH_BW):
    """
    Cascada de filtres notch IIR zero-phase a f0 i tots els seus harmònics
    fins a fs/2.  Q = f0/bw  →  amplada de banda = bw Hz.
    """
    nyq  = fs / 2.0
    freq = f0
    out  = sig.copy()
    while freq < nyq:
        Q        = freq / bw
        b_n, a_n = iirnotch(freq, Q, fs=fs)
        out      = filtfilt(b_n, a_n, out)
        freq    += f0
    return out


def step_normalize(sig):
    """
    Normalització robusta per segment:
      z = (x - mediana) / (1.4826 * MAD)
    Molt menys sensible a artefactes puntuals que z-score estàndard.
    """
    med  = np.median(sig)
    mad  = np.median(np.abs(sig - med))
    if mad < 1e-12:
        return sig - med
    return (sig - med) / (1.4826 * mad)


def preprocess_signal(raw, fs_orig=FS_ORIG):
    """
    Aplica tots els passos en ordre i retorna:
      (resampled, butterworded, notched, normalized)
    """
    s1 = step_resample(raw,   fs_orig=fs_orig)
    s2 = step_butterworth(s1)
    s3 = step_notch_comb(s2)
    s4 = step_normalize(s3)
    return s1, s2, s3, s4


# =============================================================================
# 3. VISUALITZACIONS PAS A PAS
# =============================================================================

def _t(sig, fs):
    return np.arange(len(sig)) / fs


def plot_preprocessing_steps(raw, s1, s2, s3, s4,
                              fs_orig=FS_ORIG, fs_target=FS_TARGET,
                              title="", save_path=None, dur=PLOT_DUR):
    """
    Figura principal: 5 files (una per pas) × 2 columnes (temps | PSD).
    """
    steps = [
        (raw, fs_orig,   C['raw'],    '① Senyal bruta  (12500 Hz)'),
        (s1,  fs_target, C['resamp'], f'② Resample  ({fs_orig}→{fs_target} Hz)'),
        (s2,  fs_target, C['butter'], f'③ Butterworth BP  {BP_LOW}–{BP_HIGH} Hz  ord.{BP_ORDER}'),
        (s3,  fs_target, C['notch'],  f'④ Notch Comb  {NOTCH_F0} Hz + harmònics  BW={NOTCH_BW} Hz'),
        (s4,  fs_target, C['norm'],   '⑤ Normalització robusta  (MAD z-score)'),
    ]

    fig = plt.figure(figsize=(18, 18), facecolor=C['bg'])
    fig.suptitle(title, fontsize=13, color=C['text'], y=0.995,
                 fontweight='bold', fontfamily='monospace')
    gs  = gridspec.GridSpec(len(steps), 2, figure=fig,
                            hspace=0.55, wspace=0.30,
                            left=0.07, right=0.97, top=0.975, bottom=0.04)

    for row, (sig, fs, col, label) in enumerate(steps):
        n_plot = min(int(dur * fs), len(sig))
        t      = _t(sig[:n_plot], fs)

        # ── columna esquerra: domini temporal ────────────────────────────────
        ax_t = fig.add_subplot(gs[row, 0])
        ax_t.plot(t, sig[:n_plot], color=col, linewidth=0.6, alpha=0.9)
        ax_t.set_title(label, color=col, fontsize=10, pad=4, loc='left')
        ax_t.set_xlabel('Temps (s)', fontsize=8)
        ax_t.set_ylabel('Amplitud (V)', fontsize=8)
        ax_t.grid(True, alpha=0.3)
        ax_t.set_xlim(0, t[-1])
        # Anotació RMS
        rms = np.sqrt(np.mean(sig**2))
        ax_t.text(0.98, 0.95, f'RMS={rms:.4f}', transform=ax_t.transAxes,
                  ha='right', va='top', fontsize=7.5,
                  color=col, alpha=0.9,
                  bbox=dict(facecolor='#1a1d26', edgecolor=col, alpha=0.6, pad=2))

        # ── columna dreta: PSD (Welch) ────────────────────────────────────────
        ax_f = fig.add_subplot(gs[row, 1])
        nperseg = min(1024, len(sig))
        freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
        ax_f.semilogy(freqs, psd, color=col, linewidth=0.8)
        ax_f.set_xlabel('Freqüència (Hz)', fontsize=8)
        ax_f.set_ylabel('PSD (V²/Hz)', fontsize=8)
        ax_f.grid(True, alpha=0.3)
        ax_f.set_xlim(0, fs/2)
        # Línies de referència de la banda objectiu
        if row >= 1:
            ax_f.axvline(BP_LOW,  color='#ffffff', linewidth=0.7,
                         linestyle='--', alpha=0.4, label=f'{BP_LOW} Hz')
            ax_f.axvline(BP_HIGH, color='#ffffff', linewidth=0.7,
                         linestyle='--', alpha=0.4, label=f'{BP_HIGH} Hz')
        # Marcar 50 Hz i harmònics
        if row >= 1:
            for h in range(50, int(fs/2), 50):
                ax_f.axvline(h, color='#ffcc00', linewidth=0.5,
                             linestyle=':', alpha=0.25)
        if row == 0:
            ax_f.legend(fontsize=7, framealpha=0.3,
                        loc='upper right', labelcolor=C['text'])

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches='tight',
                    facecolor=C['bg'])
    plt.close(fig)
    return fig


def plot_spectral_comparison(raw, s1, s2, s3, s4,
                             fs_orig=FS_ORIG, fs_target=FS_TARGET,
                             save_path=None):
    """
    Superposició de les 5 PSD per veure clarament l'efecte de cada filtre.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5),
                                    facecolor=C['bg'])
    fig.suptitle('Comparació Espectral – Tots els Passos Superposats',
                 fontsize=12, color=C['text'])

    steps = [
        (raw, fs_orig,   C['raw'],    'Bruta (12500Hz)'),
        (s1,  fs_target, C['resamp'], 'Resample (4000Hz)'),
        (s2,  fs_target, C['butter'], f'Butterworth {BP_LOW}–{BP_HIGH}Hz'),
        (s3,  fs_target, C['notch'],  'Notch Comb'),
        (s4,  fs_target, C['norm'],   'Normalitzada'),
    ]

    for sig, fs, col, lbl in steps:
        f, p = welch(sig, fs=fs, nperseg=min(1024, len(sig)))
        ax1.semilogy(f, p, color=col, linewidth=1.0, label=lbl, alpha=0.85)
        ax2.semilogy(f, p, color=col, linewidth=1.0, label=lbl, alpha=0.85)

    # Zoom zona d'interès
    ax2.set_xlim(0, 2100)
    ax2.axvspan(BP_LOW, BP_HIGH, alpha=0.06, color='#ffffff',
                label=f'Banda {BP_LOW}–{BP_HIGH}Hz')
    for h in range(50, 2100, 50):
        ax2.axvline(h, color='#ffcc00', linewidth=0.4, linestyle=':', alpha=0.3)

    for ax in (ax1, ax2):
        ax.set_xlabel('Freqüència (Hz)', fontsize=9)
        ax.set_ylabel('PSD (V²/Hz)', fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, framealpha=0.25, loc='upper right',
                  labelcolor=C['text'])
        ax.set_facecolor(C['bg'])

    ax1.set_title('Espectre complet', color=C['text'])
    ax2.set_title('Zoom 0–2100 Hz  (+ harmònics 50Hz en groc)', color=C['text'])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=C['bg'])
    plt.close(fig)


def plot_notch_effect(s2, s3, fs=FS_TARGET, save_path=None):
    """
    Zoom al voltant dels harmònics de 50 Hz per veure l'efecte del notch.
    """
    harmonics = [50, 100, 150, 200, 250, 300]
    n_h = len(harmonics)
    fig, axes = plt.subplots(2, n_h, figsize=(18, 6), facecolor=C['bg'])
    fig.suptitle(f'Efecte del Notch Comb  (BW={NOTCH_BW} Hz) – Zoom a cada harmònic',
                 fontsize=11, color=C['text'])

    f2, p2 = welch(s2, fs=fs, nperseg=4096)
    f3, p3 = welch(s3, fs=fs, nperseg=4096)

    for col_i, hz in enumerate(harmonics):
        # Espectres
        ax_s = axes[0, col_i]
        for f, p, color, lbl in [(f2, p2, C['butter'], 'Pre-notch'),
                                  (f3, p3, C['notch'],  'Post-notch')]:
            ax_s.semilogy(f, p, color=color, linewidth=0.9, label=lbl)
        ax_s.set_xlim(hz-10, hz+10)
        ax_s.axvline(hz, color='#ffcc00', linewidth=1, linestyle='--', alpha=0.6)
        ax_s.set_title(f'{hz} Hz', color=C['text'], fontsize=9)
        ax_s.grid(True, alpha=0.3)
        if col_i == 0:
            ax_s.set_ylabel('PSD (V²/Hz)', fontsize=8)
            ax_s.legend(fontsize=7, framealpha=0.2, labelcolor=C['text'])
        ax_s.set_facecolor(C['bg'])

        # Diferència en dB
        ax_d = axes[1, col_i]
        mask = (f2 >= hz-10) & (f2 <= hz+10)
        if mask.sum() > 0:
            db_diff = 10*np.log10(p2[mask]+1e-30) - 10*np.log10(p3[mask]+1e-30)
            ax_d.plot(f2[mask], db_diff, color='#f4e04d', linewidth=1.0)
            ax_d.axvline(hz, color='#ffcc00', linewidth=1, linestyle='--', alpha=0.6)
            ax_d.axhline(0, color='#555', linewidth=0.5)
        ax_d.set_xlabel('Hz', fontsize=8)
        ax_d.set_title('Atenuació (dB)', fontsize=8, color=C['text'])
        ax_d.grid(True, alpha=0.3)
        ax_d.set_facecolor(C['bg'])
        if col_i == 0:
            ax_d.set_ylabel('Pre – Post (dB)', fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=C['bg'])
    plt.close(fig)


def plot_segmentation(sig_proc, time_marks, fs=FS_TARGET,
                      participant_id="", save_path=None, dur=30.0):
    """
    Visualitza la senyal processada amb les regions d'inspiració i espiració
    ressaltades per als primers `dur` segons.
    """
    n_plot = min(int(dur * fs), len(sig_proc))
    t      = _t(sig_proc[:n_plot], fs)

    fig, ax = plt.subplots(figsize=(16, 4), facecolor=C['bg'])
    ax.plot(t, sig_proc[:n_plot], color='#adb5bd', linewidth=0.5, alpha=0.85)
    ax.set_title(f'Segmentació respiratòria – {participant_id}  '
                 f'(primers {dur:.0f}s)',
                 color=C['text'], fontsize=10)
    ax.set_xlabel('Temps (s)'); ax.set_ylabel('Amplitud (norm.)')
    ax.grid(True, alpha=0.2)
    ax.set_facecolor(C['bg'])

    insp_patch = plt.Rectangle((0,0), 0, 0, fc=C['insp'], alpha=0.3,
                                label='Inspiració')
    exp_patch  = plt.Rectangle((0,0), 0, 0, fc=C['exp'],  alpha=0.3,
                                label='Espiració')

    for cyc in time_marks:
        cyc = np.asarray(cyc).flatten()
        if len(cyc) < 4: continue
        tsi, tei, tse, tee = cyc[:4]
        if tsi > dur: break
        # Inspiració
        tei_c = min(tei, dur)
        ax.axvspan(tsi, tei_c, color=C['insp'], alpha=0.18)
        # Espiració
        if tse < dur:
            tee_c = min(tee, dur)
            ax.axvspan(tse, tee_c, color=C['exp'], alpha=0.18)

    ax.legend(handles=[insp_patch, exp_patch], loc='upper right',
              fontsize=9, framealpha=0.25, labelcolor=C['text'])
    ax.set_xlim(0, dur)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=C['bg'])
    plt.close(fig)


def plot_segment_examples(segments_insp, segments_exp, fs=FS_TARGET,
                          n_show=6, save_path=None):
    """
    Mostra N exemples de segments d'inspiració i espiració per comparar.
    """
    fig, axes = plt.subplots(2, n_show, figsize=(18, 6), facecolor=C['bg'])
    fig.suptitle('Exemples de Segments Segmentats – Inspiració vs Espiració',
                 fontsize=11, color=C['text'])

    for col, (segs, color, phase) in enumerate([
            (segments_insp[:n_show], C['insp'], 'Inspiració'),
            (segments_exp[:n_show],  C['exp'],  'Espiració')]):
        for i, seg in enumerate(segs[:n_show]):
            if i >= n_show: break
            ax = axes[col, i]
            t  = _t(seg, fs)
            ax.plot(t, seg, color=color, linewidth=0.7)
            ax.set_title(f'{phase} #{i+1}\n{len(seg)/fs:.2f}s',
                         fontsize=8, color=color)
            ax.set_xlabel('s', fontsize=7)
            ax.set_facecolor(C['bg'])
            ax.grid(True, alpha=0.2)
            if i > 0: ax.set_yticklabels([])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=C['bg'])
    plt.close(fig)


def plot_pipeline_summary(all_signals, metadata_df, save_path=None):
    """
    Resum estadístic de tots els segments: distribució de longituds,
    amplitud RMS, i recompte per participant/fase.
    """
    durations = np.array([len(s)/FS_TARGET for s in all_signals])
    rms_vals  = np.array([np.sqrt(np.mean(s**2)) for s in all_signals])

    fig = plt.figure(figsize=(16, 10), facecolor=C['bg'])
    fig.suptitle('Resum del Dataset Preprocessat', fontsize=13,
                 color=C['text'], fontweight='bold')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                            left=0.07, right=0.97, top=0.93, bottom=0.08)

    # 1. Histograma durades
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(durations, bins=50, color=C['notch'], alpha=0.8, edgecolor='none')
    ax1.axvline(np.median(durations), color='#f4e04d', linestyle='--',
                linewidth=1.2, label=f'Mediana {np.median(durations):.2f}s')
    ax1.set_title('Distribució de durades', color=C['text'])
    ax1.set_xlabel('Durada (s)'); ax1.set_ylabel('Freqüència')
    ax1.legend(fontsize=8, labelcolor=C['text'], framealpha=0.2)
    ax1.set_facecolor(C['bg']); ax1.grid(True, alpha=0.2)

    # 2. Histograma RMS
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(rms_vals, bins=50, color=C['butter'], alpha=0.8, edgecolor='none')
    ax2.axvline(np.median(rms_vals), color='#f4e04d', linestyle='--',
                linewidth=1.2, label=f'Mediana {np.median(rms_vals):.4f}')
    ax2.set_title('Distribució RMS', color=C['text'])
    ax2.set_xlabel('RMS (u.a.)'); ax2.set_ylabel('Freqüència')
    ax2.legend(fontsize=8, labelcolor=C['text'], framealpha=0.2)
    ax2.set_facecolor(C['bg']); ax2.grid(True, alpha=0.2)

    # 3. Scatter durada vs RMS
    ax3 = fig.add_subplot(gs[0, 2])
    sc  = ax3.scatter(durations, rms_vals,
                      c=metadata_df['phase'].values,
                      cmap='coolwarm', s=2, alpha=0.5)
    ax3.set_title('Durada vs RMS  (color=fase)', color=C['text'])
    ax3.set_xlabel('Durada (s)'); ax3.set_ylabel('RMS')
    cbar = plt.colorbar(sc, ax=ax3); cbar.set_label('1=Insp  2=Esp', fontsize=7)
    ax3.set_facecolor(C['bg']); ax3.grid(True, alpha=0.2)

    # 4. Segments per participant
    ax4 = fig.add_subplot(gs[1, 0])
    counts = metadata_df.groupby('participant').size()
    colors_bar = ['#d62728' if p<=23 else '#2ca02c' for p in counts.index]
    ax4.bar(counts.index, counts.values, color=colors_bar, alpha=0.85, width=0.7)
    ax4.set_title('Segments per participant\n(vermell=pacient, verd=control)',
                  color=C['text'])
    ax4.set_xlabel('Participant ID'); ax4.set_ylabel('# segments')
    ax4.set_facecolor(C['bg']); ax4.grid(True, alpha=0.2, axis='y')

    # 5. Inspiració vs Espiració
    ax5 = fig.add_subplot(gs[1, 1])
    phase_counts = metadata_df['phase'].value_counts().sort_index()
    ax5.bar(['Inspiració', 'Espiració'], phase_counts.values,
            color=[C['insp'], C['exp']], alpha=0.85, width=0.5)
    ax5.set_title('Segments per fase', color=C['text'])
    ax5.set_ylabel('# segments')
    ax5.set_facecolor(C['bg']); ax5.grid(True, alpha=0.2, axis='y')
    for i, v in enumerate(phase_counts.values):
        ax5.text(i, v+5, str(v), ha='center', color=C['text'], fontsize=9)

    # 6. Pre vs Post BD
    ax6 = fig.add_subplot(gs[1, 2])
    pp_counts = metadata_df['prepost'].value_counts().sort_index()
    ax6.bar(['Pre-BD', 'Post-BD'], pp_counts.values,
            color=['#e76f51', '#06d6a0'], alpha=0.85, width=0.5)
    ax6.set_title('Segments Pre vs Post\nbroncodilatador', color=C['text'])
    ax6.set_ylabel('# segments')
    ax6.set_facecolor(C['bg']); ax6.grid(True, alpha=0.2, axis='y')
    for i, v in enumerate(pp_counts.values):
        ax6.text(i, v+5, str(v), ha='center', color=C['text'], fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=C['bg'])
    plt.close(fig)
    print(f"  Resum guardat → {save_path}")


# =============================================================================
# 4. LECTURA DE MARQUES TEMPORALS
# =============================================================================

def read_time_marks(path):
    """
    Llegeix el fitxer tP*.mat o tC*.mat.
    Retorna: dict  {blk_idx: array (N_cicles × 4)}
    """
    mat   = scipy.io.loadmat(path)
    keys  = [k for k in mat if not k.startswith('__')]
    if not keys:
        return {}
    cell  = mat[keys[0]]
    # Normalitzar forma
    if cell.ndim == 2:
        cell = cell[0] if cell.shape[0] == 1 else cell[:, 0]
    marks = {}
    for blk, cm in enumerate(cell):
        if isinstance(cm, np.ndarray) and cm.dtype == object:
            cm = cm.flat[0] if cm.size else None
        if cm is None or not isinstance(cm, np.ndarray) or cm.size == 0:
            marks[blk] = np.empty((0, 4))
            continue
        marks[blk] = np.atleast_2d(cm)
    return marks


# =============================================================================
# 5. PIPELINE PRINCIPAL
# =============================================================================

def run_pipeline(data_dir, out_dir, plot_demo_only=False):
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    subjects = (
        [{'id': p,    'sig': f'P{p}.mat',  'time': f'tP{p}.mat'} for p in range(1, 24)] +
        [{'id': 23+c, 'sig': f'C{c}.mat',  'time': f'tC{c}.mat'} for c in range(1, 6)]
    )

    all_signals  = []
    meta_rows    = []
    demo_done    = False
    seg_examples = {'insp': [], 'exp': []}

    total_subj = sum(1 for s in subjects
                     if os.path.exists(os.path.join(data_dir, s['sig'])))
    print(f"\n  Participants detectats: {total_subj}")
    print(f"  Directori de sortida:   {out_dir}\n")

    for subj_i, subj in enumerate(subjects):
        pth_sig  = os.path.join(data_dir, subj['sig'])
        pth_time = os.path.join(data_dir, subj['time'])
        if not os.path.exists(pth_sig):
            continue

        t0 = time.time()
        pid    = subj['id']
        label  = 'P' if pid <= 23 else 'C'
        idx    = pid if pid <= 23 else pid - 23
        print(f"  [{subj_i+1:02d}/{total_subj}] {subj['sig']} … ", end='', flush=True)

        signals, fs_orig, titles, nch, nblk = read_mat(pth_sig)
        marks = read_time_marks(pth_time) if os.path.exists(pth_time) else {}

        for ch in range(nch):
            for blk in range(nblk):
                raw = signals[ch][blk]
                if len(raw) == 0:
                    continue

                # ── Preprocessament ───────────────────────────────────────────
                s1, s2, s3, s4 = preprocess_signal(raw, fs_orig=fs_orig)

                # ── Figura comparativa PAS A PAS (primera vegada per subjecte) ─
                if not demo_done or (ch == PLOT_DEMO_CH and blk == PLOT_DEMO_BLK):
                    if not demo_done:
                        name_base = f"{label}{idx}_ch{ch+1}_blk{blk+1}"
                        plot_preprocessing_steps(
                            raw, s1, s2, s3, s4,
                            fs_orig=fs_orig,
                            title=f'Passos de preprocessament – {name_base}  '
                                  f'({titles[ch] if ch < len(titles) else ""})',
                            save_path=os.path.join(plots_dir,
                                f'{name_base}_steps.png'))
                        plot_spectral_comparison(
                            raw, s1, s2, s3, s4,
                            save_path=os.path.join(plots_dir,
                                f'{name_base}_spectral_comparison.png'))
                        plot_notch_effect(
                            s2, s3,
                            save_path=os.path.join(plots_dir,
                                f'{name_base}_notch_zoom.png'))
                        demo_done = True

                # ── Segmentació ────────────────────────────────────────────────
                if blk not in marks or len(marks[blk]) == 0:
                    continue

                pre_post = 1 if blk < 3 else 2
                channel  = ch + 1           # 1=inferior, 2=superior

                # Figura de segmentació (primera maniobra, primer canal)
                if ch == 0 and blk == 0 and pid == subjects[0]['id']:
                    plot_segmentation(
                        s4, marks[blk],
                        participant_id=f"{label}{idx}",
                        save_path=os.path.join(plots_dir,
                            f'{label}{idx}_ch1_blk1_segmentation.png'))

                for cyc in marks[blk]:
                    cyc = np.asarray(cyc).flatten()
                    if len(cyc) < 4:
                        continue
                    tsi, tei, tse, tee = cyc[:4]

                    # Inspiració
                    i0, i1 = int(round(tsi*FS_TARGET)), int(round(tei*FS_TARGET))
                    if 0 <= i0 < len(s4) and i1 > i0:
                        seg = s4[i0:i1].copy()
                        if len(seg) > 0:
                            all_signals.append(seg)
                            meta_rows.append({'participant': pid,
                                              'prepost': pre_post,
                                              'channel': channel,
                                              'phase': 1})
                            if len(seg_examples['insp']) < 6:
                                seg_examples['insp'].append(seg)

                    # Espiració
                    i0, i1 = int(round(tse*FS_TARGET)), int(round(tee*FS_TARGET))
                    if 0 <= i0 < len(s4) and i1 > i0:
                        seg = s4[i0:i1].copy()
                        if len(seg) > 0:
                            all_signals.append(seg)
                            meta_rows.append({'participant': pid,
                                              'prepost': pre_post,
                                              'channel': channel,
                                              'phase': 2})
                            if len(seg_examples['exp']) < 6:
                                seg_examples['exp'].append(seg)

        elapsed = time.time() - t0
        print(f"OK  ({elapsed:.1f}s)  →  total segments fins ara: {len(all_signals)}")

    # ── Figura d'exemples de segments ────────────────────────────────────────
    if seg_examples['insp'] and seg_examples['exp']:
        plot_segment_examples(
            seg_examples['insp'], seg_examples['exp'],
            save_path=os.path.join(plots_dir, 'segment_examples.png'))

    # ── Metadades ─────────────────────────────────────────────────────────────
    metadata_df = pd.DataFrame(meta_rows)

    # ── Figura de resum ────────────────────────────────────────────────────────
    if len(all_signals) > 0:
        plot_pipeline_summary(
            all_signals, metadata_df,
            save_path=os.path.join(plots_dir, 'dataset_summary.png'))

    return all_signals, metadata_df


# =============================================================================
# 6. GUARDAT PER A ML/DL
# =============================================================================

def save_dataset(all_signals, metadata_df, out_dir):
    """
    Guarda:
      signals.pkl     → llista Python de N arrays numpy (longitud variable)
      metadata.csv    → DataFrame: participant, prepost, channel, phase
      signals_padded.npy → matriu (N, max_len) amb zero-padding [opcional DL]
      lengths.npy     → longitud real de cada segment
    """
    pkl_path = os.path.join(out_dir, 'signals.pkl')
    csv_path = os.path.join(out_dir, 'metadata.csv')

    with open(pkl_path, 'wb') as f:
        pickle.dump(all_signals, f, protocol=4)
    metadata_df.to_csv(csv_path, index=True, index_label='segment_id')

    # Matriu amb padding per a xarxes DL
    lengths = np.array([len(s) for s in all_signals])
    max_len = int(np.percentile(lengths, 95))   # p95 per no inflar massa
    X_pad   = np.zeros((len(all_signals), max_len), dtype=np.float32)
    for i, s in enumerate(all_signals):
        L = min(len(s), max_len)
        X_pad[i, :L] = s[:L]
    np.save(os.path.join(out_dir, 'signals_padded.npy'), X_pad)
    np.save(os.path.join(out_dir, 'lengths.npy'), lengths)

    print(f"\n  signals.pkl         → {pkl_path}")
    print(f"  metadata.csv        → {csv_path}")
    print(f"  signals_padded.npy  → shape {X_pad.shape}  (p95 len={max_len})")
    print(f"  lengths.npy         → {lengths.shape}")
    print(f"\n  Total segments guardats: {len(all_signals)}")
    print(f"  Durada mitja:            {np.mean(lengths/FS_TARGET):.3f} s")
    print(f"  Durada mediana:          {np.median(lengths/FS_TARGET):.3f} s")
    return pkl_path, csv_path


# =============================================================================
# BLOC PRINCIPAL
# =============================================================================

if __name__ == '__main__':
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR  = os.path.join(DATA_DIR, 'preprocessed')
    os.makedirs(OUT_DIR, exist_ok=True)

    print('=' * 60)
    print('  PIPELINE PREPROCESSAMENT SENYALS RESPIRATÒRIES')
    print('=' * 60)
    print(f'  Directori de dades: {DATA_DIR}')

    # Comprova si hi ha almenys un fitxer de dades
    has_data = any(os.path.exists(os.path.join(DATA_DIR, f'P{i}.mat'))
                   for i in range(1, 24))
    has_data = has_data or any(os.path.exists(os.path.join(DATA_DIR, f'C{i}.mat'))
                                for i in range(1, 6))
    if not has_data:
        print('\n  ERROR: No s\'han trobat fitxers P*.mat ni C*.mat al directori.')
        sys.exit(1)

    t_start = time.time()
    all_signals, metadata_df = run_pipeline(DATA_DIR, OUT_DIR)
    save_dataset(all_signals, metadata_df, OUT_DIR)

    print(f'\n  Temps total: {time.time()-t_start:.1f} s')
    plots_list = os.listdir(os.path.join(OUT_DIR, 'plots'))
    print(f'  Figures generades ({len(plots_list)}):')
    for f in sorted(plots_list):
        print(f'    · {f}')
    print('=' * 60)
