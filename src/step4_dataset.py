"""
Construcción del dataset completo de segmentos respiratorios.

Recorre los 28 sujetos (23 pacientes + 5 controles), aplica la cadena
completa de preprocesado y segmentación, y ensambla las 14 900 señales
junto con sus cuatro vectores de metadatos:
  v_subject  — número de sujeto (1–23 pacientes, 24–28 controles)
  v_bd       — 1 = pre-BD (maniobras 1–3), 2 = post-BD (maniobras 4–6)
  v_channel  — 1 = canal inferior, 2 = canal superior
  v_phase    — 1 = inspiración, 2 = espiración
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Permite importar módulos vecinos desde src/
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from step1_read_signals import read_signals
from step2_preprocessing import preprocess_signal
from step3_segmentation import load_markers, segment_signal

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
N_PATIENTS     = 23
N_CONTROLS     = 5
N_CHANNELS     = 2
N_MANOEUVRES   = 6
N_PRE_BD       = 3   # primeras 3 maniobras (índices 0–2) = pre-BD
EXPECTED_TOTAL = 14900
FS_TARGET      = 4000   # Hz

_PROJECT_ROOT  = os.path.dirname(_SRC_DIR)
DATA_DIR       = os.path.join(_PROJECT_ROOT, "Data")
METADATA_CSV   = os.path.join(_PROJECT_ROOT, "Data", "database", "subject_metadata.csv")
OUTPUT_FIG_DIR = os.path.join(_PROJECT_ROOT, "outputs", "figures", "step4")
OUTPUT_RES_DIR = os.path.join(_PROJECT_ROOT, "outputs", "results", "step4")
# ---------------------------------------------------------------------------


# ===========================================================================
# Carga de metadatos y construcción de la lista de sujetos
# ===========================================================================

def _load_metadata(csv_path: str) -> dict[str, str]:
    """Carga el CSV de metadatos y devuelve {subject_id: bdr_label}."""
    metadata: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            metadata[row["subject_id"]] = row["bdr_label"]
    return metadata


def _build_subject_list(
    metadata: dict[str, str],
) -> list[tuple[int, str, str, str, str, str]]:
    """
    Construye la lista ordenada de los 28 sujetos a procesar.

    Retorna
    -------
    list of (subj_num, subj_id, sig_file, mkr_file, bdr_label, subj_type)
    """
    subjects: list[tuple[int, str, str, str, str, str]] = []

    for i in range(1, N_PATIENTS + 1):
        sid = f"P{i}"
        subjects.append((
            i, sid,
            os.path.join(DATA_DIR, f"{sid}.mat"),
            os.path.join(DATA_DIR, f"t{sid}.mat"),
            metadata[sid], "patient",
        ))

    for i in range(1, N_CONTROLS + 1):
        sid = f"C{i}"
        subjects.append((
            N_PATIENTS + i, sid,
            os.path.join(DATA_DIR, f"{sid}.mat"),
            os.path.join(DATA_DIR, f"t{sid}.mat"),
            metadata[sid], "control",
        ))

    return subjects


# ===========================================================================
# Construcción del dataset
# ===========================================================================

def build_dataset(
    subjects: list[tuple],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Recorre todos los sujetos, canales, maniobras y ciclos para construir
    el conjunto de datos completo.

    Orden de iteración:
        sujeto → canal → maniobra → ciclo → [inspiración, espiración]

    Retorna
    -------
    all_signals    : list[np.ndarray] de 14 900 segmentos 1D
    v_subject      : np.ndarray int32 (14900,)
    v_bd           : np.ndarray int32 (14900,)
    v_channel      : np.ndarray int32 (14900,)
    v_phase        : np.ndarray int32 (14900,)
    counts_matrix  : np.ndarray int32 (28, 6) — segmentos totales por sujeto y maniobra
    """
    all_signals: list[np.ndarray] = []
    v_subject_lst: list[int] = []
    v_bd_lst:      list[int] = []
    v_channel_lst: list[int] = []
    v_phase_lst:   list[int] = []

    n_subjects = N_PATIENTS + N_CONTROLS
    counts_matrix = np.zeros((n_subjects, N_MANOEUVRES), dtype=np.int32)

    for subj_idx, (subj_num, subj_id, sig_file, mkr_file, bdr_label, _) in enumerate(subjects):
        print(f"Procesando {subj_id} ({bdr_label})... [{subj_idx + 1}/{n_subjects}]")

        sdata   = read_signals(sig_file)
        markers = load_markers(mkr_file)

        for ch_idx in range(N_CHANNELS):
            ch_num = ch_idx + 1   # 1 = inferior (índice 0), 2 = superior (índice 1)

            for man_idx in range(N_MANOEUVRES):
                bd_val = 1 if man_idx < N_PRE_BD else 2

                raw  = sdata["signals"][ch_idx][man_idx].astype(np.float64)
                proc = preprocess_signal(raw)
                segs = segment_signal(proc, markers[man_idx])

                n_ciclos = len(segs["inspiracion"])

                for k in range(n_ciclos):
                    # Inspiración
                    all_signals.append(segs["inspiracion"][k])
                    v_subject_lst.append(subj_num)
                    v_bd_lst.append(bd_val)
                    v_channel_lst.append(ch_num)
                    v_phase_lst.append(1)

                    # Espiración
                    all_signals.append(segs["espiracion"][k])
                    v_subject_lst.append(subj_num)
                    v_bd_lst.append(bd_val)
                    v_channel_lst.append(ch_num)
                    v_phase_lst.append(2)

                # Conteo por sujeto y maniobra (ambas fases, ambos canales)
                counts_matrix[subj_idx, man_idx] += 2 * n_ciclos

    return (
        all_signals,
        np.array(v_subject_lst, dtype=np.int32),
        np.array(v_bd_lst,      dtype=np.int32),
        np.array(v_channel_lst, dtype=np.int32),
        np.array(v_phase_lst,   dtype=np.int32),
        counts_matrix,
    )


# ===========================================================================
# Validación
# ===========================================================================

def _validate(
    all_signals: list[np.ndarray],
    v_subject: np.ndarray,
    v_bd: np.ndarray,
    v_channel: np.ndarray,
    v_phase: np.ndarray,
    subjects: list[tuple],
) -> None:
    """
    Comprueba que el dataset tiene exactamente EXPECTED_TOTAL segmentos y que
    los cinco vectores de metadatos tienen la misma longitud que all_signals.
    Si el total no coincide, imprime un desglose por sujeto y maniobra (pre-BD
    y post-BD) antes de lanzar la excepción.
    """
    n = len(all_signals)
    assert len(v_subject) == n, f"v_subject tiene {len(v_subject)} entradas, se esperaban {n}"
    assert len(v_bd)      == n, f"v_bd tiene {len(v_bd)} entradas, se esperaban {n}"
    assert len(v_channel) == n, f"v_channel tiene {len(v_channel)} entradas, se esperaban {n}"
    assert len(v_phase)   == n, f"v_phase tiene {len(v_phase)} entradas, se esperaban {n}"

    if n != EXPECTED_TOTAL:
        print(f"\nERROR: se esperaban {EXPECTED_TOTAL} segmentos, se obtuvieron {n}.")
        print("Desglose por sujeto y maniobra (pre-BD / post-BD):")
        for subj_num, subj_id, *_ in subjects:
            mask      = v_subject == subj_num
            cnt_pre   = int(((v_bd == 1) & mask).sum())
            cnt_post  = int(((v_bd == 2) & mask).sum())
            cnt_total = int(mask.sum())
            print(f"  {subj_id}: {cnt_total} total  (pre-BD: {cnt_pre}, post-BD: {cnt_post})")
        raise AssertionError(
            f"El dataset contiene {n} segmentos en lugar de {EXPECTED_TOTAL}."
        )
    print(f"\nValidación OK: {n} segmentos.")


# ===========================================================================
# Guardado de resultados
# ===========================================================================

def _save_outputs(
    all_signals: list[np.ndarray],
    v_subject: np.ndarray,
    v_bd: np.ndarray,
    v_channel: np.ndarray,
    v_phase: np.ndarray,
    subjects: list[tuple],
    out_dir: str,
) -> np.ndarray:
    """
    Guarda dataset.npz, segment_lengths.npy y dataset_summary.csv.

    Retorna
    -------
    segment_lengths : np.ndarray int32 (14900,) — longitud de cada segmento en muestras
    """
    os.makedirs(out_dir, exist_ok=True)

    # Vectores de metadatos
    np.savez_compressed(
        os.path.join(out_dir, "dataset.npz"),
        v_subject=v_subject,
        v_bd=v_bd,
        v_channel=v_channel,
        v_phase=v_phase,
    )
    print(f"  Guardado: {os.path.join(out_dir, 'dataset.npz')}")

    # Longitudes de segmentos en muestras
    segment_lengths = np.array([len(s) for s in all_signals], dtype=np.int32)
    np.save(os.path.join(out_dir, "segment_lengths.npy"), segment_lengths)
    print(f"  Guardado: {os.path.join(out_dir, 'segment_lengths.npy')}")

    # Resumen por sujeto
    rows = []
    for subj_num, subj_id, _, _, bdr_label, subj_type in subjects:
        mask    = v_subject == subj_num
        seg_len = segment_lengths[mask]
        dur_ms  = seg_len / FS_TARGET * 1000.0
        rows.append({
            "subject_id":       subj_id,
            "type":             subj_type,
            "bdr_label":        bdr_label,
            "n_segments":       int(mask.sum()),
            "n_pre_bd":         int(((v_bd == 1) & mask).sum()),
            "n_post_bd":        int(((v_bd == 2) & mask).sum()),
            "n_insp":           int(((v_phase == 1) & mask).sum()),
            "n_esp":            int(((v_phase == 2) & mask).sum()),
            "n_ch1":            int(((v_channel == 1) & mask).sum()),
            "n_ch2":            int(((v_channel == 2) & mask).sum()),
            "mean_duration_ms": round(float(dur_ms.mean()), 2),
            "std_duration_ms":  round(float(dur_ms.std()), 2),
        })

    csv_path = os.path.join(out_dir, "dataset_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Guardado: {csv_path}")

    return segment_lengths


# ===========================================================================
# Resumen de consola
# ===========================================================================

def _print_summary(
    v_subject: np.ndarray,
    v_bd: np.ndarray,
    v_channel: np.ndarray,
    v_phase: np.ndarray,
    subjects: list[tuple],
) -> None:
    """Imprime el resumen final del dataset en consola."""
    bdrpos_nums = {s[0] for s in subjects if s[4] == "BDR+"}
    patient_nums = {s[0] for s in subjects if s[5] == "patient"}
    bdrneg_patient_nums = {s[0] for s in subjects if s[5] == "patient" and s[4] == "BDR-"}

    mask_bdrpos = np.isin(v_subject, list(bdrpos_nums))
    mask_ctrl   = ~np.isin(v_subject, list(patient_nums))
    mask_bdrneg = np.isin(v_subject, list(bdrneg_patient_nums))

    print(f"\n{'='*50}")
    print(f"Total de segmentos     : {len(v_subject)}")
    print(f"  BDR+ (pacientes)     : {int(mask_bdrpos.sum())}")
    print(f"  BDR- (pacientes)     : {int(mask_bdrneg.sum())}")
    print(f"  Controles            : {int(mask_ctrl.sum())}")
    print(f"  Pre-BD               : {int((v_bd == 1).sum())}")
    print(f"  Post-BD              : {int((v_bd == 2).sum())}")
    print(f"  Canal inferior       : {int((v_channel == 1).sum())}")
    print(f"  Canal superior       : {int((v_channel == 2).sum())}")
    print(f"  Inspiración          : {int((v_phase == 1).sum())}")
    print(f"  Espiración           : {int((v_phase == 2).sum())}")
    print(f"{'='*50}")


# ===========================================================================
# Figuras
# ===========================================================================

def _guardar(fig: plt.Figure, out_dir: str, filename: str) -> None:
    """Guarda la figura y libera memoria."""
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardada: {path}")


def _subject_colors(subjects: list[tuple]) -> list[str]:
    """Devuelve una lista de colores por sujeto: verde=BDR+, azul=BDR-(paciente), gris=control."""
    colors = []
    for _, _, _, _, bdr_label, subj_type in subjects:
        if subj_type == "control":
            colors.append("silver")
        elif bdr_label == "BDR+":
            colors.append("mediumseagreen")
        else:
            colors.append("steelblue")
    return colors


_LEGEND_PATCHES = [
    Patch(facecolor="mediumseagreen", label="BDR+"),
    Patch(facecolor="steelblue",      label="BDR- (paciente)"),
    Patch(facecolor="silver",         label="Control"),
]


def _fig1_segmentos_por_sujeto(
    v_subject: np.ndarray,
    subjects: list[tuple],
    out_dir: str,
) -> None:
    """Figura 1: barras de número de segmentos por sujeto, codificadas por grupo."""
    counts = np.array([np.sum(v_subject == s[0]) for s in subjects])
    colors = _subject_colors(subjects)
    x      = np.arange(1, len(subjects) + 1)
    labels = [s[1] for s in subjects]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(
        counts.mean(), color="black", linestyle="--", linewidth=1.0,
        label=f"Media ({counts.mean():.1f})"
    )
    ax.legend(handles=_LEGEND_PATCHES + [
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.0,
                   label=f"Media ({counts.mean():.1f})")
    ], fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Sujeto")
    ax.set_ylabel("Número de segmentos")
    ax.set_title("Segmentos por sujeto", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _guardar(fig, out_dir, "fig1_segmentos_por_sujeto.png")


def _fig2_distribucion_vectores(
    v_subject: np.ndarray,
    v_bd: np.ndarray,
    v_channel: np.ndarray,
    v_phase: np.ndarray,
    out_dir: str,
) -> None:
    """Figura 2: cuatro gráficos de tarta con la distribución de los vectores de metadatos."""
    n_patients = int(np.sum(v_subject <= N_PATIENTS))
    n_controls = int(np.sum(v_subject > N_PATIENTS))

    configs = [
        ((0, 0),
         [int(np.sum(v_bd == 1)), int(np.sum(v_bd == 2))],
         ["Pre-BD", "Post-BD"],
         ["#5599cc", "#cc7744"],
         "Pre-BD vs Post-BD"),
        ((0, 1),
         [int(np.sum(v_channel == 1)), int(np.sum(v_channel == 2))],
         ["Canal inferior", "Canal superior"],
         ["#66aacc", "#cc6688"],
         "Canal"),
        ((1, 0),
         [int(np.sum(v_phase == 1)), int(np.sum(v_phase == 2))],
         ["Inspiración", "Espiración"],
         ["#55bb77", "#ee9944"],
         "Fase respiratoria"),
        ((1, 1),
         [n_patients, n_controls],
         ["Pacientes", "Controles"],
         ["#7788dd", "#aaaaaa"],
         "Tipo de sujeto"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    for (r, c), sizes, labels, colors, title in configs:
        axes[r, c].pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 10},
        )
        axes[r, c].set_title(title, fontsize=11)

    fig.suptitle("Distribución de los vectores de metadatos", fontsize=13)
    fig.tight_layout()
    _guardar(fig, out_dir, "fig2_distribucion_vectores.png")


def _fig3_duracion_segmentos(
    v_subject: np.ndarray,
    v_phase: np.ndarray,
    segment_lengths: np.ndarray,
    subjects: list[tuple],
    out_dir: str,
) -> None:
    """Figura 3: duración media ± desviación estándar de inspiración y espiración por sujeto."""
    colors = _subject_colors(subjects)
    x      = np.arange(1, len(subjects) + 1)
    labels = [s[1] for s in subjects]

    mean_insp = np.zeros(len(subjects))
    std_insp  = np.zeros(len(subjects))
    mean_esp  = np.zeros(len(subjects))
    std_esp   = np.zeros(len(subjects))

    for idx, (subj_num, *_) in enumerate(subjects):
        dur_i = segment_lengths[(v_subject == subj_num) & (v_phase == 1)] / FS_TARGET * 1000.0
        dur_e = segment_lengths[(v_subject == subj_num) & (v_phase == 2)] / FS_TARGET * 1000.0
        if len(dur_i):
            mean_insp[idx], std_insp[idx] = dur_i.mean(), dur_i.std()
        if len(dur_e):
            mean_esp[idx], std_esp[idx] = dur_e.mean(), dur_e.std()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, means, stds, fase in [
        (axes[0], mean_insp, std_insp, "Inspiración"),
        (axes[1], mean_esp,  std_esp,  "Espiración"),
    ]:
        ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.5)
        ax.errorbar(x, means, yerr=stds, fmt="none",
                    color="black", capsize=3, linewidth=0.8)
        ax.set_ylabel("Duración media (ms)")
        ax.set_title(f"Duración media de segmentos de {fase}", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].legend(handles=_LEGEND_PATCHES, fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_xlabel("Sujeto")

    fig.suptitle("Duración media de segmentos por sujeto", fontsize=13)
    fig.tight_layout()
    _guardar(fig, out_dir, "fig3_duracion_segmentos.png")


def _fig4_heatmap_segmentos(
    counts_matrix: np.ndarray,
    subjects: list[tuple],
    out_dir: str,
) -> None:
    """
    Figura 4: mapa de calor (28 sujetos × 6 maniobras) con el número de segmentos
    por celda. Línea vertical entre maniobra 3 y 4 para separar pre-BD de post-BD.
    """
    labels = [s[1] for s in subjects]
    vmax   = counts_matrix.max()

    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(counts_matrix, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label="Número de segmentos")

    # Anotaciones numéricas en cada celda
    for i in range(counts_matrix.shape[0]):
        for j in range(counts_matrix.shape[1]):
            color = "white" if counts_matrix[i, j] > vmax * 0.65 else "black"
            ax.text(j, i, str(counts_matrix[i, j]),
                    ha="center", va="center", fontsize=7, color=color)

    # Separador pre-BD / post-BD
    ax.axvline(2.5, color="black", linestyle="--", linewidth=1.5)
    ax.text(0.5, -0.8, "Pre-BD", ha="center", fontsize=8, color="black",
            transform=ax.get_xaxis_transform())
    ax.text(3.5, -0.8, "Post-BD", ha="center", fontsize=8, color="black",
            transform=ax.get_xaxis_transform())

    ax.set_xticks(range(N_MANOEUVRES))
    ax.set_xticklabels([f"M{j + 1}" for j in range(N_MANOEUVRES)])
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Maniobra")
    ax.set_ylabel("Sujeto")
    ax.set_title("Segmentos por sujeto y maniobra", fontsize=13)

    fig.tight_layout()
    _guardar(fig, out_dir, "fig4_heatmap_segmentos.png")


# ===========================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_RES_DIR, exist_ok=True)

    print("Cargando metadatos...")
    metadata = _load_metadata(METADATA_CSV)
    subjects = _build_subject_list(metadata)

    print(f"\nProcesando {len(subjects)} sujetos...\n")
    all_signals, v_subject, v_bd, v_channel, v_phase, counts_matrix = build_dataset(subjects)

    _validate(all_signals, v_subject, v_bd, v_channel, v_phase, subjects)

    print("\nGuardando resultados...")
    segment_lengths = _save_outputs(
        all_signals, v_subject, v_bd, v_channel, v_phase, subjects, OUTPUT_RES_DIR
    )

    _print_summary(v_subject, v_bd, v_channel, v_phase, subjects)

    print("\nGenerando figuras...")
    _fig1_segmentos_por_sujeto(v_subject, subjects, OUTPUT_FIG_DIR)
    _fig2_distribucion_vectores(v_subject, v_bd, v_channel, v_phase, OUTPUT_FIG_DIR)
    _fig3_duracion_segmentos(v_subject, v_phase, segment_lengths, subjects, OUTPUT_FIG_DIR)
    _fig4_heatmap_segmentos(counts_matrix, subjects, OUTPUT_FIG_DIR)

    print("\nDataset completado.")
