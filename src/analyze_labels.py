"""
Inspección y análisis de proy_labels.mat.

Carga el archivo de etiquetas, lo cruza con los vectores de metadatos
del dataset (step4) y el CSV de metadatos de sujetos, y produce un
informe estructurado en consola. No escribe ningún archivo.

Uso:
    python src/analyze_labels.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

# ---------------------------------------------------------------------------
def _find_root() -> Path:
    """Busca la raíz del proyecto subiendo hasta encontrar proy_labels.mat."""
    candidate = Path(__file__).resolve().parent.parent
    for _ in range(6):
        if (candidate / "proy_labels.mat").exists():
            return candidate
        candidate = candidate.parent
    raise FileNotFoundError(
        "No se encontró proy_labels.mat en los directorios padre del script."
    )

ROOT = _find_root()

LABELS_FILE   = ROOT / "proy_labels.mat"
DATASET_NPZ   = ROOT / "outputs" / "results" / "step4" / "dataset.npz"
METADATA_CSV  = ROOT / "Data" / "database" / "subject_metadata.csv"
SUMMARY_CSV   = ROOT / "outputs" / "results" / "step4" / "dataset_summary.csv"

SEP  = "=" * 64
SEP2 = "-" * 64

LABEL_NAMES = {1: "1 (desconocida)", 2: "2 (CAS)", 3: "3 (NO CAS)", 6: "6 (desconocida)"}

# Mapeo subject_num → subject_id
SUBJ_ID = {
    **{i: f"P{i}" for i in range(1, 24)},
    24: "C1", 25: "C2", 26: "C3", 27: "C4", 28: "C5",
}

# ---------------------------------------------------------------------------


def _check(condition: bool, msg: str) -> None:
    tag = "[OK]" if condition else "[MISMATCH]"
    print(f"  {tag}  {msg}")


def main() -> None:
    # -----------------------------------------------------------------------
    # Carga de datos
    # -----------------------------------------------------------------------
    mat = scipy.io.loadmat(str(LABELS_FILE), squeeze_me=True)
    labels = np.asarray(mat["labels"]).ravel().astype(int)

    npz = np.load(str(DATASET_NPZ))
    v_subject = npz["v_subject"].ravel().astype(int)
    v_bd      = npz["v_bd"].ravel().astype(int)
    v_channel = npz["v_channel"].ravel().astype(int)
    v_phase   = npz["v_phase"].ravel().astype(int)

    meta = pd.read_csv(str(METADATA_CSV))

    # DataFrame de trabajo con todas las columnas
    df = pd.DataFrame({
        "label":     labels,
        "v_subject": v_subject,
        "v_bd":      v_bd,
        "v_channel": v_channel,
        "v_phase":   v_phase,
    })
    df = df.merge(
        meta[["subject_num", "subject_id", "type", "bdr_label"]],
        left_on="v_subject", right_on="subject_num", how="left",
    ).drop(columns="subject_num")

    # -----------------------------------------------------------------------
    print(SEP)
    print("STEP 1 — INSPECCIÓN DEL ARCHIVO proy_labels.mat")
    print(SEP)

    print(f"\nRuta : {LABELS_FILE}")
    print(f"Claves del archivo .mat : {[k for k in mat.keys() if not k.startswith('_')]}")
    print(f"Shape  : {np.asarray(mat['labels']).shape}")
    print(f"Dtype  : {np.asarray(mat['labels']).dtype}")
    print(f"Total de segmentos : {len(labels)}\n")

    counts = pd.Series(labels).value_counts().sort_index()
    print("Distribución de valores únicos:")
    print(f"  {'Etiqueta':>12}  {'Count':>8}  {'%':>7}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*7}")
    for val, cnt in counts.items():
        name = LABEL_NAMES.get(int(val), str(val))
        pct = 100.0 * cnt / len(labels)
        print(f"  {name:>12}  {cnt:>8}  {pct:>6.2f}%")
    print(f"  {'TOTAL':>12}  {len(labels):>8}  {100.00:>6.2f}%")

    # -----------------------------------------------------------------------
    print(f"\n{SEP}")
    print("STEP 2 — Q1: DISTRIBUCIÓN DE ETIQUETAS POR SUJETO")
    print(SEP)

    for lbl in sorted(counts.index):
        sub = df[df["label"] == lbl]
        grp = sub.groupby("v_subject").size().reset_index(name="count")
        grp["subject_id"] = grp["v_subject"].map(SUBJ_ID)
        name = LABEL_NAMES.get(int(lbl), str(lbl))
        print(f"\n  Etiqueta {name}  ({len(sub)} segmentos, {grp.shape[0]} sujetos)")
        rows = [f"{row.subject_id}:{row['count']}" for _, row in grp.iterrows()]
        print("    " + "  ".join(rows))

    # -----------------------------------------------------------------------
    print(f"\n{SEP}")
    print("STEP 2 — Q2: CAS / NO CAS POR GRUPO CLÍNICO")
    print(SEP)

    cas_df = df[df["label"].isin([2, 3])].copy()

    def _bdr_group(row: pd.Series) -> str:
        if row["type"] == "control":
            return "Control"
        return "BDR+" if row["bdr_label"] == "BDR+" else "BDR-"

    cas_df["group"] = cas_df.apply(_bdr_group, axis=1)
    tbl = cas_df.groupby(["group", "label"]).size().unstack(fill_value=0)
    tbl.columns = [LABEL_NAMES.get(int(c), str(c)) for c in tbl.columns]
    tbl["Total"] = tbl.sum(axis=1)
    print(f"\n{tbl.to_string()}")

    # -----------------------------------------------------------------------
    print(f"\n{SEP}")
    print("STEP 2 — Q3: CAS / NO CAS POR FASE, CANAL Y CONDICIÓN BD")
    print(SEP)

    for dim, col, mapping in [
        ("Fase respiratoria", "v_phase",   {1: "Inspiración", 2: "Espiración"}),
        ("Canal",             "v_channel", {1: "Inferior (CH1)", 2: "Superior (CH2)"}),
        ("Condición BD",      "v_bd",      {1: "Pre-BD", 2: "Post-BD"}),
    ]:
        sub = cas_df.copy()
        sub[col] = sub[col].map(mapping)
        tbl2 = sub.groupby([col, "label"]).size().unstack(fill_value=0)
        tbl2.columns = [LABEL_NAMES.get(int(c), str(c)) for c in tbl2.columns]
        tbl2["Total"] = tbl2.sum(axis=1)
        print(f"\n  {dim}:")
        print(tbl2.to_string(index=True))

    # -----------------------------------------------------------------------
    print(f"\n{SEP}")
    print("STEP 2 — Q4: SUJETOS CON ETIQUETAS 2 O 3")
    print(SEP)

    labeled_df = df[df["label"].isin([2, 3])]
    labeled_subjs = sorted(labeled_df["v_subject"].unique())
    n_labeled_subjs = len(labeled_subjs)
    subj_ids = [SUBJ_ID[s] for s in labeled_subjs]

    print(f"\n  Número de sujetos con etiqueta 2 o 3 : {n_labeled_subjs}")
    print(f"  Sujetos : {', '.join(subj_ids)}")

    # Detalle por sujeto
    print(f"\n  {'Sujeto':>8}  {'BDR':>5}  {'Tipo':>8}  {'CAS(2)':>8}  {'NOCAS(3)':>10}  Total")
    print(f"  {'-'*8}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*6}")
    for s in labeled_subjs:
        sub = labeled_df[labeled_df["v_subject"] == s]
        cas  = (sub["label"] == 2).sum()
        noca = (sub["label"] == 3).sum()
        sid  = SUBJ_ID[s]
        bdr  = sub["bdr_label"].iloc[0]
        typ  = sub["type"].iloc[0]
        print(f"  {sid:>8}  {bdr:>5}  {typ:>8}  {cas:>8}  {noca:>10}  {len(sub):>6}")

    # -----------------------------------------------------------------------
    print(f"\n{SEP}")
    print("STEP 2 — Q5: PATRÓN DE ETIQUETAS 1 Y 6")
    print(SEP)

    for lbl in [1, 6]:
        sub = df[df["label"] == lbl].copy()
        name = LABEL_NAMES.get(lbl, str(lbl))
        print(f"\n  {'-'*48}")
        print(f"  Etiqueta {name}  ({len(sub)} segmentos)")
        print(f"  {'-'*48}")

        # Por tipo
        grp_type = sub.groupby("type").size()
        print(f"\n  Por tipo de sujeto:")
        for k, v in grp_type.items():
            print(f"    {k:>10} : {v:>6}  ({100*v/len(sub):.1f}%)")

        # Por BDR
        grp_bdr = sub.groupby("bdr_label").size()
        print(f"\n  Por grupo BDR:")
        for k, v in grp_bdr.items():
            print(f"    {k:>6} : {v:>6}  ({100*v/len(sub):.1f}%)")

        # Por condición BD
        grp_bd = sub.groupby("v_bd").size()
        bd_names = {1: "Pre-BD", 2: "Post-BD"}
        print(f"\n  Por condición broncodilatadora:")
        for k, v in grp_bd.items():
            print(f"    {bd_names.get(k, k):>8} : {v:>6}  ({100*v/len(sub):.1f}%)")

        # Por canal
        grp_ch = sub.groupby("v_channel").size()
        ch_names = {1: "Inferior(CH1)", 2: "Superior(CH2)"}
        print(f"\n  Por canal:")
        for k, v in grp_ch.items():
            print(f"    {ch_names.get(k, k):>14} : {v:>6}  ({100*v/len(sub):.1f}%)")

        # Por fase
        grp_ph = sub.groupby("v_phase").size()
        ph_names = {1: "Inspiración", 2: "Espiración"}
        print(f"\n  Por fase respiratoria:")
        for k, v in grp_ph.items():
            print(f"    {ph_names.get(k, k):>12} : {v:>6}  ({100*v/len(sub):.1f}%)")

        # Top sujetos
        grp_subj = sub.groupby("v_subject").size().sort_values(ascending=False)
        print(f"\n  Top sujetos (todos los {grp_subj.shape[0]} con esta etiqueta):")
        for s, cnt in grp_subj.items():
            print(f"    {SUBJ_ID[int(s)]:>4} : {cnt:>6}  ({100*cnt/len(sub):.1f}%)")

    # -----------------------------------------------------------------------
    print(f"\n{SEP}")
    print("STEP 3 — VERIFICACIÓN DE CONSISTENCIA")
    print(SEP)
    print()

    total = len(labels)
    _check(total == 14900, f"len(labels) == 14900  (obtenido: {total})")

    n_labeled = int(np.isin(labels, [2, 3]).sum())
    _check(n_labeled == 1923, f"labels in {{2,3}} == 1923  (obtenido: {n_labeled})")

    _check(n_labeled_subjs == 18,
           f"sujetos con etiqueta 2 o 3 == 18  (obtenido: {n_labeled_subjs})")

    total_check = sum(counts)
    _check(total_check == 14900,
           f"suma de todos los counts == 14900  (obtenido: {total_check})")

    print()
    print(SEP)
    print("FIN DEL INFORME")
    print(SEP)


if __name__ == "__main__":
    main()
