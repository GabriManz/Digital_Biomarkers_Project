"""
Clasificación CAS con Deep Learning — espectrogramas como entrada.

Implementa dos enfoques de aprendizaje profundo y los compara contra el
baseline SVM/RF de step6:
  - PART 0: Reconstrucción y caché de las 1923 señales etiquetadas.
  - PART 1: Generación de espectrogramas (64×64 float32).
  - PART 2: CNN pequeña entrenada desde cero con validación LOSO.
  - PART 3: Embeddings VGGish preentrenados + SVM con validación LOSO.
  - PART 4: Figuras comparativas.
  - PART 5: Tabla de comparación y exportación JSON.

Uso:
    python src/step8_deep_learning.py
"""

from __future__ import annotations

import copy
import csv as _csv
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import scipy.ndimage
import scipy.signal
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Permite importar módulos vecinos desde src/
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from step4_dataset import N_CONTROLS, N_PATIENTS, build_dataset

# ---------------------------------------------------------------------------
# Importaciones opcionales: PyTorch y VGGish
# ---------------------------------------------------------------------------
TORCH_AVAILABLE: bool = False
VGGISH_AVAILABLE: bool = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    print("WARNING: PyTorch no instalado. La parte CNN se omitirá.")
    print("         Ejecuta: pip install torch --break-system-packages")

try:
    import torchvggish
    VGGISH_AVAILABLE = True
except ImportError:
    print("WARNING: torchvggish no disponible. La parte VGGish se omitirá.")
    print("         Ejecuta: pip install torchvggish --break-system-packages")


# ===========================================================================
# Localización de la raíz del proyecto
# ===========================================================================

def _find_project_root() -> Path:
    """
    Busca la raíz del proyecto subiendo en la jerarquía de directorios
    hasta encontrar proy_labels.mat.
    """
    candidate = Path(__file__).resolve().parent.parent
    for _ in range(6):
        if (candidate / "proy_labels.mat").exists():
            return candidate
        candidate = candidate.parent
    return Path(__file__).resolve().parent.parent


_PROJECT_ROOT = _find_project_root()

# ===========================================================================
# Constantes globales
# ===========================================================================

FS_TARGET         = 4000
SPECTROGRAM_SHAPE = (64, 64)
CNN_EPOCHS        = 50
CNN_BATCH_SIZE    = 32
CNN_LR            = 1e-3
PATIENCE          = 10
RANDOM_STATE      = 42
MIN_CAS_TRAIN     = 5      # igual que step6; excluye sujeto 8
PRETRAIN_EPOCHS   = 30
PRETRAIN_LR       = 1e-3

LABEL_CAS    = 2
LABEL_NO_CAS = 3

RESULTS_DIR = _PROJECT_ROOT / "outputs" / "results" / "step8"
FIGURES_DIR = _PROJECT_ROOT / "outputs" / "figures" / "step8"

_LABELS_FILE   = _PROJECT_ROOT / "proy_labels.mat"
_METADATA_CSV  = _PROJECT_ROOT / "Data" / "database" / "subject_metadata.csv"
_DATA_DIR      = _PROJECT_ROOT / "Data"

_STEP6_SVM_CSV = _PROJECT_ROOT / "outputs" / "results" / "step6" / "svm_loso_results.csv"
_STEP6_RF_CSV  = _PROJECT_ROOT / "outputs" / "results" / "step6" / "rf_loso_results.csv"


# ===========================================================================
# Helpers locales para lista de sujetos (misma lógica que step5)
# ===========================================================================

def _load_metadata_local(csv_path: Path) -> dict[str, str]:
    """Carga el CSV de metadatos y devuelve {subject_id: bdr_label}."""
    metadata: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            metadata[row["subject_id"]] = row["bdr_label"]
    return metadata


def _build_subjects() -> list[tuple]:
    """
    Construye la lista de los 28 sujetos con paths anclados a _PROJECT_ROOT.
    Retorna lista de (subj_num, subj_id, sig_file, mkr_file, bdr_label, type).
    """
    metadata = _load_metadata_local(_METADATA_CSV)
    subjects: list[tuple] = []
    for i in range(1, N_PATIENTS + 1):
        sid = f"P{i}"
        subjects.append((
            i, sid,
            str(_DATA_DIR / f"{sid}.mat"),
            str(_DATA_DIR / f"t{sid}.mat"),
            metadata[sid], "patient",
        ))
    for i in range(1, N_CONTROLS + 1):
        sid = f"C{i}"
        subjects.append((
            N_PATIENTS + i, sid,
            str(_DATA_DIR / f"{sid}.mat"),
            str(_DATA_DIR / f"t{sid}.mat"),
            metadata[sid], "control",
        ))
    return subjects


# ===========================================================================
# PART 0 — Reconstrucción de señales etiquetadas
# ===========================================================================

def load_labeled_signals(
    force_rebuild: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Reconstruye las 1923 señales preprocesadas y segmentadas que tienen
    etiqueta CAS (2) o NO_CAS (3) en proy_labels.mat.

    Retorna
    -------
    signals_labeled : list[np.ndarray]
        1923 señales 1D a FS_TARGET=4000 Hz (longitud variable).
    y_labeled : np.ndarray shape (1923,)
        Etiquetas binarias: 1=CAS, 0=NO_CAS.
    groups : np.ndarray shape (1923,)
        IDs de sujeto para validación LOSO.
    """
    cache_path = RESULTS_DIR / "labeled_signals.pkl"

    if cache_path.exists() and not force_rebuild:
        print("  [caché] Cargando señales etiquetadas desde disco...")
        with open(cache_path, "rb") as fh:
            data = pickle.load(fh)
        return data["signals"], data["y"], data["groups"]

    print("  Reconstruyendo 14 900 señales (build_dataset)...")
    t0 = time.time()
    subjects = _build_subjects()
    all_signals, v_subject, *_ = build_dataset(subjects)
    print(f"  build_dataset completado en {(time.time()-t0)/60:.1f} min.")

    print("  Cargando etiquetas desde proy_labels.mat...")
    mat    = scipy.io.loadmat(str(_LABELS_FILE), squeeze_me=True)
    labels = np.asarray(mat["labels"]).ravel().astype(int)

    mask   = (labels == LABEL_CAS) | (labels == LABEL_NO_CAS)
    assert mask.sum() == 1923, f"Se esperaban 1923 señales etiquetadas, hay {mask.sum()}"

    idx_labeled      = np.where(mask)[0]
    signals_labeled  = [all_signals[i] for i in idx_labeled]
    y_labeled        = (labels[mask] == LABEL_CAS).astype(int)
    groups           = v_subject[mask]

    print(f"  → CAS: {y_labeled.sum()}, NO_CAS: {(1-y_labeled).sum()}, total: {len(y_labeled)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump({"signals": signals_labeled, "y": y_labeled, "groups": groups}, fh, protocol=4)
    print(f"  Señales guardadas en {cache_path}")

    return signals_labeled, y_labeled, groups


# ===========================================================================
# PART 1 — Generación de espectrogramas
# ===========================================================================

def signal_to_spectrogram(
    signal: np.ndarray,
    fs: int = FS_TARGET,
    target_shape: tuple[int, int] = SPECTROGRAM_SHAPE,
) -> np.ndarray:
    """
    Convierte una señal 1D en un espectrograma normalizado de tamaño fijo.

    Pasos:
      1. Calcula espectrograma con scipy.signal.spectrogram.
      2. Convierte a dB: 10·log10(Sxx + 1e-10).
      3. Restringe a 70–2000 Hz.
      4. Redimensiona a target_shape con scipy.ndimage.zoom.
      5. Normaliza a [0, 1].

    Retorna
    -------
    np.ndarray float32 de forma (64, 64) con valores en [0, 1].
    """
    sig = np.asarray(signal, dtype=np.float64)
    if len(sig) < 16:
        sig = np.pad(sig, (0, 16 - len(sig)))

    nperseg = max(4, min(128, len(sig) // 4))
    f, _t, Sxx = scipy.signal.spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=None)

    Sxx_db   = 10.0 * np.log10(Sxx + 1e-10)
    freq_msk = (f >= 70) & (f <= 2000)
    Sxx_db   = Sxx_db[freq_msk, :]

    # Redimensionar a target_shape
    rows, cols = Sxx_db.shape
    zoom_r = target_shape[0] / max(rows, 1)
    zoom_c = target_shape[1] / max(cols, 1)
    S = scipy.ndimage.zoom(Sxx_db, (zoom_r, zoom_c), order=1)
    # Recortar si zoom produce un píxel extra
    S = S[: target_shape[0], : target_shape[1]]
    # Rellenar si queda corto (raro)
    if S.shape != target_shape:
        padded = np.full(target_shape, S.min(), dtype=np.float64)
        padded[: S.shape[0], : S.shape[1]] = S
        S = padded

    vmin, vmax = S.min(), S.max()
    S = (S - vmin) / (vmax - vmin + 1e-10)
    return S.astype(np.float32)


def build_spectrograms(
    signals: list[np.ndarray],
    force_rebuild: bool = False,
) -> np.ndarray:
    """
    Genera la matriz de espectrogramas para todas las señales etiquetadas.

    Retorna
    -------
    X_spectro : np.ndarray shape (1923, 64, 64) float32
    """
    cache_path = RESULTS_DIR / "X_spectrograms.npy"

    if cache_path.exists() and not force_rebuild:
        print("  [caché] Cargando espectrogramas desde disco...")
        return np.load(str(cache_path))

    n = len(signals)
    H, W = SPECTROGRAM_SHAPE
    X_spectro = np.empty((n, H, W), dtype=np.float32)

    print(f"  Generando {n} espectrogramas...")
    t0 = time.time()
    for i, sig in enumerate(signals):
        X_spectro[i] = signal_to_spectrogram(sig)
        if (i + 1) % 200 == 0 or (i + 1) == n:
            print(f"    {i+1}/{n} ({(time.time()-t0):.1f}s)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), X_spectro)
    print(f"  Espectrogramas guardados en {cache_path}")
    return X_spectro


# ===========================================================================
# PART 2b — Preentrenamiento semi-supervisado con autoencoder
# ===========================================================================

def build_all_spectrograms(
    signals_all: list,
    force_rebuild: bool = False,
) -> np.ndarray:
    """
    Genera la matriz de espectrogramas para las 14 900 señales (etiquetadas
    + no etiquetadas). Reutiliza signal_to_spectrogram().

    Retorna
    -------
    np.ndarray shape (14900, 64, 64) float32
    """
    cache_path = RESULTS_DIR / "X_spectrograms_all14900.npy"

    if cache_path.exists() and not force_rebuild:
        print("  [caché] Cargando espectrogramas (14 900) desde disco...")
        return np.load(str(cache_path))

    n = len(signals_all)
    H, W = SPECTROGRAM_SHAPE
    X_all = np.empty((n, H, W), dtype=np.float32)

    print(f"  Generando {n} espectrogramas (todas las señales)...")
    t0 = time.time()
    for i, sig in enumerate(signals_all):
        X_all[i] = signal_to_spectrogram(sig)
        if (i + 1) % 200 == 0 or (i + 1) == n:
            print(f"    {i+1}/{n} ({time.time()-t0:.1f}s)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), X_all)
    print(f"  Espectrogramas (14 900) guardados en {cache_path}")
    return X_all


def pretrain_autoencoder(
    X_all: np.ndarray,
    pretrain_epochs: int = PRETRAIN_EPOCHS,
    pretrain_lr: float = PRETRAIN_LR,
    force_retrain: bool = False,
) -> "CAS_Autoencoder":
    """
    Entrena el autoencoder sobre las 14 900 señales sin usar etiquetas.
    Guarda los pesos en autoencoder_weights.pt.
    Si el archivo existe y force_retrain=False, carga y devuelve directamente.

    Retorna
    -------
    CAS_Autoencoder con pesos entrenados.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch no disponible.")

    cache_path = RESULTS_DIR / "autoencoder_weights.pt"
    ae = CAS_Autoencoder()

    if cache_path.exists() and not force_retrain:
        print("  [caché] Cargando pesos del autoencoder desde disco...")
        ae.load_state_dict(torch.load(str(cache_path), weights_only=True))
        ae.eval()
        return ae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Dispositivo preentrenamiento: {device}")
    ae = ae.to(device)

    X_tensor = torch.tensor(X_all, dtype=torch.float32).unsqueeze(1)  # (N,1,64,64)
    ds = torch.utils.data.TensorDataset(X_tensor)
    dl = torch.utils.data.DataLoader(ds, batch_size=CNN_BATCH_SIZE, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=pretrain_lr)

    t0 = time.time()
    for epoch in range(1, pretrain_epochs + 1):
        ae.train()
        epoch_loss = 0.0
        for (Xb,) in dl:
            Xb = Xb.to(device)
            optimizer.zero_grad()
            recon, _ = ae(Xb)
            loss = criterion(recon, Xb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(Xb)
        epoch_loss /= max(len(X_all), 1)
        if epoch % 5 == 0 or epoch == pretrain_epochs:
            print(f"  [Pretrain] Época {epoch}/{pretrain_epochs} — "
                  f"Loss reconstrucción: {epoch_loss:.4f}")

    elapsed = (time.time() - t0) / 60
    print(f"Preentrenamiento completado en {elapsed:.1f} minutos.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ae.state_dict(), str(cache_path))
    print(f"  Pesos guardados en {cache_path}")

    ae.eval()
    return ae


def run_pretrained_cnn_loso(
    X_spectro: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    autoencoder: "CAS_Autoencoder",
) -> dict:
    """
    Igual que run_cnn_loso() pero instancia CAS_CNN_Pretrained con el encoder
    preentrenado y usa lr diferenciado (encoder ×0.1) en el optimizador Adam.

    Retorna
    -------
    dict con claves: per_fold, mean, std, training_curves.
    """
    if not TORCH_AVAILABLE:
        return {}

    cas_per_subj = {s: int((y[groups == s] == 1).sum()) for s in np.unique(groups)}
    valid_subjs  = [s for s, n in cas_per_subj.items() if n >= MIN_CAS_TRAIN]
    mask_valid   = np.isin(groups, valid_subjs)

    X_loso   = X_spectro[mask_valid]
    y_loso   = y[mask_valid]
    grp_loso = groups[mask_valid]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Dispositivo: {device}")

    loso    = LeaveOneGroupOut()
    n_folds = loso.get_n_splits(X_loso, y_loso, grp_loso)
    per_fold: list[dict] = []

    last_train_loss: list[float] = []
    last_val_auc:    list[float] = []
    last_stop_epoch: int         = 0
    last_subj_label: str         = ""

    t_start = time.time()

    for fold_i, (train_idx, test_idx) in enumerate(loso.split(X_loso, y_loso, grp_loso)):
        subj_num   = int(grp_loso[test_idx[0]])
        subj_label = f"P{subj_num}" if subj_num <= N_PATIENTS else f"C{subj_num - N_PATIENTS}"

        X_tr_all = X_loso[train_idx]
        y_tr_all = y_loso[train_idx]
        X_te     = X_loso[test_idx]
        y_te     = y_loso[test_idx]

        if y_tr_all.sum() >= 2 and (1 - y_tr_all).sum() >= 2:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
            tr_idx_inner, val_idx_inner = next(sss.split(X_tr_all, y_tr_all))
        else:
            tr_idx_inner  = np.arange(len(y_tr_all))
            val_idx_inner = np.array([], dtype=int)

        X_tr  = X_tr_all[tr_idx_inner]
        y_tr  = y_tr_all[tr_idx_inner]
        X_val = X_tr_all[val_idx_inner]
        y_val = y_tr_all[val_idx_inner]

        use_val = len(y_val) > 0 and y_val.sum() > 0

        ds_tr = TensorDataset(
            torch.tensor(X_tr,  dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_tr,  dtype=torch.float32),
        )
        dl_tr = DataLoader(ds_tr, batch_size=CNN_BATCH_SIZE, shuffle=True)

        if use_val:
            ds_val = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
                torch.tensor(y_val, dtype=torch.float32),
            )
            dl_val = DataLoader(ds_val, batch_size=CNN_BATCH_SIZE, shuffle=False)

        n_pos = max(int(y_tr.sum()), 1)
        n_neg = max(len(y_tr) - n_pos, 1)
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

        model = CAS_CNN_Pretrained(autoencoder).to(device)
        # lr reducido ×10 para el encoder preentrenado, lr completo para el clasificador
        optimizer = optim.Adam([
            {"params": model.features.parameters(),   "lr": CNN_LR / 10},
            {"params": model.pool.parameters(),        "lr": CNN_LR / 10},
            {"params": model.classifier.parameters(),  "lr": CNN_LR},
        ])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_auc     = -1.0
        patience_counter = 0
        best_weights     = copy.deepcopy(model.state_dict())
        stop_epoch       = CNN_EPOCHS

        fold_train_loss: list[float] = []
        fold_val_auc:    list[float] = []
        is_last = (fold_i == n_folds - 1)

        for epoch in range(CNN_EPOCHS):
            model.train()
            epoch_loss = 0.0
            for Xb, yb in dl_tr:
                Xb, yb = Xb.to(device), yb.to(device)
                Xb = spec_augment(Xb)
                optimizer.zero_grad()
                logits = model(Xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(yb)
            epoch_loss /= max(len(y_tr), 1)
            if is_last:
                fold_train_loss.append(epoch_loss)

            if use_val:
                model.eval()
                val_probs_list: list[float] = []
                with torch.no_grad():
                    for Xb, _ in dl_val:
                        logits = model(Xb.to(device))
                        probs  = torch.sigmoid(logits).cpu().numpy()
                        val_probs_list.extend(probs.tolist())
                val_probs = np.array(val_probs_list)
                try:
                    val_auc = float(roc_auc_score(y_val, val_probs))
                except ValueError:
                    val_auc = 0.0
                if is_last:
                    fold_val_auc.append(val_auc)

                if val_auc > best_val_auc:
                    best_val_auc     = val_auc
                    patience_counter = 0
                    best_weights     = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        stop_epoch = epoch + 1
                        break
            else:
                best_weights = copy.deepcopy(model.state_dict())
                if is_last:
                    fold_val_auc.append(0.0)

        model.load_state_dict(best_weights)
        model.eval()

        X_te_t = torch.tensor(X_te, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            logits_te = model(X_te_t)
            probs_te  = torch.sigmoid(logits_te).cpu().numpy()
        preds_te = (probs_te >= 0.5).astype(int)

        metrics = _compute_metrics(y_te, preds_te, probs_te)
        metrics["fold"]       = fold_i + 1
        metrics["subject_id"] = subj_label
        per_fold.append(metrics)

        print(f"  Fold {fold_i+1}/{n_folds} — {subj_label} — AUC: {metrics['auc']:.3f}")

        if is_last:
            last_train_loss = fold_train_loss
            last_val_auc    = fold_val_auc
            last_stop_epoch = stop_epoch
            last_subj_label = subj_label

    elapsed = time.time() - t_start
    print(f"CNN+Pretrain LOSO completado en {elapsed/60:.1f} minutos.")

    mean, std = _aggregate_folds(per_fold)
    return {
        "per_fold": per_fold,
        "mean":     mean,
        "std":      std,
        "training_curves": {
            "train_loss":        last_train_loss,
            "val_auc":           last_val_auc,
            "early_stop_epoch":  last_stop_epoch,
            "last_fold_subject": last_subj_label,
        },
    }


# ===========================================================================
# PART 2 — CNN desde cero con validación LOSO
# ===========================================================================

if TORCH_AVAILABLE:
    class CAS_CNN(nn.Module):
        """
        CNN ligera para clasificación binaria CAS/NO_CAS sobre espectrogramas
        (batch, 1, 64, 64). Versión reducida (~35K parámetros vs 561K anterior)
        con AdaptiveAvgPool para mejor regularización con datasets pequeños.
        """

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1,  8,  kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.MaxPool2d(2),          # → (8, 32, 32)

                nn.Conv2d(8,  16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),          # → (16, 16, 16)

                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),          # → (32, 8, 8)
            )
            # Pooling adaptativo: reduce a (32, 4, 4) = 512 features
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(32 * 4 * 4, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                # Sin Sigmoid — se usa BCEWithLogitsLoss
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.features(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x).squeeze(1)

    class CAS_Autoencoder(nn.Module):
        """
        Autoencoder convolucional simétrico a CAS_CNN.
        Se entrena con MSELoss sobre espectrogramas sin etiquetas.
        forward() devuelve (reconstrucción, embedding_bottleneck).
        """

        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1,  8,  kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.MaxPool2d(2),           # → (8, 32, 32)

                nn.Conv2d(8,  16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),           # → (16, 16, 16)

                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),           # → (32, 8, 8)
            )
            self.pool = nn.AdaptiveAvgPool2d((4, 4))   # bottleneck → (32, 4, 4)

            self.decoder = nn.Sequential(
                # (32, 4, 4) → (16, 8, 8)
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                # (16, 8, 8) → (8, 16, 16)
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                # (8, 16, 16) → (1, 64, 64): stride=4 para 4× upsampling
                # output = (16-1)*4 + 4 = 64 ✓
                nn.ConvTranspose2d(8, 1, kernel_size=4, stride=4, padding=0),
                nn.Sigmoid(),
            )

        def forward(
            self, x: "torch.Tensor"
        ) -> "tuple[torch.Tensor, torch.Tensor]":
            z = self.encoder(x)
            z = self.pool(z)
            embedding = z.view(z.size(0), -1)   # (B, 512)
            recon = self.decoder(z)              # (B, 1, 64, 64)
            return recon, embedding

    class CAS_CNN_Pretrained(nn.Module):
        """
        Igual que CAS_CNN pero inicializa features y pool con los pesos
        del encoder del autoencoder preentrenado.
        """

        def __init__(self, autoencoder: "CAS_Autoencoder") -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1,  8,  kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(8,  16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(32 * 4 * 4, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )
            # Copiar pesos del encoder preentrenado
            self.features.load_state_dict(autoencoder.encoder.state_dict())
            self.pool.load_state_dict(autoencoder.pool.state_dict())

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.features(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x).squeeze(1)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calcula accuracy, sensitivity, specificity, precision, f1 y AUC."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = f1_score(y_true, y_pred, zero_division=0)
    accuracy    = (tp + tn) / len(y_true)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.0
    return {
        "accuracy":    float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision":   float(precision),
        "f1":          float(f1),
        "auc":         auc,
    }


def _aggregate_folds(per_fold: list[dict]) -> tuple[dict, dict]:
    """Calcula media y desviación estándar de las métricas por fold."""
    keys = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]
    mean = {k: float(np.mean([f[k] for f in per_fold])) for k in keys}
    std  = {k: float(np.std([f[k] for f in per_fold]))  for k in keys}
    return mean, std


def spec_augment(
    x: "torch.Tensor",
    freq_mask_max: int = 10,
    time_mask_max: int = 10,
    n_freq_masks: int = 1,
    n_time_masks: int = 1,
) -> "torch.Tensor":
    """
    Aplica SpecAugment a un batch de espectrogramas (B, 1, F, T).

    Enmascara aleatoriamente bandas de frecuencia y franjas de tiempo
    poniendo a cero la región, forzando al modelo a no depender de
    zonas concretas del espectrograma. Solo se llama durante el
    entrenamiento, nunca en validación ni test.
    """
    import random as _random
    B, _, F, T = x.shape
    x = x.clone()
    for i in range(B):
        for _ in range(n_freq_masks):
            f = _random.randint(0, freq_mask_max)
            if f > 0:
                f0 = _random.randint(0, max(F - f, 0))
                x[i, :, f0:f0 + f, :] = 0.0
        for _ in range(n_time_masks):
            t = _random.randint(0, time_mask_max)
            if t > 0:
                t0 = _random.randint(0, max(T - t, 0))
                x[i, :, :, t0:t0 + t] = 0.0
    return x


def run_cnn_loso(
    X_spectro: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """
    Valida la CNN con Leave-One-Subject-Out cross-validation.

    Parámetros
    ----------
    X_spectro : np.ndarray (1923, 64, 64)
    y         : np.ndarray (1923,) binario
    groups    : np.ndarray (1923,) IDs de sujeto

    Retorna
    -------
    dict con claves: per_fold, mean, std, training_curves.
    """
    if not TORCH_AVAILABLE:
        return {}

    # Filtro de sujetos válidos (mismo que step6)
    cas_per_subj = {s: int((y[groups == s] == 1).sum()) for s in np.unique(groups)}
    valid_subjs  = [s for s, n in cas_per_subj.items() if n >= MIN_CAS_TRAIN]
    mask_valid   = np.isin(groups, valid_subjs)

    X_loso = X_spectro[mask_valid]
    y_loso = y[mask_valid]
    grp_loso = groups[mask_valid]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Dispositivo: {device}")

    loso     = LeaveOneGroupOut()
    n_folds  = loso.get_n_splits(X_loso, y_loso, grp_loso)
    per_fold: list[dict] = []

    # Curvas de entrenamiento del último fold
    last_train_loss: list[float] = []
    last_val_auc:    list[float] = []
    last_stop_epoch: int         = 0
    last_subj_label: str         = ""

    t_start = time.time()

    for fold_i, (train_idx, test_idx) in enumerate(loso.split(X_loso, y_loso, grp_loso)):
        subj_num   = int(grp_loso[test_idx[0]])
        subj_label = f"P{subj_num}" if subj_num <= N_PATIENTS else f"C{subj_num - N_PATIENTS}"

        X_tr_all = X_loso[train_idx]
        y_tr_all = y_loso[train_idx]
        X_te     = X_loso[test_idx]
        y_te     = y_loso[test_idx]

        # División 90/10 para early stopping
        if y_tr_all.sum() >= 2 and (1 - y_tr_all).sum() >= 2:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
            tr_idx_inner, val_idx_inner = next(sss.split(X_tr_all, y_tr_all))
        else:
            # Sin suficientes clases para estratificar → sin validación
            tr_idx_inner  = np.arange(len(y_tr_all))
            val_idx_inner = np.array([], dtype=int)

        X_tr  = X_tr_all[tr_idx_inner]
        y_tr  = y_tr_all[tr_idx_inner]
        X_val = X_tr_all[val_idx_inner]
        y_val = y_tr_all[val_idx_inner]

        use_val = len(y_val) > 0 and y_val.sum() > 0

        # DataLoaders
        ds_tr  = TensorDataset(
            torch.tensor(X_tr,  dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_tr,  dtype=torch.float32),
        )
        dl_tr  = DataLoader(ds_tr, batch_size=CNN_BATCH_SIZE, shuffle=True)

        if use_val:
            ds_val = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
                torch.tensor(y_val, dtype=torch.float32),
            )
            dl_val = DataLoader(ds_val, batch_size=CNN_BATCH_SIZE, shuffle=False)

        # Peso para clase positiva (CAS)
        n_pos = max(int(y_tr.sum()), 1)
        n_neg = max(len(y_tr) - n_pos, 1)
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

        model     = CAS_CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=CNN_LR)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_auc     = -1.0
        patience_counter = 0
        best_weights     = copy.deepcopy(model.state_dict())
        stop_epoch       = CNN_EPOCHS

        fold_train_loss: list[float] = []
        fold_val_auc:    list[float] = []
        is_last = (fold_i == n_folds - 1)

        for epoch in range(CNN_EPOCHS):
            # --- Entrenamiento ---
            model.train()
            epoch_loss = 0.0
            for Xb, yb in dl_tr:
                Xb, yb = Xb.to(device), yb.to(device)
                Xb = spec_augment(Xb)      # SpecAugment solo en train
                optimizer.zero_grad()
                logits = model(Xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(yb)
            epoch_loss /= max(len(y_tr), 1)
            if is_last:
                fold_train_loss.append(epoch_loss)

            # --- Validación ---
            if use_val:
                model.eval()
                val_probs_list: list[float] = []
                with torch.no_grad():
                    for Xb, _ in dl_val:
                        logits = model(Xb.to(device))
                        probs  = torch.sigmoid(logits).cpu().numpy()
                        val_probs_list.extend(probs.tolist())
                val_probs = np.array(val_probs_list)
                try:
                    val_auc = float(roc_auc_score(y_val, val_probs))
                except ValueError:
                    val_auc = 0.0
                if is_last:
                    fold_val_auc.append(val_auc)

                if val_auc > best_val_auc:
                    best_val_auc     = val_auc
                    patience_counter = 0
                    best_weights     = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        stop_epoch = epoch + 1
                        break
            else:
                # Sin validación: guardar siempre
                best_weights = copy.deepcopy(model.state_dict())
                if is_last:
                    fold_val_auc.append(0.0)

        # Cargar mejores pesos
        model.load_state_dict(best_weights)
        model.eval()

        # Predicción en test
        X_te_t = torch.tensor(X_te, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            logits_te = model(X_te_t)
            probs_te  = torch.sigmoid(logits_te).cpu().numpy()
        preds_te = (probs_te >= 0.5).astype(int)

        metrics = _compute_metrics(y_te, preds_te, probs_te)
        metrics["fold"]       = fold_i + 1
        metrics["subject_id"] = subj_label
        per_fold.append(metrics)

        print(f"  Fold {fold_i+1}/{n_folds} — {subj_label} — AUC: {metrics['auc']:.3f}")

        if is_last:
            last_train_loss = fold_train_loss
            last_val_auc    = fold_val_auc
            last_stop_epoch = stop_epoch
            last_subj_label = subj_label

    elapsed = time.time() - t_start
    print(f"CNN LOSO completado en {elapsed/60:.1f} minutos.")

    mean, std = _aggregate_folds(per_fold)
    return {
        "per_fold": per_fold,
        "mean":     mean,
        "std":      std,
        "training_curves": {
            "train_loss":        last_train_loss,
            "val_auc":           last_val_auc,
            "early_stop_epoch":  last_stop_epoch,
            "last_fold_subject": last_subj_label,
        },
    }


# ===========================================================================
# PART 3 — VGGish + SVM
# ===========================================================================

def extract_vggish_embedding(
    signal: np.ndarray,
    vggish_model: object,
    fs_in: int = FS_TARGET,
) -> np.ndarray:
    """
    Extrae un embedding de 128 dimensiones usando VGGish preentrenado.

    La señal se remuestrea de 4 kHz a 16 kHz (tasa nativa de VGGish).
    Si la señal es demasiado corta, devuelve un vector de ceros.
    """
    if not TORCH_AVAILABLE or not VGGISH_AVAILABLE:
        return np.zeros(128, dtype=np.float32)

    # Remuestrear a 16 kHz
    sig_16k = scipy.signal.resample_poly(signal, up=4, down=1).astype(np.float32)

    if len(sig_16k) < 400:  # menos de 25 ms a 16 kHz
        return np.zeros(128, dtype=np.float32)

    # Normalizar a [-1, 1]
    max_val = np.abs(sig_16k).max()
    if max_val > 0:
        sig_16k = sig_16k / max_val

    try:
        with torch.no_grad():
            tensor = torch.tensor(sig_16k).unsqueeze(0)  # (1, T)
            emb = vggish_model(tensor)                    # (T_frames, 128)
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)
            result = emb.mean(dim=0).cpu().numpy()        # (128,)
    except Exception:
        result = np.zeros(128, dtype=np.float32)

    return result.astype(np.float32)


def build_vggish_embeddings(
    signals: list[np.ndarray],
    force_rebuild: bool = False,
) -> np.ndarray:
    """
    Extrae embeddings VGGish para todas las señales etiquetadas.

    Retorna
    -------
    X_vggish : np.ndarray shape (1923, 128) float32
    """
    cache_path = RESULTS_DIR / "X_vggish_embeddings.npy"

    if cache_path.exists() and not force_rebuild:
        print("  [caché] Cargando embeddings VGGish desde disco...")
        return np.load(str(cache_path))

    print("  Cargando modelo VGGish preentrenado...")
    model = torchvggish.vggish()
    model.eval()

    n = len(signals)
    X_vggish = np.empty((n, 128), dtype=np.float32)

    print(f"  Extrayendo {n} embeddings VGGish...")
    t0 = time.time()
    for i, sig in enumerate(signals):
        X_vggish[i] = extract_vggish_embedding(sig, model)
        if (i + 1) % 200 == 0 or (i + 1) == n:
            print(f"    {i+1}/{n} ({time.time()-t0:.1f}s)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), X_vggish)
    print(f"  Embeddings guardados en {cache_path}")
    return X_vggish


def run_vggish_svm_loso(
    X_vggish: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """
    Valida SVM sobre embeddings VGGish con Leave-One-Subject-Out.

    Retorna dict con per_fold, mean, std (mismo formato que run_cnn_loso).
    """
    # Filtro de sujetos válidos
    cas_per_subj = {s: int((y[groups == s] == 1).sum()) for s in np.unique(groups)}
    valid_subjs  = [s for s, n in cas_per_subj.items() if n >= MIN_CAS_TRAIN]
    mask_valid   = np.isin(groups, valid_subjs)

    X_loso   = X_vggish[mask_valid]
    y_loso   = y[mask_valid]
    grp_loso = groups[mask_valid]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", probability=True,
            random_state=RANDOM_STATE,
        )),
    ])

    loso    = LeaveOneGroupOut()
    n_folds = loso.get_n_splits(X_loso, y_loso, grp_loso)
    per_fold: list[dict] = []

    t_start = time.time()

    for fold_i, (train_idx, test_idx) in enumerate(loso.split(X_loso, y_loso, grp_loso)):
        subj_num   = int(grp_loso[test_idx[0]])
        subj_label = f"P{subj_num}" if subj_num <= N_PATIENTS else f"C{subj_num - N_PATIENTS}"

        p = copy.deepcopy(pipe)
        p.fit(X_loso[train_idx], y_loso[train_idx])

        y_pred = p.predict(X_loso[test_idx])
        y_prob = p.predict_proba(X_loso[test_idx])[:, 1]

        metrics = _compute_metrics(y_loso[test_idx], y_pred, y_prob)
        metrics["fold"]       = fold_i + 1
        metrics["subject_id"] = subj_label
        per_fold.append(metrics)

        print(f"  Fold {fold_i+1}/{n_folds} — {subj_label} — AUC: {metrics['auc']:.3f}")

    elapsed = time.time() - t_start
    print(f"VGGish+SVM LOSO completado en {elapsed/60:.1f} minutos.")

    mean, std = _aggregate_folds(per_fold)
    return {"per_fold": per_fold, "mean": mean, "std": std}


# ===========================================================================
# PART 4 — Figuras
# ===========================================================================

def plot_sample_spectrograms(
    X_spectro: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Figura 1: Cuadrícula 2×4 con ejemplos de espectrogramas CAS y NO CAS.
    """
    rng      = np.random.default_rng(RANDOM_STATE)
    cas_idx  = np.where(y == 1)[0]
    noc_idx  = np.where(y == 0)[0]
    sel_cas  = rng.choice(cas_idx, size=4, replace=False)
    sel_noc  = rng.choice(noc_idx, size=4, replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    fig.suptitle("Ejemplos de espectrogramas — CAS vs NO CAS", fontsize=14)

    for col, idx in enumerate(sel_cas):
        ax = axes[0, col]
        ax.imshow(X_spectro[idx], aspect="auto", origin="lower",
                  cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"CAS #{col+1}")
        ax.set_xlabel("Tiempo (ms)")
        ax.set_ylabel("Frec. (Hz)" if col == 0 else "")
        ax.set_yticks([])
        ax.set_xticks([])

    for col, idx in enumerate(sel_noc):
        ax = axes[1, col]
        ax.imshow(X_spectro[idx], aspect="auto", origin="lower",
                  cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"NO CAS #{col+1}")
        ax.set_xlabel("Tiempo (ms)")
        ax.set_ylabel("Frec. (Hz)" if col == 0 else "")
        ax.set_yticks([])
        ax.set_xticks([])

    plt.tight_layout()
    out = FIGURES_DIR / "fig1_sample_spectrograms.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {out}")


def plot_auc_comparison(
    results_cnn: Optional[dict],
    results_vggish: Optional[dict],
    step6_aucs: dict,
    results_pretrained: Optional[dict] = None,
) -> None:
    """
    Figura 2: Comparativa de AUC LOSO entre todos los modelos (barras + scatter).
    """
    models  = []
    means   = []
    stds    = []
    folds_  = []
    colors  = []

    color_map = {"SVM": "steelblue", "RF": "forestgreen",
                 "CNN": "darkorange", "VGGish+SVM": "mediumpurple",
                 "CNN+Pretrain": "crimson"}

    for name in ["SVM", "RF"]:
        if name in step6_aucs:
            models.append(name)
            means.append(step6_aucs[name]["mean_auc"])
            stds.append(step6_aucs[name]["std_auc"])
            folds_.append(step6_aucs[name]["per_fold_auc"])
            colors.append(color_map[name])

    if results_cnn:
        models.append("CNN")
        means.append(results_cnn["mean"]["auc"])
        stds.append(results_cnn["std"]["auc"])
        folds_.append([f["auc"] for f in results_cnn["per_fold"]])
        colors.append(color_map["CNN"])

    if results_vggish:
        models.append("VGGish+SVM")
        means.append(results_vggish["mean"]["auc"])
        stds.append(results_vggish["std"]["auc"])
        folds_.append([f["auc"] for f in results_vggish["per_fold"]])
        colors.append(color_map["VGGish+SVM"])

    if results_pretrained:
        models.append("CNN+Pretrain")
        means.append(results_pretrained["mean"]["auc"])
        stds.append(results_pretrained["std"]["auc"])
        folds_.append([f["auc"] for f in results_pretrained["per_fold"]])
        colors.append(color_map["CNN+Pretrain"])

    if not models:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.8,
                  error_kw={"elinewidth": 1.5})

    # Scatter de AUC por fold
    for xi, fold_aucs in zip(x, folds_):
        jitter = np.random.default_rng(RANDOM_STATE).uniform(-0.15, 0.15, len(fold_aucs))
        ax.scatter(xi + jitter, fold_aucs, color="black", alpha=0.4, s=15, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("AUC (media ± std)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparativa AUC LOSO — Modelos tradicionales vs Deep Learning", fontsize=13)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Azar")
    ax.legend()
    plt.tight_layout()

    out = FIGURES_DIR / "fig2_auc_comparison.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {out}")


def plot_cnn_training_curves(training_curves: dict) -> None:
    """
    Figura 3: Curvas de entrenamiento CNN (último fold).
    """
    train_loss = training_curves["train_loss"]
    val_auc    = training_curves["val_auc"]
    stop_ep    = training_curves["early_stop_epoch"]
    subj       = training_curves["last_fold_subject"]

    if not train_loss:
        return

    epochs = np.arange(1, len(train_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Curvas de entrenamiento CNN — último fold ({subj})", fontsize=13)

    ax1.plot(epochs, train_loss, "b-o", markersize=3, label="Train loss")
    if stop_ep < len(train_loss):
        ax1.axvline(stop_ep, color="red", linestyle="--", label=f"Early stop (ep {stop_ep})")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("BCEWithLogitsLoss")
    ax1.set_title("Pérdida de entrenamiento")
    ax1.legend()

    if val_auc:
        ax2.plot(np.arange(1, len(val_auc) + 1), val_auc, "g-o", markersize=3, label="Val AUC")
        if stop_ep < len(val_auc):
            ax2.axvline(stop_ep, color="red", linestyle="--", label=f"Early stop (ep {stop_ep})")
        ax2.set_xlabel("Época")
        ax2.set_ylabel("AUC")
        ax2.set_ylim(0, 1.05)
        ax2.set_title("AUC de validación")
        ax2.legend()

    plt.tight_layout()
    out = FIGURES_DIR / "fig3_cnn_training_curves.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {out}")


def plot_mean_spectrograms(
    X_spectro: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Figura 4: Espectrograma medio de señales CAS vs NO CAS.
    """
    mean_cas  = X_spectro[y == 1].mean(axis=0)
    mean_noc  = X_spectro[y == 0].mean(axis=0)
    vmin = min(mean_cas.min(), mean_noc.min())
    vmax = max(mean_cas.max(), mean_noc.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Espectrograma medio — CAS vs NO CAS", fontsize=13)

    im1 = ax1.imshow(mean_cas, aspect="auto", origin="lower",
                     cmap="inferno", vmin=vmin, vmax=vmax)
    ax1.set_title(f"CAS (n={int((y==1).sum())})")
    ax1.set_xlabel("Tiempo (relativo)")
    ax1.set_ylabel("Frecuencia (70–2000 Hz)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(mean_noc, aspect="auto", origin="lower",
                     cmap="inferno", vmin=vmin, vmax=vmax)
    ax2.set_title(f"NO CAS (n={int((y==0).sum())})")
    ax2.set_xlabel("Tiempo (relativo)")
    ax2.set_ylabel("Frecuencia (70–2000 Hz)")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = FIGURES_DIR / "fig4_mean_spectrograms.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {out}")


# ===========================================================================
# PART 5 — Tabla comparativa y exportación JSON
# ===========================================================================

def _load_step6_aucs() -> dict:
    """
    Carga los resultados LOSO de step6 (SVM y RF) desde sus CSVs.

    Retorna dict con claves 'SVM' y/o 'RF', cada una con
    {mean_auc, std_auc, per_fold_auc, mean_sensitivity,
     std_sensitivity, mean_specificity, std_specificity}.
    """
    result: dict = {}
    for name, path in [("SVM", _STEP6_SVM_CSV), ("RF", _STEP6_RF_CSV)]:
        if path.exists():
            df = pd.read_csv(str(path))
            result[name] = {
                "mean_auc":          float(df["auc"].mean()),
                "std_auc":           float(df["auc"].std()),
                "per_fold_auc":      df["auc"].tolist(),
                "mean_sensitivity":  float(df["sensitivity"].mean()),
                "std_sensitivity":   float(df["sensitivity"].std()),
                "mean_specificity":  float(df["specificity"].mean()),
                "std_specificity":   float(df["specificity"].std()),
                "mean_f1":           float(df["f1"].mean()),
                "std_f1":            float(df["f1"].std()),
            }
        else:
            print(f"  Advertencia: no se encontró {path}")
    return result


def _print_comparison_table(
    results_cnn: Optional[dict],
    results_vggish: Optional[dict],
    step6_aucs: dict,
    results_pretrained: Optional[dict] = None,
) -> None:
    """Imprime la tabla ASCII comparativa de todos los modelos."""

    def fmt(mean: Optional[float], std: Optional[float]) -> str:
        if mean is None:
            return "     N/A     "
        return f" {mean:.3f}±{std:.3f} "

    def row(name: str, auc_m: Optional[float], auc_s: Optional[float],
            sen_m: Optional[float], sen_s: Optional[float],
            spe_m: Optional[float], spe_s: Optional[float]) -> str:
        return (f"║ {name:<14}║{fmt(auc_m, auc_s)}║{fmt(sen_m, sen_s)}"
                f"║{fmt(spe_m, spe_s)}║")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          COMPARATIVA MODELOS — AUC LOSO                     ║")
    print("╠══════════════╦══════════════╦═════════════╦════════════════╣")
    print("║    Modelo    ║     AUC      ║ Sensitivity ║  Specificity   ║")
    print("╠══════════════╬══════════════╬═════════════╬════════════════╣")

    for name in ["SVM", "RF"]:
        if name in step6_aucs:
            d = step6_aucs[name]
            print(row(name, d["mean_auc"], d["std_auc"],
                       d["mean_sensitivity"], d["std_sensitivity"],
                       d["mean_specificity"], d["std_specificity"]))
        else:
            print(row(name, None, None, None, None, None, None))

    if results_cnn:
        m = results_cnn["mean"]
        s = results_cnn["std"]
        print(row("CNN", m["auc"], s["auc"], m["sensitivity"], s["sensitivity"],
                  m["specificity"], s["specificity"]))
    else:
        print(row("CNN", None, None, None, None, None, None))

    if results_vggish:
        m = results_vggish["mean"]
        s = results_vggish["std"]
        print(row("VGGish+SVM", m["auc"], s["auc"], m["sensitivity"], s["sensitivity"],
                  m["specificity"], s["specificity"]))
    else:
        print(row("VGGish+SVM", None, None, None, None, None, None))

    if results_pretrained:
        m = results_pretrained["mean"]
        s = results_pretrained["std"]
        print(row("CNN+Pretrain", m["auc"], s["auc"], m["sensitivity"], s["sensitivity"],
                  m["specificity"], s["specificity"]))
    else:
        print(row("CNN+Pretrain", None, None, None, None, None, None))

    print("╚══════════════╩══════════════╩═════════════╩════════════════╝")

    # Conclusión
    svm_auc = step6_aucs.get("SVM", {}).get("mean_auc")
    cnn_auc = results_cnn["mean"]["auc"] if results_cnn else None

    if cnn_auc is not None and svm_auc is not None:
        if cnn_auc > svm_auc:
            print()
            print("✅ CNN mejora el baseline — considerar usar CNN para inferencia en step7")
        else:
            print()
            print("⚠️  CNN no mejora el baseline con este dataset.")
            print("    Causa probable: insuficientes datos etiquetados")
            print("    (n=1923, 17 sujetos) para generalización LOSO.")
    print()


def save_results_json(
    results_cnn: Optional[dict],
    results_vggish: Optional[dict],
    step6_aucs: dict,
    results_pretrained: Optional[dict] = None,
) -> None:
    """Guarda el diccionario comparativo completo en JSON."""
    out = {
        "metadata": {
            "n_labeled":    1923,
            "n_cas":        590,
            "n_nocas":      1333,
            "n_folds":      17,
            "random_state": RANDOM_STATE,
            "date":         datetime.now().isoformat(timespec="seconds"),
        },
        "baselines":      step6_aucs,
        "CNN":            results_cnn if results_cnn else "no_disponible",
        "VGGish_SVM":     results_vggish if results_vggish else "no_disponible",
        "CNN_Pretrained": results_pretrained if results_pretrained else "no_disponible",
    }

    path = RESULTS_DIR / "dl_comparison_results.json"
    with open(str(path), "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2, default=float)
    print(f"  Resultados JSON guardados en {path}")


# ===========================================================================
# Punto de entrada principal
# ===========================================================================

def main() -> None:
    """Ejecuta el pipeline completo de step8."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 8 — DEEP LEARNING PARA CLASIFICACIÓN CAS")
    print("=" * 60)

    # ------------------------------------------------------------------
    # PART 0: Señales etiquetadas
    # ------------------------------------------------------------------
    print("\n[PART 0] Cargando señales etiquetadas...")
    signals, y_labeled, groups = load_labeled_signals()

    # ------------------------------------------------------------------
    # PART 1: Espectrogramas
    # ------------------------------------------------------------------
    print("\n[PART 1] Generando espectrogramas...")
    X_spectro = build_spectrograms(signals)
    print(f"  X_spectro shape: {X_spectro.shape}")

    # ------------------------------------------------------------------
    # PART 2b: Preentrenamiento semi-supervisado con autoencoder
    # ------------------------------------------------------------------
    results_pretrained_cnn: Optional[dict] = None
    if TORCH_AVAILABLE:
        print("\n[PART 2b] Preentrenamiento autoencoder (14 900 señales)...")
        _cache_all = RESULTS_DIR / "X_spectrograms_all14900.npy"
        if not _cache_all.exists():
            print("  Reconstruyendo 14 900 señales para espectrogramas...")
            _subjects_all = _build_subjects()
            _all_signals, *_ = build_dataset(_subjects_all)
        else:
            _all_signals = []  # no se usa; build_all_spectrograms carga de caché
        X_all_spectro = build_all_spectrograms(_all_signals)
        autoencoder = pretrain_autoencoder(X_all_spectro)
        print("\n[PART 2b] CNN con encoder preentrenado — LOSO...")
        results_pretrained_cnn = run_pretrained_cnn_loso(
            X_spectro, y_labeled, groups, autoencoder
        )
    else:
        print("\n[PART 2b] Preentrenamiento omitido (PyTorch no disponible).")

    # ------------------------------------------------------------------
    # PART 2: CNN
    # ------------------------------------------------------------------
    results_cnn: Optional[dict] = None
    if TORCH_AVAILABLE:
        print("\n[PART 2] CNN — validación LOSO...")
        results_cnn = run_cnn_loso(X_spectro, y_labeled, groups)
    else:
        print("\n[PART 2] CNN omitida (PyTorch no instalado).")

    # ------------------------------------------------------------------
    # PART 3: VGGish + SVM
    # ------------------------------------------------------------------
    results_vggish: Optional[dict] = None
    if VGGISH_AVAILABLE:
        print("\n[PART 3] VGGish + SVM — extracción de embeddings...")
        X_vggish = build_vggish_embeddings(signals)
        print("  Validación LOSO con SVM sobre embeddings...")
        results_vggish = run_vggish_svm_loso(X_vggish, y_labeled, groups)
    else:
        print("\n[PART 3] VGGish+SVM omitido (torchvggish no instalado).")

    # ------------------------------------------------------------------
    # PART 4: Figuras
    # ------------------------------------------------------------------
    print("\n[PART 4] Generando figuras...")
    step6_aucs = _load_step6_aucs()

    plot_sample_spectrograms(X_spectro, y_labeled)
    plot_mean_spectrograms(X_spectro, y_labeled)
    plot_auc_comparison(results_cnn, results_vggish, step6_aucs, results_pretrained_cnn)

    if TORCH_AVAILABLE and results_cnn:
        plot_cnn_training_curves(results_cnn["training_curves"])

    # ------------------------------------------------------------------
    # PART 5: Tabla y JSON
    # ------------------------------------------------------------------
    print("\n[PART 5] Tabla comparativa de resultados:")
    _print_comparison_table(results_cnn, results_vggish, step6_aucs, results_pretrained_cnn)
    save_results_json(results_cnn, results_vggish, step6_aucs, results_pretrained_cnn)

    print("=" * 60)
    print("Step 8 completado.")
    print("=" * 60)


if __name__ == "__main__":
    main()
