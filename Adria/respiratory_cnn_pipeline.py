"""
PIPELINE COMPLET: Preprocessament + Espectrogrames + CNN (ResNet + Custom)
===========================================================================
Evaluació de la resposta broncodilatadora – Classificació CAS vs no-CAS

Passos:
  1.  Lectura i preprocessament de les senyals (resample, Butterworth, notch)
  2.  Segmentació inspiració/espiració amb marques temporals
  3.  Generació d'espectrogrames (mel-spectrogram o STFT)
  4.  Entrenament de dos models CNN:
        a) ResNet-18 transferit (últimes 2 capes reentrenades + classificador)
        b) CNN Custom lleugera dissenyada per a so respiratori
  5.  Avaluació: matrius de confusió, mètriques (Acc, F1, AUC, Prec, Rec)
  6.  Guardat de figures (espectrogrames, matrius, corbes ROC/AUC)

Ús:
  python respiratory_cnn_pipeline.py --data_dir /path/to/mat/files

Arguments opcionals:
  --data_dir   Directori amb els fitxers P*.mat, tP*.mat, C*.mat, tC*.mat
  --labels     Ruta al fitxer proy_labels.mat
  --out_dir    Directori de sortida (default: ./cnn_output)
  --epochs     Nombre d'epochs (default: 30)
  --batch_size Mida del batch (default: 32)
  --img_size   Mida de la imatge d'espectrograma (default: 128)
  --n_mels     Nombre de bandes mel (default: 64)
  --demo_only  Salta preprocessament si ja existeix signals.pkl
"""

import os, sys, argparse, time, pickle, warnings
import numpy as np
import scipy.io
from scipy.signal import resample_poly, butter, filtfilt, iirnotch
from scipy.signal import spectrogram as scipy_spectrogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, f1_score, accuracy_score)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# ── Torch imports (amb comprovació) ──────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision import models, transforms
    from torchvision.models import resnet18, ResNet18_Weights
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] PyTorch {torch.__version__}  |  Device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch no disponible – s'usarà mode de demostració (sklearn SVM/RF)")

# ══════════════════════════════════════════════════════════════════════════════
# PARÀMETRES GLOBALS
# ══════════════════════════════════════════════════════════════════════════════
FS_ORIG    = 12500
FS_TARGET  = 4000
UP, DOWN   = 8, 25
BP_LOW     = 70
BP_HIGH    = 1900
BP_ORDER   = 8
NOTCH_F0   = 50
NOTCH_BW   = 1.0
IMG_H      = 128          # píxels altura espectrograma
IMG_W      = 128          # píxels amplada espectrograma
N_MELS     = 64           # bandes mel
NPERSEG    = 256          # finestra STFT
NOVERLAP   = 192          # solapament STFT (75%)
MIN_SEG_LEN = 400         # mostres mínimes per segent vàlid (0.1 s @ 4000 Hz)

# Paleta de colors
BG   = 'white'
TEXT = 'black'
GRID = '#2a2d35'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': GRID,
    'grid.linewidth': 0.5, 'font.family': 'monospace',
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. LECTURA I PREPROCESSAMENT
# ══════════════════════════════════════════════════════════════════════════════

def read_mat(path):
    mat        = scipy.io.loadmat(path)
    data       = mat['data'].ravel().astype(np.float64)
    datastart  = mat['datastart']
    dataend    = mat['dataend']
    samplerate = float(mat['samplerate'].ravel()[0])
    titles     = list(mat.get('titles', ['CH0', 'CH1']))
    nch, nblk  = datastart.shape
    signals    = {}
    for ch in range(nch):
        signals[ch] = {}
        for blk in range(nblk):
            s = int(datastart[ch, blk]) - 1
            e = int(dataend[ch, blk])
            signals[ch][blk] = data[s:e]
    return signals, samplerate, titles, nch, nblk


def read_time_marks(path):
    mat  = scipy.io.loadmat(path)
    keys = [k for k in mat if not k.startswith('__')]
    if not keys:
        return {}
    cell = mat[keys[0]]
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


def preprocess_signal(raw, fs_orig=FS_ORIG):
    """Resample → Butterworth BP → Notch comb → normalització robusta."""
    # 1. Resample
    sig = resample_poly(raw, UP, DOWN)
    # 2. Butterworth pas-banda zero-phase
    nyq  = FS_TARGET / 2.0
    b, a = butter(BP_ORDER, [BP_LOW/nyq, BP_HIGH/nyq], btype='bandpass')
    sig  = filtfilt(b, a, sig)
    # 3. Notch comb 50 Hz + harmònics
    freq = NOTCH_F0
    while freq < FS_TARGET / 2:
        Q       = freq / NOTCH_BW
        b_n, a_n = iirnotch(freq, Q, fs=FS_TARGET)
        sig     = filtfilt(b_n, a_n, sig)
        freq   += NOTCH_F0
    # 4. Normalització robusta (MAD z-score)
    med = np.median(sig)
    mad = np.median(np.abs(sig - med))
    if mad > 1e-12:
        sig = (sig - med) / (1.4826 * mad)
    else:
        sig = sig - med
    return sig.astype(np.float32)


def run_preprocessing(data_dir, out_dir):
    """
    Processa tots els pacients i controls.
    Retorna all_signals (llista) + metadata (DataFrame).
    Guarda signals.pkl i metadata.csv.
    """
    pkl_path = os.path.join(out_dir, 'signals.pkl')
    csv_path = os.path.join(out_dir, 'metadata.csv')

    # Si ja existeix, carregar directament
    if os.path.exists(pkl_path) and os.path.exists(csv_path):
        print(f"  [CACHE] Carregant signals.pkl existent…")
        with open(pkl_path, 'rb') as f:
            all_signals = pickle.load(f)
        metadata_df = pd.read_csv(csv_path, index_col='segment_id')
        print(f"  Segments carregats: {len(all_signals)}")
        return all_signals, metadata_df

    subjects = (
        [{'id': p,    'sig': f'P{p}.mat',  'time': f'tP{p}.mat'} for p in range(1, 24)] +
        [{'id': 23+c, 'sig': f'C{c}.mat',  'time': f'tC{c}.mat'} for c in range(1, 6)]
    )

    all_signals = []
    meta_rows   = []

    for subj in subjects:
        pth_sig  = os.path.join(data_dir, subj['sig'])
        pth_time = os.path.join(data_dir, subj['time'])
        if not os.path.exists(pth_sig):
            continue

        pid   = subj['id']
        print(f"  Processant {subj['sig']} (id={pid})…", end=' ', flush=True)
        t0 = time.time()

        signals, fs_orig, titles, nch, nblk = read_mat(pth_sig)
        marks = read_time_marks(pth_time) if os.path.exists(pth_time) else {}

        for ch in range(nch):
            for blk in range(nblk):
                raw = signals[ch][blk]
                if len(raw) < 100:
                    continue

                s_proc   = preprocess_signal(raw, fs_orig=fs_orig)
                pre_post = 1 if blk < 3 else 2
                channel  = ch + 1

                if blk not in marks or len(marks[blk]) == 0:
                    continue

                for cyc in marks[blk]:
                    cyc = np.asarray(cyc).flatten()
                    if len(cyc) < 4:
                        continue
                    tsi, tei, tse, tee = cyc[:4]

                    # Inspiració
                    i0, i1 = int(round(tsi * FS_TARGET)), int(round(tei * FS_TARGET))
                    if 0 <= i0 < len(s_proc) and (i1 - i0) >= MIN_SEG_LEN:
                        seg = s_proc[i0:min(i1, len(s_proc))].copy()
                        all_signals.append(seg)
                        meta_rows.append({'participant': pid, 'prepost': pre_post,
                                          'channel': channel, 'phase': 1})

                    # Espiració
                    i0, i1 = int(round(tse * FS_TARGET)), int(round(tee * FS_TARGET))
                    if 0 <= i0 < len(s_proc) and (i1 - i0) >= MIN_SEG_LEN:
                        seg = s_proc[i0:min(i1, len(s_proc))].copy()
                        all_signals.append(seg)
                        meta_rows.append({'participant': pid, 'prepost': pre_post,
                                          'channel': channel, 'phase': 2})

        print(f"OK ({time.time()-t0:.1f}s) → total: {len(all_signals)}")

    metadata_df = pd.DataFrame(meta_rows)

    # Guardar
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_signals, f, protocol=4)
    metadata_df.to_csv(csv_path, index=True, index_label='segment_id')
    print(f"  Guardat: {pkl_path}  |  {csv_path}")
    print(f"  Total segments: {len(all_signals)}")
    return all_signals, metadata_df


# ══════════════════════════════════════════════════════════════════════════════
# 2. ESPECTROGRAMES
# ══════════════════════════════════════════════════════════════════════════════

def mel_filterbank(n_mels, n_fft, fs):
    """Banc de filtres mel simple (sense librosa)."""
    f_min, f_max = 0.0, fs / 2.0
    mel_min  = 2595 * np.log10(1 + f_min / 700)
    mel_max  = 2595 * np.log10(1 + f_max / 700)
    mel_pts  = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts   = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_pts  = np.floor((n_fft + 1) * hz_pts / fs).astype(int)
    fbank    = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, cen, hi = bin_pts[m-1], bin_pts[m], bin_pts[m+1]
        for k in range(lo, cen):
            if cen != lo:
                fbank[m-1, k] = (k - lo) / (cen - lo)
        for k in range(cen, hi):
            if hi != cen:
                fbank[m-1, k] = (hi - k) / (hi - cen)
    return fbank


def signal_to_spectrogram(sig, fs=FS_TARGET, img_h=IMG_H, img_w=IMG_W,
                           n_mels=N_MELS, nperseg=NPERSEG, noverlap=NOVERLAP):
    """
    Converteix una senyal 1-D a imatge d'espectrograma mel [img_h × img_w].
    """
    # Padding/truncating per tenir almenys nperseg mostres
    if len(sig) < nperseg:
        sig = np.pad(sig, (0, nperseg - len(sig)))

    f, t, Sxx = scipy_spectrogram(sig, fs=fs, nperseg=nperseg,
                                   noverlap=noverlap, window='hann',
                                   scaling='spectrum')
    # Escala logarítmica
    Sxx = np.log1p(Sxx + 1e-8)

    # Aplicar banc de filtres mel
    n_fft  = nperseg
    fbank  = mel_filterbank(n_mels, n_fft, fs)
    mel_S  = fbank @ Sxx   # (n_mels, n_time)

    # Normalitzar [0, 1]
    mel_S -= mel_S.min()
    if mel_S.max() > 1e-8:
        mel_S /= mel_S.max()

    # Resize a img_h × img_w amb interpolació bilineal simple (numpy)
    from scipy.ndimage import zoom
    sh, sw = mel_S.shape
    zy = img_h / sh
    zx = img_w / sw
    img = zoom(mel_S, (zy, zx), order=1)
    img = np.clip(img, 0, 1)

    return img.astype(np.float32)


def save_spectrogram_examples(all_signals, labels_bin, metadata_df,
                               out_dir, n_examples=12):
    """Guarda una figura amb exemples d'espectrogrames CAS i no-CAS."""
    os.makedirs(out_dir, exist_ok=True)

    idx_cas   = np.where(labels_bin == 1)[0][:n_examples//2]
    idx_nocas = np.where(labels_bin == 0)[0][:n_examples//2]
    idxs      = list(idx_cas) + list(idx_nocas)
    lbls      = ['CAS'] * len(idx_cas) + ['no-CAS'] * len(idx_nocas)
    colors    = ['#e63946'] * len(idx_cas) + ['#2a9d8f'] * len(idx_nocas)

    fig, axes = plt.subplots(2, n_examples//2, figsize=(20, 7),
                              facecolor=BG)
    fig.suptitle('Exemples d\'espectrogrames mel – CAS vs no-CAS',
                 fontsize=13, color=TEXT, fontweight='bold')

    for ax_i, (idx, lbl, col) in enumerate(zip(idxs, lbls, colors)):
        row = ax_i // (n_examples//2)
        col_i = ax_i % (n_examples//2)
        ax  = axes[row, col_i]

        sig = all_signals[idx]
        img = signal_to_spectrogram(sig)
        im  = ax.imshow(img, aspect='auto', origin='lower',
                        cmap='inferno', vmin=0, vmax=1)
        meta = metadata_df.iloc[idx]
        phase_str = 'Insp' if meta['phase'] == 1 else 'Esp'
        pp_str    = 'Pre' if meta['prepost'] == 1 else 'Post'
        ax.set_title(f'{lbl}  [{phase_str}|{pp_str}]',
                     color=col, fontsize=8.5, fontweight='bold')
        ax.set_xlabel('Temps →', fontsize=7)
        if col_i == 0:
            ax.set_ylabel('Mel →', fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, 'spectrogram_examples.png')
    fig.savefig(path, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  Espectrogrames d'exemple → {path}")


def save_spectrogram_detail(sig, label_str, out_path, fs=FS_TARGET):
    """Guarda un espectrograma detallat d'una sola senyal."""
    f_arr, t_arr, Sxx = scipy_spectrogram(
        sig, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP,
        window='hann', scaling='spectrum')
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor=BG,
                              gridspec_kw={'height_ratios': [1, 2.5]})
    fig.suptitle(f'Espectrograma detallat – {label_str}',
                 fontsize=12, color=TEXT, fontweight='bold')

    # Senyal temporal
    t_sig = np.arange(len(sig)) / fs
    axes[0].plot(t_sig, sig, color='#adb5bd', linewidth=0.6)
    axes[0].set_ylabel('Amplitud (norm.)', fontsize=9)
    axes[0].set_xlabel('Temps (s)', fontsize=9)
    axes[0].grid(True, alpha=0.2)
    axes[0].set_facecolor(BG)
    axes[0].set_xlim(t_sig[0], t_sig[-1])

    # Espectrograma
    im = axes[1].pcolormesh(t_arr, f_arr, Sxx_db,
                             shading='gouraud', cmap='inferno')
    axes[1].set_ylabel('Freqüència (Hz)', fontsize=9)
    axes[1].set_xlabel('Temps (s)', fontsize=9)
    axes[1].set_ylim(0, FS_TARGET/2)
    axes[1].set_facecolor(BG)
    cbar = plt.colorbar(im, ax=axes[1], pad=0.01)
    cbar.set_label('dB', fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATASET PyTorch
# ══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class SpectrogramDataset(Dataset):
        """Dataset que genera imatges d'espectrograma on-the-fly."""
        def __init__(self, signals, labels, transform=None,
                     img_h=IMG_H, img_w=IMG_W):
            self.signals   = signals
            self.labels    = labels
            self.transform = transform
            self.img_h     = img_h
            self.img_w     = img_w

        def __len__(self):
            return len(self.signals)

        def __getitem__(self, idx):
            img = signal_to_spectrogram(
                self.signals[idx],
                img_h=self.img_h, img_w=self.img_w)
            # 3 canals (RGB) per ResNet
            img3 = np.stack([img, img, img], axis=0)  # (3, H, W)
            x    = torch.tensor(img3, dtype=torch.float32)
            if self.transform:
                x = self.transform(x)
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODELS CNN
# ══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    # ── 4a. ResNet-18 Transfer Learning ─────────────────────────────────────
    class ResNetCAS(nn.Module):
        """
        ResNet-18 pre-entrenat en ImageNet.
        Estratègia de fine-tuning:
          - Congelar totes les capes excepte layer4 i el classificador.
          - Substituir les 2 últimes capes: avgpool → AdaptiveAvgPool2d
            i fc → Linear(512, 256) + ReLU + Dropout + Linear(256, 2)
        """
        def __init__(self, num_classes=2, dropout=0.4):
            super().__init__()
            base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

            # Congelar tot excepte layer4
            for name, param in base.named_parameters():
                if not name.startswith('layer4'):
                    param.requires_grad = False

            # Extreure el backbone sense el cap final
            self.backbone = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4,
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            feat_dim  = 512

            # Cap de classificació nou (2 últimes capes substituïdes)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            x = self.backbone(x)
            x = self.pool(x)
            x = self.classifier(x)
            return x


    # ── 4b. CNN Custom lleugera ──────────────────────────────────────────────
    class CustomCNN(nn.Module):
        """
        CNN custom dissenyada per a espectrogrames de so respiratori.
        Arquitectura:
          Conv2D(3,32,3) → BN → ReLU → MaxPool
          Conv2D(32,64,3) → BN → ReLU → MaxPool
          Conv2D(64,128,3) → BN → ReLU → MaxPool
          Conv2D(128,256,3) → BN → ReLU → AdaptiveAvgPool(4×4)
          FC(256*4*4,512) → BN → ReLU → Dropout
          FC(512,128) → ReLU → Dropout
          FC(128,2)
        """
        def __init__(self, num_classes=2, dropout=0.5):
            super().__init__()

            def conv_block(in_c, out_c, kernel=3, pool=True):
                layers = [
                    nn.Conv2d(in_c, out_c, kernel, padding=kernel//2, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                ]
                if pool:
                    layers.append(nn.MaxPool2d(2))
                return nn.Sequential(*layers)

            self.features = nn.Sequential(
                conv_block(3,   32,  3, pool=True),   # 64×64
                conv_block(32,  64,  3, pool=True),   # 32×32
                conv_block(64,  128, 3, pool=True),   # 16×16
                conv_block(128, 256, 3, pool=False),  # 16×16
                nn.AdaptiveAvgPool2d(4),               # 4×4
            )
            feat_dim = 256 * 4 * 4

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.6),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))


# ══════════════════════════════════════════════════════════════════════════════
# 5. ENTRENAMENT
# ══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    def make_weighted_sampler(labels_array):
        """Oversampling de la classe minoritària."""
        classes, counts = np.unique(labels_array, return_counts=True)
        w_class   = 1.0 / counts
        w_samples = np.array([w_class[np.where(classes==l)[0][0]]
                               for l in labels_array])
        return WeightedRandomSampler(
            weights=torch.DoubleTensor(w_samples),
            num_samples=len(w_samples),
            replacement=True)


    def train_one_epoch(model, loader, criterion, optimizer, device):
        model.train()
        total_loss, correct, n = 0, 0, 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (out.argmax(1) == y).sum().item()
            n          += len(y)
        return total_loss / n, correct / n


    @torch.no_grad()
    def evaluate(model, loader, device):
        model.eval()
        all_probs, all_preds, all_labels = [], [], []
        for X, y in loader:
            X = X.to(device)
            out   = model(X)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(y.numpy())
        all_probs  = np.vstack(all_probs)
        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        return all_probs, all_preds, all_labels


    def train_model(model, train_sigs, train_lbls, val_sigs, val_lbls,
                    epochs=30, batch_size=32, lr=1e-3, device=None,
                    model_name="model"):
        if device is None:
            device = DEVICE
        model = model.to(device)

        train_ds  = SpectrogramDataset(train_sigs, train_lbls)
        val_ds    = SpectrogramDataset(val_sigs,   val_lbls)
        sampler   = make_weighted_sampler(train_lbls)
        train_dl  = DataLoader(train_ds, batch_size=batch_size,
                               sampler=sampler, num_workers=0)
        val_dl    = DataLoader(val_ds, batch_size=batch_size,
                               shuffle=False, num_workers=0)

        # Weighted loss per classe desbalancejada
        n0 = (train_lbls == 0).sum()
        n1 = (train_lbls == 1).sum()
        w  = torch.tensor([1.0, n0/max(n1, 1)], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {'train_loss': [], 'train_acc': [],
                   'val_loss':   [], 'val_acc':   [], 'val_f1': []}

        best_f1, best_state = 0.0, None
        for ep in range(1, epochs + 1):
            t_loss, t_acc = train_one_epoch(model, train_dl, criterion,
                                             optimizer, device)
            v_probs, v_preds, v_lbls = evaluate(model, val_dl, device)
            v_loss = -np.mean(np.log(v_probs[np.arange(len(v_lbls)),
                                              v_lbls] + 1e-8))
            v_acc  = (v_preds == v_lbls).mean()
            v_f1   = f1_score(v_lbls, v_preds, zero_division=0)

            history['train_loss'].append(t_loss)
            history['train_acc'].append(t_acc)
            history['val_loss'].append(v_loss)
            history['val_acc'].append(v_acc)
            history['val_f1'].append(v_f1)

            if v_f1 >= best_f1:
                best_f1    = v_f1
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

            scheduler.step()
            if ep % 5 == 0 or ep == 1:
                print(f"    [{model_name}] Ep {ep:3d}/{epochs}  "
                      f"loss={t_loss:.4f}  acc={t_acc:.3f}  "
                      f"val_f1={v_f1:.3f}")

        if best_state:
            model.load_state_dict(best_state)
        return model, history


# ══════════════════════════════════════════════════════════════════════════════
# 6. AVALUACIÓ I FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, class_names, title, save_path, cmap='Blues'):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel('Predicció', fontsize=12)
    ax.set_ylabel('Valor real', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', color=TEXT)
    ax.set_facecolor(BG)

    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            col_txt = 'black' if cm_norm[i, j] > thresh else 'black'
            ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.1%})',
                    ha='center', va='center', color=col_txt,
                    fontsize=11, fontweight='bold')

    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close(fig)


def plot_roc_curve(y_true, y_prob, model_name, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val     = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    ax.plot(fpr, tpr, color='#e63946', lw=2,
            label=f'{model_name}  (AUC = {auc_val:.4f})')
    ax.plot([0,1],[0,1], '--', color='#666', lw=1, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.12, color='#e63946')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'Corba ROC – {model_name}', fontsize=13,
                 fontweight='bold', color=TEXT)
    ax.legend(fontsize=10, framealpha=0.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.2)
    ax.set_facecolor(BG)
    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return auc_val


def plot_pr_curve(y_true, y_prob, model_name, save_path):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap           = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    ax.plot(rec, prec, color='#2a9d8f', lw=2,
            label=f'{model_name}  (AP = {ap:.4f})')
    ax.fill_between(rec, prec, alpha=0.12, color='#2a9d8f')
    ax.axhline(y_true.mean(), color='#666', ls='--', lw=1,
               label=f'Baseline (prev={y_true.mean():.3f})')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(f'Corba Precision-Recall – {model_name}',
                 fontsize=13, fontweight='bold', color=TEXT)
    ax.legend(fontsize=10, framealpha=0.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.2)
    ax.set_facecolor(BG)
    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return ap


def plot_training_history(history, model_name, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG)
    fig.suptitle(f'Historial d\'entrenament – {model_name}',
                 fontsize=13, color=TEXT, fontweight='bold')

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], color='#e63946', lw=1.8,
            label='Train loss')
    ax.plot(epochs, history['val_loss'],   color='#f4a261', lw=1.8,
            ls='--', label='Val loss')
    ax.set_title('Loss', color=TEXT); ax.set_xlabel('Epoch')
    ax.legend(framealpha=0.2); ax.grid(True, alpha=0.2)
    ax.set_facecolor(BG)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history['train_acc'], color='#2a9d8f', lw=1.8,
            label='Train acc')
    ax.plot(epochs, history['val_acc'],   color='#457b9d', lw=1.8,
            ls='--', label='Val acc')
    ax.set_title('Accuracy', color=TEXT); ax.set_xlabel('Epoch')
    ax.legend(framealpha=0.2); ax.grid(True, alpha=0.2)
    ax.set_facecolor(BG); ax.set_ylim(0, 1)

    # F1 Val
    ax = axes[2]
    ax.plot(epochs, history['val_f1'], color='#6a4c93', lw=2,
            label='Val F1')
    ax.fill_between(epochs, history['val_f1'], alpha=0.15, color='#6a4c93')
    ax.set_title('F1 Validació', color=TEXT); ax.set_xlabel('Epoch')
    ax.legend(framealpha=0.2); ax.grid(True, alpha=0.2)
    ax.set_facecolor(BG); ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close(fig)


def plot_metrics_comparison(results_dict, save_path):
    """Barra comparativa de les mètriques finals dels dos models."""
    models   = list(results_dict.keys())
    metrics  = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'ap']
    labels_m = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'AP']
    colors   = ['#e63946', '#2a9d8f', '#457b9d', '#6a4c93', '#f4a261', '#06d6a0']

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    for m_i, mname in enumerate(models):
        vals = [results_dict[mname].get(m, 0) for m in metrics]
        offset = (m_i - 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=mname, alpha=0.85,
                      color=['#e63946', '#2a9d8f'][m_i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=8.5, color=TEXT)

    ax.set_xticks(x); ax.set_xticklabels(labels_m, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Comparació de mètriques: ResNet-18 vs CNN Custom',
                 fontsize=13, fontweight='bold', color=TEXT)
    ax.legend(fontsize=11, framealpha=0.2)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_facecolor(BG)
    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close(fig)


def plot_roc_comparison(results_dict, save_path):
    """Superposició de les corbes ROC dels dos models."""
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
    palette = ['#e63946', '#2a9d8f', '#f4a261', '#457b9d']

    for i, (mname, res) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
        auc_val     = res['auc']
        ax.plot(fpr, tpr, color=palette[i], lw=2.5,
                label=f'{mname}  (AUC={auc_val:.4f})')
        ax.fill_between(fpr, tpr, alpha=0.07, color=palette[i])

    ax.plot([0,1],[0,1], '--', color='#555', lw=1, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Comparació Corbes ROC – CAS vs no-CAS',
                 fontsize=13, fontweight='bold', color=TEXT)
    ax.legend(fontsize=11, framealpha=0.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.2)
    ax.set_facecolor(BG)
    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 7. FALLBACK SKLEARN (si PyTorch no disponible)
# ══════════════════════════════════════════════════════════════════════════════

def sklearn_fallback(train_sigs, train_lbls, val_sigs, val_lbls,
                     test_sigs, test_lbls, out_dir):
    """
    Quan PyTorch no és disponible: extreu features d'espectrograma i
    entrena un SVM i Random Forest.
    """
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print("\n  [FALLBACK] Extraient features d'espectrograma…")

    def extract_features(sigs):
        feats = []
        for sig in sigs:
            img = signal_to_spectrogram(sig, img_h=32, img_w=32)
            # Estadístics de l'espectrograma per banda de freqüència
            row_means = img.mean(axis=1)  # (32,)
            row_stds  = img.std(axis=1)   # (32,)
            col_means = img.mean(axis=0)  # (32,)
            # RMS, max, skewness
            rms = np.sqrt(np.mean(sig**2))
            mx  = np.max(np.abs(sig))
            feat = np.concatenate([row_means, row_stds, col_means,
                                    [rms, mx, img.mean(), img.std()]])
            feats.append(feat)
        return np.array(feats)

    all_sigs_tr = train_sigs + val_sigs
    all_lbls_tr = np.concatenate([train_lbls, val_lbls])

    print("  Extraient train…")
    X_tr = extract_features(all_sigs_tr)
    print("  Extraient test…")
    X_te = extract_features(test_sigs)

    models_sk = {
        'SVM': Pipeline([('sc', StandardScaler()),
                         ('clf', SVC(kernel='rbf', probability=True,
                                     class_weight='balanced', C=10))]),
        'RandomForest': Pipeline([('sc', StandardScaler()),
                                  ('clf', RandomForestClassifier(
                                      n_estimators=200, class_weight='balanced',
                                      random_state=42, n_jobs=-1))]),
    }

    results = {}
    for mname, clf in models_sk.items():
        print(f"\n  Entrenant {mname}…")
        clf.fit(X_tr, all_lbls_tr)
        probs  = clf.predict_proba(X_te)[:, 1]
        preds  = clf.predict(X_te)
        y_true = test_lbls
        cm     = confusion_matrix(y_true, preds)
        auc    = roc_auc_score(y_true, probs)
        ap     = average_precision_score(y_true, probs)
        f1     = f1_score(y_true, preds, zero_division=0)
        acc    = accuracy_score(y_true, preds)
        from sklearn.metrics import precision_score, recall_score
        prec   = precision_score(y_true, preds, zero_division=0)
        rec    = recall_score(y_true, preds, zero_division=0)

        print(f"    AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")
        results[mname] = {
            'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'auc': auc, 'ap': ap,
            'y_true': y_true, 'y_prob': probs, 'cm': cm,
        }

        # Figures
        plot_confusion_matrix(
            cm, ['no-CAS', 'CAS'], f'Matriu de Confusió – {mname}',
            os.path.join(out_dir, f'confusion_{mname.lower()}.png'))
        plot_roc_curve(
            y_true, probs, mname,
            os.path.join(out_dir, f'roc_{mname.lower()}.png'))
        plot_pr_curve(
            y_true, probs, mname,
            os.path.join(out_dir, f'pr_{mname.lower()}.png'))

    plot_metrics_comparison(results,
        os.path.join(out_dir, 'metrics_comparison.png'))
    plot_roc_comparison(results,
        os.path.join(out_dir, 'roc_comparison.png'))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 8. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, 'figures')
    os.makedirs(plots_dir, exist_ok=True)

    print('\n' + '='*65)
    print('  PIPELINE CLASSIFICACIÓ CAS – SO RESPIRATORI')
    print('='*65)

    # ── 1. Preprocessament ────────────────────────────────────────────────────
    print('\n[1/6] PREPROCESSAMENT…')
    all_signals, metadata_df = run_preprocessing(args.data_dir, args.out_dir)

    if len(all_signals) == 0:
        print("ERROR: No s'han trobat senyals. "
              "Comprova que --data_dir conté els fitxers .mat")
        sys.exit(1)

    # ── 2. Carregar etiquetes ─────────────────────────────────────────────────
    print('\n[2/6] CARREGANT ETIQUETES…')
    mat_labels = scipy.io.loadmat(args.labels)
    raw_labels = mat_labels['labels'].ravel()

    if len(raw_labels) != len(all_signals):
        print(f"  ADVERTÈNCIA: {len(raw_labels)} etiquetes ≠ "
              f"{len(all_signals)} senyals")
        min_len    = min(len(raw_labels), len(all_signals))
        raw_labels = raw_labels[:min_len]
        all_signals = all_signals[:min_len]
        metadata_df = metadata_df.iloc[:min_len]

    # Interpretació de les etiquetes:
    #   1 = Normal (no-CAS)
    #   2 = Wheeze (CAS)
    #   3 = Crackle (CAS)
    #   6 = Wheeze + Crackle (CAS)
    # Classificació binària: CAS (1) vs no-CAS (0)
    labels_bin = (raw_labels >= 2).astype(int)

    # Subset dels 1923 segments anotats (labels 2 i 3, les 2 classes pures)
    # + tots els no-CAS per tenir context suficient
    # → Per a l'entrenament fem servir tots els segments etiquetats
    cas_mask  = (labels_bin == 1)
    nocas_mask= (labels_bin == 0)
    print(f"  Segments CAS:    {cas_mask.sum()} "
          f"(wheeze={np.sum(raw_labels==2)}, crackle={np.sum(raw_labels==3)}, "
          f"both={np.sum(raw_labels==6)})")
    print(f"  Segments no-CAS: {nocas_mask.sum()}")

    # Submostreig dels no-CAS per equilibrar (màxim 4× el nombre de CAS)
    rng         = np.random.default_rng(42)
    n_cas       = cas_mask.sum()
    nocas_idxs  = np.where(nocas_mask)[0]
    n_nocas_use = min(len(nocas_idxs), n_cas * 4)
    sel_nocas   = rng.choice(nocas_idxs, size=n_nocas_use, replace=False)
    sel_cas     = np.where(cas_mask)[0]
    sel_all     = np.sort(np.concatenate([sel_cas, sel_nocas]))

    sigs_use   = [all_signals[i] for i in sel_all]
    lbls_use   = labels_bin[sel_all]
    meta_use   = metadata_df.iloc[sel_all].reset_index(drop=True)

    print(f"\n  Subconjunt usat per a CNN:")
    print(f"    CAS:    {(lbls_use==1).sum()}")
    print(f"    no-CAS: {(lbls_use==0).sum()}")
    print(f"    Total:  {len(lbls_use)}")

    # ── 3. Espectrogrames d'exemple ───────────────────────────────────────────
    print('\n[3/6] GENERANT ESPECTROGRAMES D\'EXEMPLE…')
    save_spectrogram_examples(sigs_use, lbls_use, meta_use, plots_dir,
                               n_examples=12)

    # Espectrogrames detallats individuals
    for cls, cls_name in [(1, 'CAS'), (0, 'noCAS')]:
        idxs_cls = np.where(lbls_use == cls)[0][:3]
        for k, idx in enumerate(idxs_cls):
            save_spectrogram_detail(
                sigs_use[idx], cls_name,
                os.path.join(plots_dir, f'spectrogram_detail_{cls_name}_{k+1}.png'))
    print(f"  Espectrogrames detallats guardats a {plots_dir}")

    # ── 4. Divisió train/val/test ─────────────────────────────────────────────
    print('\n[4/6] DIVISIÓ TRAIN / VAL / TEST…')
    # Estratificada per participant (evitar data leakage)
    participants = meta_use['participant'].values
    unique_parts = np.unique(participants)
    rng.shuffle(unique_parts)

    n_test  = max(3, int(len(unique_parts) * 0.15))
    n_val   = max(2, int(len(unique_parts) * 0.15))
    test_parts = unique_parts[:n_test]
    val_parts  = unique_parts[n_test:n_test+n_val]
    train_parts= unique_parts[n_test+n_val:]

    def split_by_participants(parts):
        mask = np.isin(participants, parts)
        return ([sigs_use[i] for i in np.where(mask)[0]],
                lbls_use[mask],
                meta_use[mask])

    train_sigs, train_lbls, _ = split_by_participants(train_parts)
    val_sigs,   val_lbls,   _ = split_by_participants(val_parts)
    test_sigs,  test_lbls,  _ = split_by_participants(test_parts)

    print(f"  Train: {len(train_sigs)} segs | participants: {len(train_parts)}")
    print(f"  Val:   {len(val_sigs)} segs   | participants: {len(val_parts)}")
    print(f"  Test:  {len(test_sigs)} segs  | participants: {len(test_parts)}")

    # ── 5. Entrenament i avaluació ────────────────────────────────────────────
    print('\n[5/6] ENTRENAMENT CNN…')
    results = {}

    if TORCH_AVAILABLE:
        for model_name, model_cls in [('ResNet-18', ResNetCAS),
                                       ('CustomCNN', CustomCNN)]:
            print(f"\n  ── {model_name} ──")
            model = model_cls(num_classes=2)
            n_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
            print(f"  Paràmetres entrenables: {n_params:,}")

            model, history = train_model(
                model, train_sigs, train_lbls,
                val_sigs, val_lbls,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=DEVICE,
                model_name=model_name)

            # Avaluació en test
            test_ds = SpectrogramDataset(test_sigs, test_lbls)
            test_dl = DataLoader(test_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)
            probs, preds, y_true = evaluate(model, test_dl, DEVICE)
            y_prob = probs[:, 1]

            cm   = confusion_matrix(y_true, preds)
            auc  = roc_auc_score(y_true, y_prob)
            ap   = average_precision_score(y_true, y_prob)
            f1   = f1_score(y_true, preds, zero_division=0)
            acc  = accuracy_score(y_true, preds)
            from sklearn.metrics import precision_score, recall_score
            prec = precision_score(y_true, preds, zero_division=0)
            rec  = recall_score(y_true, preds, zero_division=0)

            print(f"\n  Resultats test – {model_name}:")
            print(f"    Accuracy:  {acc:.4f}")
            print(f"    Precision: {prec:.4f}")
            print(f"    Recall:    {rec:.4f}")
            print(f"    F1:        {f1:.4f}")
            print(f"    AUC-ROC:   {auc:.4f}")
            print(f"    AP:        {ap:.4f}")
            print(f"\n{classification_report(y_true, preds, target_names=['no-CAS','CAS'])}")

            results[model_name] = {
                'accuracy': acc, 'precision': prec, 'recall': rec,
                'f1': f1, 'auc': auc, 'ap': ap,
                'y_true': y_true, 'y_prob': y_prob, 'cm': cm,
            }

            safe = model_name.lower().replace('-', '_').replace(' ', '_')
            # Figures
            plot_confusion_matrix(
                cm, ['no-CAS', 'CAS'],
                f'Matriu de Confusió – {model_name}',
                os.path.join(plots_dir, f'confusion_{safe}.png'))
            plot_roc_curve(
                y_true, y_prob, model_name,
                os.path.join(plots_dir, f'roc_{safe}.png'))
            plot_pr_curve(
                y_true, y_prob, model_name,
                os.path.join(plots_dir, f'pr_{safe}.png'))
            plot_training_history(
                history, model_name,
                os.path.join(plots_dir, f'history_{safe}.png'))

            # Guardar model
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, f'{safe}_best.pth'))

    else:
        # Fallback sklearn
        results = sklearn_fallback(
            train_sigs, train_lbls, val_sigs, val_lbls,
            test_sigs, test_lbls, plots_dir)

    # ── 6. Figures comparatives ───────────────────────────────────────────────
    print('\n[6/6] GENERANT FIGURES COMPARATIVES…')
    if len(results) >= 1:
        plot_metrics_comparison(
            results, os.path.join(plots_dir, 'metrics_comparison.png'))
    if len(results) >= 2:
        plot_roc_comparison(
            results, os.path.join(plots_dir, 'roc_comparison.png'))

    # Resum final
    print('\n' + '='*65)
    print('  RESUM FINAL')
    print('='*65)
    for mname, res in results.items():
        print(f"\n  {mname}:")
        for k in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'ap']:
            print(f"    {k:12s}: {res[k]:.4f}")

    figs = sorted(os.listdir(plots_dir))
    print(f"\n  Figures generades ({len(figs)}) a {plots_dir}:")
    for f in figs:
        print(f"    · {f}")
    print('='*65)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Classificació CAS vs no-CAS – So Respiratori CNN')
    ap.add_argument('--data_dir',   default='.',
                    help='Directori amb P*.mat, tP*.mat, C*.mat, tC*.mat')
    ap.add_argument('--labels',     default='proy_labels.mat',
                    help='Fitxer proy_labels.mat')
    ap.add_argument('--out_dir',    default='./cnn_output',
                    help='Directori de sortida')
    ap.add_argument('--epochs',     type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--img_size',   type=int, default=128,
                    help='Mida de la imatge espectrograma (img_size × img_size)')
    ap.add_argument('--n_mels',     type=int, default=64,
                    help='Nombre de bandes mel')
    args = ap.parse_args()

    IMG_H = args.img_size
    IMG_W = args.img_size
    N_MELS = args.n_mels

    main(args)
