import os
import sys
import pickle
import numpy as np
from pathlib import Path

# Localizar la raíz
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Cargar XGBoost si está disponible
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from step8_deep_learning import CAS_CNN
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

INPUT_NPZ = PROJECT_ROOT / "outputs" / "results" / "transfer_learning" / "icbhi_processed.npz"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results" / "transfer_learning"

def pretrain_models():
    if not INPUT_NPZ.exists():
        print(f"No se encontró el archivo de datos procesados: {INPUT_NPZ}")
        return False
        
    data = np.load(INPUT_NPZ)
    X_features = data["X_features"]
    X_spectros = data["X_spectros"]
    y = data["y"]
    
    print(f"Cargado dataset ICBHI con {X_features.shape[0]} muestras.")
    
    # 1. Preentrenamiento XGBoost
    if XGB_AVAILABLE:
        print("\nPreentrenando XGBoost en features de ICBHI...")
        scale_pos_weight = float(np.sum(y == 0) / np.sum(y == 1))
        xgb = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            scale_pos_weight=scale_pos_weight, subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss", verbosity=0
        )
        xgb.fit(X_features, y)
        
        xgb_path = OUTPUT_DIR / "xgb_pretrained.pkl"
        with open(xgb_path, "wb") as f:
            pickle.dump(xgb, f)
        print(f"Modelo XGBoost guardado en {xgb_path}")
    else:
        print("XGBoost no disponible para preentrenamiento.")
        
    # 2. Preentrenamiento CNN 2D (PyTorch)
    if TORCH_AVAILABLE:
        print("\nPreentrenando CNN 2D en espectrogramas de ICBHI...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo de entrenamiento: {device}")
        
        # DataLoader
        X_tensor = torch.tensor(X_spectros, dtype=torch.float32).unsqueeze(1) # (N, 1, 64, 64)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        ds = TensorDataset(X_tensor, y_tensor)
        dl = DataLoader(ds, batch_size=32, shuffle=True)
        
        model = CAS_CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        pos_weight = torch.tensor([np.sum(y == 0) / np.sum(y == 1)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        model.train()
        for epoch in range(1, 21):  # 20 épocas de preentrenamiento es suficiente
            epoch_loss = 0.0
            for Xb, yb in dl:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(Xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(yb)
            epoch_loss /= len(y)
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Época {epoch:2d}/20 — Loss: {epoch_loss:.4f}")
                
        cnn_path = OUTPUT_DIR / "cnn_pretrained.pt"
        torch.save(model.state_dict(), cnn_path)
        print(f"Modelo CNN guardado en {cnn_path}")
    else:
        print("PyTorch no disponible para preentrenamiento de CNN.")
        
    return True

if __name__ == "__main__":
    pretrain_models()
