"""
Pre-entrena modelos (XGBoost + CNN) sobre SPRSound procesado.

Aplica Nullspace Projection a nivel de sujeto (eliminando el sesgo de paciente)
y entrena dos modelos:
  1. XGBoost sobre features proyectadas (141 dims)
  2. CNN 2D sobre espectrogramas (64×64)

Salida:
  - xgb_sprsound_pretrained.pkl
  - cnn_sprsound_pretrained.pt
  - subject_projection.npy  (para aplicar en fine-tuning local)
"""
import os
import sys
import pickle
import numpy as np
import scipy.linalg
from pathlib import Path

# Localizar la raíz
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Cargar XGBoost
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

INPUT_NPZ = PROJECT_ROOT / "outputs" / "results" / "transfer_sprsound" / "sprsound_processed.npz"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results" / "transfer_sprsound"


def compute_subject_projection(X: np.ndarray, y_subject: np.ndarray,
                                max_subjects: int = 50) -> np.ndarray:
    """
    Calcula la Nullspace Projection que elimina la identidad del sujeto.
    
    Esto es la estrategia clave para mitigar el Domain Shift:
    - Entrena un clasificador lineal multi-clase para predecir el sujeto.
    - Proyecta las features al espacio nulo del clasificador.
    - El resultado son features "sujeto-agnósticas" que conservan la información
      relevante para CAS pero eliminan las características específicas del paciente
      (timbre de voz, anatomía de las vías respiratorias, etc.).
    """
    from sklearn.linear_model import LogisticRegression
    
    unique_subjects = np.unique(y_subject)
    n_subjects = len(unique_subjects)
    
    if n_subjects <= 1:
        print("  Solo se detectó un sujeto. Omitiendo la proyección del espacio nulo.")
        return np.eye(X.shape[1])
    
    # Si hay demasiados sujetos, usar un subset aleatorio para estabilidad
    if n_subjects > max_subjects:
        print(f"  {n_subjects} sujetos detectados; muestreando {max_subjects} para la proyección...")
        np.random.seed(42)
        sampled = np.random.choice(unique_subjects, max_subjects, replace=False)
        mask = np.isin(y_subject, sampled)
        X_sub, y_sub = X[mask], y_subject[mask]
    else:
        X_sub, y_sub = X, y_subject
    
    print(f"  Calculando proyección de espacio nulo sobre {len(np.unique(y_sub))} sujetos...")
    clf = LogisticRegression(max_iter=2000, random_state=42,
                             solver='lbfgs', C=1.0)
    clf.fit(X_sub, y_sub)
    W = clf.coef_  # (n_subjects, n_features)
    null_space = scipy.linalg.null_space(W)
    
    if null_space.shape[1] == 0:
        print("  ADVERTENCIA: Espacio nulo vacío. Usando identidad.")
        return np.eye(X.shape[1])
    
    P = null_space @ null_space.T
    print(f"  Proyeccion calculada: {X.shape[1]}d -> {null_space.shape[1]}d espacio nulo "
          f"({100*null_space.shape[1]/X.shape[1]:.1f}% preservado)")
    return P


def pretrain_sprsound():
    if not INPUT_NPZ.exists():
        print(f"No se encontró el archivo de datos procesados de SPRSound: {INPUT_NPZ}")
        print("Ejecuta primero: python src/transfer_sprsound/prepare_sprsound_features.py")
        return False
    
    data = np.load(INPUT_NPZ)
    X_features = data["X_features"]
    X_spectros = data["X_spectros"]
    y = data["y"]
    subjects = data["subjects"]
    
    print(f"Cargado dataset SPRSound: {X_features.shape[0]} muestras "
          f"(CAS: {np.sum(y==1)}, NO_CAS: {np.sum(y==0)})")
    
    # ──────────────────────────────────────────────────────────────────────
    # 1. Nullspace Projection para mitigar sesgo de paciente
    # ──────────────────────────────────────────────────────────────────────
    print("\n[1/3] Calculando proyección de espacio nulo (sesgo de sujeto)...")
    
    # Estandarizar antes de la proyección
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    P_subject = compute_subject_projection(X_scaled, subjects)
    np.save(OUTPUT_DIR / "subject_projection.npy", P_subject)
    print(f"  Matriz de proyección guardada: {OUTPUT_DIR / 'subject_projection.npy'}")
    
    # Proyectar features
    X_features_proj = X_scaled @ P_subject
    
    # ──────────────────────────────────────────────────────────────────────
    # 2. Pre-entrenar XGBoost
    # ──────────────────────────────────────────────────────────────────────
    if XGB_AVAILABLE:
        print("\n[2/3] Pre-entrenando XGBoost en features proyectadas de SPRSound...")
        scale_pos_weight = float(np.sum(y == 0) / max(np.sum(y == 1), 1))
        print(f"  scale_pos_weight = {scale_pos_weight:.2f}")
        
        xgb = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            scale_pos_weight=scale_pos_weight, subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss", verbosity=1
        )
        xgb.fit(X_features_proj, y)
        
        # Evaluar en training set (referencia)
        train_probs = xgb.predict_proba(X_features_proj)[:, 1]
        from sklearn.metrics import roc_auc_score
        train_auc = roc_auc_score(y, train_probs)
        print(f"  AUC en training set SPRSound: {train_auc:.4f}")
        
        xgb_path = OUTPUT_DIR / "xgb_sprsound_pretrained.pkl"
        with open(xgb_path, "wb") as f:
            pickle.dump(xgb, f)
        print(f"  Modelo guardado: {xgb_path}")
    else:
        print("\n[2/3] XGBoost no disponible - omitido.")
    
    # ──────────────────────────────────────────────────────────────────────
    # 3. Pre-entrenar CNN
    # ──────────────────────────────────────────────────────────────────────
    if TORCH_AVAILABLE:
        print("\n[3/3] Pre-entrenando CNN 2D en espectrogramas de SPRSound...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Dispositivo: {device}")
        
        X_tensor = torch.tensor(X_spectros, dtype=torch.float32).unsqueeze(1)  # (N, 1, 64, 64)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        ds = TensorDataset(X_tensor, y_tensor)
        dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
        
        model = CAS_CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        pos_weight = torch.tensor([np.sum(y == 0) / max(np.sum(y == 1), 1)],
                                  dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        model.train()
        n_epochs = 25
        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0.0
            n_correct = 0
            n_total = 0
            for Xb, yb in dl:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(Xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(yb)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                n_correct += (preds == yb).sum().item()
                n_total += len(yb)
            epoch_loss /= len(y)
            epoch_acc = n_correct / n_total
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoca {epoch:2d}/{n_epochs} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        
        cnn_path = OUTPUT_DIR / "cnn_sprsound_pretrained.pt"
        torch.save(model.state_dict(), cnn_path)
        print(f"  Modelo CNN guardado: {cnn_path}")
    else:
        print("\n[3/3] PyTorch no disponible - omitido.")
    
    print("\n[OK] Pre-entrenamiento de SPRSound completado.")
    return True


if __name__ == "__main__":
    pretrain_sprsound()
