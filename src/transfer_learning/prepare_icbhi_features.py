import os
import sys
import glob
import numpy as np
from pathlib import Path
import librosa

# Localizar la raíz
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from step5_features import extract_features
from step8_deep_learning import signal_to_spectrogram

# Rutas de entrada/salida
ICBHI_DATA_DIR = PROJECT_ROOT / "Data" / "ICBHI_2017"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results" / "transfer_learning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_icbhi():
    # Buscar todos los archivos .txt de anotaciones
    # Nota: el zip puede extraerse directamente en ICBHI_2017 o en una subcarpeta
    txt_files = glob.glob(str(ICBHI_DATA_DIR / "**" / "*.txt"), recursive=True)
    # Excluir archivos de metadatos/diagnósticos si hay alguno
    txt_files = [f for f in txt_files if not os.path.basename(f).endswith("diagnosis.txt") 
                 and not os.path.basename(f).endswith("demographic_info.txt")
                 and not os.path.basename(f).endswith("train_test.txt")]
    
    if len(txt_files) == 0:
        print("No se encontraron archivos de anotación de ICBHI 2017.")
        return False
        
    print(f"Encontrados {len(txt_files)} archivos de pacientes en ICBHI.")
    
    all_features = []
    all_spectrograms = []
    all_labels = []
    all_subjects = []
    
    cycle_count = 0
    
    for txt_path in txt_files:
        wav_path = txt_path.replace(".txt", ".wav")
        if not os.path.exists(wav_path):
            continue
            
        # El nombre del archivo suele empezar con el ID del sujeto, ej: 101_1b1_Al_sc_Meditron.txt
        subject_id = int(os.path.basename(txt_path).split("_")[0])
        
        # Cargar audio a 4000 Hz
        try:
            y, sr = librosa.load(wav_path, sr=4000)
        except Exception as e:
            print(f"Error cargando {wav_path}: {e}")
            continue
            
        # Leer anotaciones del ciclo
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                start = float(parts[0])
                end = float(parts[1])
                crackles = int(parts[2])
                wheezes = int(parts[3])
                
                # Slicing del ciclo de audio
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                cycle_audio = y[start_sample:end_sample]
                
                # Evitar segmentos vacíos
                if len(cycle_audio) < 100:
                    continue
                    
                # Extraer features tradicionales (137 features)
                feats = extract_features(cycle_audio, fs=4000)
                
                # Extraer 4 features SOTA adicionales (Tonal Index, etc.)
                # Para simplificar y hacerlo compatible con las 141 features de step6:
                # Calculamos las 4 features manuales y las concatenamos.
                # Let's write helper or call extract_sota_features_global from src/sota_pipeline.py
                try:
                    from sota_pipeline import extract_sota_features_global
                    sota_feats = extract_sota_features_global(cycle_audio, fs=4000)
                except ImportError:
                    # Fallback si no está importable
                    sota_feats = np.zeros(4)
                
                full_feats = np.concatenate([feats, sota_feats])
                
                # Extraer espectrograma Mel 2D (64x64)
                spectro = signal_to_spectrogram(cycle_audio, fs=4000)
                
                # Etiqueta: 1 = Wheeze presente (CAS homólogo), 0 = Wheeze ausente (NO CAS)
                label = 1 if wheezes == 1 else 0
                
                all_features.append(full_feats)
                all_spectrograms.append(spectro)
                all_labels.append(label)
                all_subjects.append(subject_id)
                cycle_count += 1
                
        if cycle_count % 500 == 0 or cycle_count < 10:
            print(f"Procesados {cycle_count} ciclos respiratorios...")
            
    X_features = np.array(all_features)
    X_spectros = np.array(all_spectrograms)
    y_labels = np.array(all_labels)
    subjects = np.array(all_subjects)
    
    print(f"Procesamiento finalizado. Total de ciclos: {len(y_labels)}")
    print(f"  CAS (Wheezes): {np.sum(y_labels == 1)}")
    print(f"  NO_CAS: {np.sum(y_labels == 0)}")
    print(f"  Shape features: {X_features.shape}")
    print(f"  Shape espectrogramas: {X_spectros.shape}")
    
    # Guardar matrices
    np.savez(
        OUTPUT_DIR / "icbhi_processed.npz",
        X_features=X_features,
        X_spectros=X_spectros,
        y=y_labels,
        subjects=subjects
    )
    print(f"Resultados guardados en {OUTPUT_DIR / 'icbhi_processed.npz'}")
    return True

if __name__ == "__main__":
    process_icbhi()
