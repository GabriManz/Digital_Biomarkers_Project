"""
Extrae features y espectrogramas de SPRSound (Detection task) para Transfer Learning.

Lee los JSONs de anotación event-level de Detection (train + valid + test),
localiza los WAVs correspondientes, y genera un .npz con:
  - X_features : (N, 141) features clásicas + SOTA
  - X_spectros : (N, 64, 64) espectrogramas normalizados
  - y           : (N,) etiquetas binarias (1=CAS, 0=NO_CAS)
  - subjects    : (N,) ID numérico del paciente (para LOSO si se desea)
  - event_types : (N,) tipo de evento original (string → int mapping)

CAS positivos: Wheeze, Rhonchi, Wheeze+Crackle, Stridor
CAS negativos: Normal, Fine Crackle, Coarse Crackle
"""
import os
import sys
import glob
import json
import time
import numpy as np
from pathlib import Path
import librosa

# Localizar la raíz
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from step5_features import extract_features
from step8_deep_learning import signal_to_spectrogram

try:
    from sota_pipeline import extract_sota_features_global
    SOTA_AVAILABLE = True
except ImportError:
    SOTA_AVAILABLE = False
    print("[WARN] sota_pipeline no disponible; se usarán zeros para las 4 features SOTA.")

# ──────────────────────────────────────────────────────────────────────────────
# Rutas
# ──────────────────────────────────────────────────────────────────────────────
SPRSOUND_ROOT = PROJECT_ROOT / "data" / "SPRSound" / "SPRSound-main" / "SPRSound-main"
DETECTION_DIR = SPRSOUND_ROOT / "Detection"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results" / "transfer_sprsound"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tipos de evento → etiqueta CAS binaria
CAS_TYPES = {"wheeze", "rhonchi", "wheeze+crackle", "stridor"}

FS_TARGET = 4000  # Sample rate de nuestro pipeline


def resolve_wav_path(json_path: str) -> str | None:
    """
    Dado un JSON en .../xxx_detection_json/file.json,
    resuelve el WAV correspondiente en .../xxx_detection_wav/file.wav
    """
    wav_path = json_path.replace(".json", ".wav").replace("_json", "_wav")
    if os.path.exists(wav_path):
        return wav_path
    # Fallback: buscar en todas las carpetas wav
    filename = os.path.basename(json_path).replace(".json", ".wav")
    candidates = list(DETECTION_DIR.glob(f"**/{filename}"))
    wav_candidates = [c for c in candidates if "_wav" in str(c)]
    if wav_candidates:
        return str(wav_candidates[0])
    return None


def extract_subject_id(filename: str) -> str:
    """Extrae el ID del paciente del primer campo del filename.
    Ejemplo: '00014365_4.3_1_p4_7545.json' → '00014365'
    """
    return filename.split("_")[0]


def process_sprsound():
    """Pipeline principal: parsea JSONs, extrae features/espectrogramas, guarda .npz."""
    
    if not DETECTION_DIR.exists():
        print(f"ERROR: No se encontró el directorio Detection: {DETECTION_DIR}")
        print("Asegúrate de colocar el dataset en data/SPRSound/SPRSound-main/SPRSound-main/Detection/")
        return False
    
    # Buscar JSONs en las subcarpetas de Detection (train + valid + test)
    json_files = sorted(glob.glob(str(DETECTION_DIR / "**" / "*.json"), recursive=True))
    
    if len(json_files) == 0:
        print("No se encontraron archivos de anotación (.json) en Detection.")
        return False
    
    print(f"Encontrados {len(json_files)} archivos de anotación en Detection.")
    
    # Contenedores
    all_features = []
    all_spectrograms = []
    all_labels = []
    all_subjects = []
    all_event_types = []
    
    # Mapeo de subject ID (string) → entero
    subject_mapping = {}
    subject_counter = 0
    
    # Mapeo de tipo de evento → entero
    event_type_mapping = {}
    event_type_counter = 0
    
    # Contadores
    n_files_ok = 0
    n_files_no_wav = 0
    n_files_error = 0
    n_events_total = 0
    n_events_too_short = 0
    n_events_error = 0
    
    t0 = time.time()
    
    for fi, json_path in enumerate(json_files):
        # Resolver WAV
        wav_path = resolve_wav_path(json_path)
        if wav_path is None:
            n_files_no_wav += 1
            continue
        
        # Cargar JSON
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                annotation_data = json.load(f)
        except Exception as e:
            n_files_error += 1
            continue
        
        # Extraer eventos
        event_list = annotation_data.get("event_annotation", [])
        if not event_list:
            continue
        
        # Cargar audio una sola vez por archivo (lazy: solo si hay eventos)
        try:
            y_audio, sr = librosa.load(wav_path, sr=FS_TARGET)
        except Exception as e:
            n_files_error += 1
            continue
        
        # Extraer subject ID del filename
        basename = os.path.basename(json_path)
        subj_str = extract_subject_id(basename)
        if subj_str not in subject_mapping:
            subject_mapping[subj_str] = subject_counter
            subject_counter += 1
        subj_id = subject_mapping[subj_str]
        
        n_files_ok += 1
        
        for event in event_list:
            start_ms = event.get("start", None)
            end_ms = event.get("end", None)
            event_type = event.get("type", "")
            
            if start_ms is None or end_ms is None:
                continue
            
            # Parsear tiempos
            try:
                start_sec = float(start_ms) / 1000.0
                end_sec = float(end_ms) / 1000.0
            except (ValueError, TypeError):
                continue
            
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            
            # Cortar el segmento
            event_audio = y_audio[start_sample:end_sample]
            if len(event_audio) < 200:  # <50ms → demasiado corto
                n_events_too_short += 1
                continue
            
            n_events_total += 1
            
            # Mapeo de tipo
            if event_type not in event_type_mapping:
                event_type_mapping[event_type] = event_type_counter
                event_type_counter += 1
            
            # Etiqueta binaria CAS
            is_cas = 1 if event_type.lower() in CAS_TYPES else 0
            
            # Extraer 137 features clásicas
            try:
                feats = extract_features(event_audio, fs=FS_TARGET)
            except Exception:
                n_events_error += 1
                continue
            
            # Extraer 4 features SOTA
            if SOTA_AVAILABLE:
                try:
                    sota_feats = extract_sota_features_global(event_audio, fs=FS_TARGET)
                except Exception:
                    sota_feats = np.zeros(4)
            else:
                sota_feats = np.zeros(4)
            
            full_feats = np.concatenate([feats, sota_feats])  # 137 + 4 = 141
            
            # Extraer espectrograma 64×64
            try:
                spectro = signal_to_spectrogram(event_audio, fs=FS_TARGET)
            except Exception:
                n_events_error += 1
                continue
            
            all_features.append(full_feats)
            all_spectrograms.append(spectro)
            all_labels.append(is_cas)
            all_subjects.append(subj_id)
            all_event_types.append(event_type_mapping[event_type])
            
        # Progreso
        if (fi + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed
            eta = (len(json_files) - fi - 1) / rate
            print(f"  [{fi+1:5d}/{len(json_files)}] Eventos: {len(all_labels):5d} | "
                  f"CAS: {sum(all_labels)} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
    
    elapsed_total = time.time() - t0
    
    if len(all_labels) == 0:
        print("\nERROR: No se pudieron extraer eventos con anotaciones válidas.")
        return False
    
    # Convertir a arrays
    X_features = np.array(all_features, dtype=np.float32)
    X_spectros = np.array(all_spectrograms, dtype=np.float32)
    y_labels = np.array(all_labels, dtype=np.int32)
    subjects = np.array(all_subjects, dtype=np.int32)
    event_types = np.array(all_event_types, dtype=np.int32)
    
    print(f"\n{'='*60}")
    print(f"  PROCESAMIENTO DE SPRSOUND COMPLETADO ({elapsed_total:.1f}s)")
    print(f"{'='*60}")
    print(f"  Archivos procesados:     {n_files_ok}")
    print(f"  Archivos sin WAV:        {n_files_no_wav}")
    print(f"  Archivos con error:      {n_files_error}")
    print(f"  Eventos extraídos:       {len(y_labels)}")
    print(f"  Eventos demasiado cortos:{n_events_too_short}")
    print(f"  Eventos con error:       {n_events_error}")
    print(f"  CAS (positivos):         {np.sum(y_labels == 1)}")
    print(f"  NO_CAS (negativos):      {np.sum(y_labels == 0)}")
    print(f"  Sujetos únicos:          {len(subject_mapping)}")
    print(f"  X_features shape:        {X_features.shape}")
    print(f"  X_spectros shape:        {X_spectros.shape}")
    
    # Tipo → nombre para referencia
    type_inv = {v: k for k, v in event_type_mapping.items()}
    print(f"\n  Distribución de tipos de evento:")
    for tid in sorted(type_inv.keys()):
        mask = (event_types == tid)
        tname = type_inv[tid]
        cas_label = "CAS" if tname.lower() in CAS_TYPES else "NO_CAS"
        print(f"    {tname:20s} ({cas_label:6s}): {np.sum(mask):5d}")
    
    # Guardar NPZ
    output_path = OUTPUT_DIR / "sprsound_processed.npz"
    np.savez(
        output_path,
        X_features=X_features,
        X_spectros=X_spectros,
        y=y_labels,
        subjects=subjects,
        event_types=event_types
    )
    print(f"\n  Resultados guardados en {output_path}")
    
    # Guardar mappings para referencia
    mappings = {
        "subject_mapping": subject_mapping,
        "event_type_mapping": event_type_mapping
    }
    with open(OUTPUT_DIR / "sprsound_mappings.json", "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)
    print(f"  Mappings guardados en {OUTPUT_DIR / 'sprsound_mappings.json'}")
    
    return True


if __name__ == "__main__":
    process_sprsound()
