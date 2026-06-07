import os
from pathlib import Path

# Configuración de rutas
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "SPRSound"

INSTRUCTIONS = """
=== INSTRUCCIONES PARA DESCARGAR EL DATASET SPRSOUND ===
1. Acceda al repositorio oficial de SPRSound en GitHub:
   https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound
   o en Gitee (si prefiere un acceso más rápido desde servidores asiáticos):
   https://gitee.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound

2. Descargue la carpeta del dataset (usualmente bajo el directorio de los retos BioCAS, ej: 2022 o posterior).
   Necesitamos los archivos de audio en formato .wav y sus respectivas anotaciones en formato .json.

3. Coloque los archivos descomprimidos en el siguiente directorio:
   {data_dir}

4. La estructura interna esperada debe ser:
   {data_dir}/record/           <- Archivos de audio (.wav)
   {data_dir}/annotation/       <- Archivos de anotación (.json) o archivos .json en el raíz
   {data_dir}/metadata.csv      <- O archivo general del reto (ej: train_test_split.json)
"""

def verify_and_prepare():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Comprobar si hay archivos .wav en Data/SPRSound
    wav_files = list(DATA_DIR.glob("**/*.wav"))
    json_files = list(DATA_DIR.glob("**/*.json"))
    
    print(f"Buscando archivos en {DATA_DIR}...")
    
    if len(wav_files) == 0:
        print("ADVERTENCIA: No se encontraron archivos de audio (.wav) en el directorio.")
        print(INSTRUCTIONS.format(data_dir=DATA_DIR))
        return False
        
    print(f"Se encontraron {len(wav_files)} archivos de audio y {len(json_files)} anotaciones JSON.")
    print("El dataset está listo para ser preprocesado.")
    return True

if __name__ == "__main__":
    verify_and_prepare()
