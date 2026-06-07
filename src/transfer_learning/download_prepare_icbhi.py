import os
import zipfile
import urllib.request
from pathlib import Path

# Configuración de rutas
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "ICBHI_2017"
ZIP_PATH = DATA_DIR / "ICBHI_final_database.zip"
URL = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip"

def download_and_extract():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if not ZIP_PATH.exists():
        print(f"Descargando ICBHI 2017 desde {URL}...")
        try:
            import ssl
            context = ssl._create_unverified_context()
            # Configurar un User-Agent para evitar bloqueos del servidor
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
            opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(URL, ZIP_PATH)
            print("Descarga completada con éxito.")
        except Exception as e:
            print(f"Error al descargar desde el servidor principal: {e}")
            print("Por favor, descargue el archivo manualmente y colóquelo en:")
            print(f"  {ZIP_PATH}")
            return False
    else:
        print("El archivo ZIP ya existe.")

    # Extraer el zip
    extracted_folder = DATA_DIR / "ICBHI_final_database"
    if not extracted_folder.exists() or len(list(extracted_folder.glob("*"))) == 0:
        print("Extrayendo archivos...")
        try:
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            print("Extracción completada con éxito.")
        except Exception as e:
            print(f"Error al extraer el archivo zip: {e}")
            return False
    else:
        print("Los archivos ya están extraídos.")
        
    return True

if __name__ == "__main__":
    download_and_extract()
