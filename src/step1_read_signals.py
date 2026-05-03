"""
Lectura de archivos .mat exportados desde LabChart.

Extrae y organiza las señales de cada canal y bloque a partir del
vector de datos continuo almacenado en el archivo, usando los índices
de inicio y fin de segmento que LabChart incluye en la exportación.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import scipy.io

# ---------------------------------------------------------------------------
# Ruta de ejemplo — modificar para apuntar al archivo .mat deseado
# ---------------------------------------------------------------------------
EXAMPLE_FILE = r"C:\DATA\01_Proyectos\Master\Digital_Biomarkers\Project\Data\P1.mat"
# ---------------------------------------------------------------------------


def _flatten_string(arr: Any) -> str:
    """Extrae una cadena limpia de un valor anidado devuelto por loadmat."""
    # Desanidar arrays de tipo objeto (celdas MATLAB) hasta llegar al contenido
    while isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = arr.flat[0]
    # Array de caracteres: unir todos los caracteres y eliminar relleno
    if isinstance(arr, np.ndarray):
        return "".join(arr.flat).strip()
    if isinstance(arr, bytes):
        return arr.decode("latin-1").strip()
    return str(arr).strip()


def _parse_string_field(raw: np.ndarray) -> list[str]:
    """
    Convierte un campo de texto de loadmat a una lista de cadenas limpias.

    Maneja tanto matrices de caracteres (char array) como celdas de texto
    (cell array) según cómo las devuelva loadmat.
    """
    if raw.dtype == object:
        # Celda MATLAB: cada elemento plano es una entrada de texto
        return [_flatten_string(item) for item in raw.flat]
    if raw.dtype.kind == "U" and raw.dtype.itemsize > 4:
        # Array de strings completos (ej. dtype '<U15'): cada elemento es
        # una cadena ya formada, no un carácter suelto
        return [str(item).strip() for item in raw.flat]
    # Array de caracteres individuales (dtype 'U1' o 'S1'):
    # unir los caracteres de cada fila para reconstruir la cadena
    if raw.ndim == 1:
        return ["".join(str(c) for c in raw).strip()]
    return ["".join(str(c) for c in row).strip() for row in raw]


def read_signals(pth: str) -> dict:
    """
    Lee un archivo .mat de LabChart y extrae las señales por canal y bloque.

    Parámetros
    ----------
    pth : str
        Ruta absoluta al archivo .mat con los campos estándar de exportación
        LabChart: data, datastart, dataend, samplerate, titles, unittext,
        unittextmap.

    Retorna
    -------
    dict con las siguientes claves:
        signals     : list[list[np.ndarray]] de forma [ncanales][nbloques],
                      cada elemento es un array 1D con las muestras del segmento.
        nchannels   : int, número de canales.
        nblocks     : int, número de bloques de grabación.
        samplerate  : np.ndarray de forma (ncanales, nbloques) con la
                      frecuencia de muestreo de cada segmento.
        titles      : list[str] con el nombre de cada canal.
        unittext    : list[str] con las unidades disponibles.
        unittextmap : np.ndarray de forma (ncanales, nbloques) con el índice
                      en unittext correspondiente a cada segmento.

    Lanza
    -----
    FileNotFoundError
        Si el archivo indicado no existe.
    """
    if not os.path.isfile(pth):
        raise FileNotFoundError(
            f"No se encontró el archivo: {pth}\n"
            "Formato esperado: /ruta/absoluta/a/archivo.mat"
        )

    mat = scipy.io.loadmat(pth, squeeze_me=False)

    # Vector continuo de muestras — aplanar a 1D independientemente del shape
    data: np.ndarray = mat["data"].ravel()

    # Matrices de índices de inicio y fin (1-indexadas, convención MATLAB)
    # Shape esperado: (ncanales, nbloques)
    datastart: np.ndarray = mat["datastart"]
    dataend: np.ndarray = mat["dataend"]

    nchannels: int = int(datastart.shape[0])
    nblocks: int = int(datastart.shape[1])

    # Extraer cada segmento convirtiendo índices MATLAB (base 1, inclusivo)
    # a slices de Python (base 0, extremo derecho exclusivo):
    #   inicio_python = datastart[i,j] - 1
    #   fin_python    = dataend[i,j]       (equivale a dataend[i,j]-1 + 1)
    signals: list[list[np.ndarray]] = [
        [
            data[int(datastart[i, j]) - 1 : int(dataend[i, j])]
            for j in range(nblocks)
        ]
        for i in range(nchannels)
    ]

    return {
        "signals": signals,
        "nchannels": nchannels,
        "nblocks": nblocks,
        "samplerate": mat["samplerate"].astype(float),
        "titles": _parse_string_field(mat["titles"]),
        "unittext": _parse_string_field(mat["unittext"]),
        "unittextmap": mat["unittextmap"],
    }


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Cargando: {EXAMPLE_FILE}\n")
    sdata = read_signals(EXAMPLE_FILE)

    nch = sdata["nchannels"]
    nbl = sdata["nblocks"]
    sr = sdata["samplerate"]

    print(f"Canales : {nch}")
    print(f"Bloques : {nbl}")

    tasas_unicas = np.unique(sr)
    print(f"Frecuencias de muestreo únicas: {tasas_unicas} Hz\n")

    for i, titulo in enumerate(sdata["titles"]):
        # Índice de unidad del primer bloque para representar el canal
        idx_unidad = int(sdata["unittextmap"][i, 0]) - 1
        unidad = sdata["unittext"][idx_unidad] if idx_unidad >= 0 else "?"
        print(f"Canal {i + 1}: '{titulo}'  [{unidad}]")
        for j in range(nbl):
            seg = sdata["signals"][i][j]
            duracion = len(seg) / sr[i, j]
            print(f"  Bloque {j + 1}: {seg.shape[0]} muestras  ({duracion:.2f} s)")
        print()
