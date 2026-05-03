# Evaluación de la Respuesta Broncodilatadora en Pacientes Asmáticos mediante el Análisis de Sonidos Respiratorios

---

## 1. Descripción del proyecto

### 1.1 Contexto clínico

El asma es una enfermedad inflamatoria crónica de las vías respiratorias caracterizada por episodios recurrentes de obstrucción bronquial, disnea, sibilancias y tos. En los pacientes asmáticos, la obstrucción al flujo aéreo se produce por la combinación de edema de la mucosa, hipersecreción de moco y broncoespasmo. Aunque este estrechamiento es frecuentemente reversible, su evaluación precisa resulta fundamental para el diagnóstico, la estratificación de la gravedad y el seguimiento terapéutico.

La respuesta broncodilatadora (BDR) es la mejora funcional observada en un paciente tras la administración de un fármaco broncodilatador de corta acción, habitualmente salbutamol. Clínicamente, la BDR se cuantifica mediante espirometría: se considera positiva (BDR+) cuando el volumen espiratorio forzado en el primer segundo (FEV₁) aumenta un 12 % o más respecto al valor basal, con una mejora absoluta de al menos 200 mL. Una BDR positiva indica reversibilidad significativa de la obstrucción bronquial y tiene implicaciones tanto diagnósticas como terapéuticas. La espirometría, no obstante, requiere la colaboración activa del paciente, equipamiento especializado y personal entrenado para su correcta interpretación.

En este contexto, los sonidos respiratorios se presentan como un biomarcador digital potencialmente complementario. Estos sonidos son generados por el flujo turbulento del aire en las vías respiratorias y contienen información acústica directamente relacionada con el grado de obstrucción bronquial. Entre los sonidos respiratorios de interés clínico se encuentran los sonidos continuos adventicios (CAS, del inglés *Continuous Adventitious Sounds*), cuya manifestación más característica son las sibilancias (*wheezes*). Las sibilancias se producen por el fenómeno de flutter en las paredes de las vías aéreas parcialmente obstruidas: cuando la luz bronquial se estrecha críticamente, las paredes entran en vibración cuasi-sinusoidal a una frecuencia relativamente estable, comprendida generalmente entre 100 y 1.000 Hz. La presencia y abundancia de sibilancias se correlaciona con el grado de obstrucción bronquial, lo que sugiere que su conteo cuantitativo podría emplearse como indicador de la reversibilidad tras la broncodilatación, de forma no invasiva y sin requerir maniobras espirométricas forzadas.

### 1.2 Objetivos del proyecto

Los objetivos del proyecto, formulados por el profesorado de la asignatura, son los siguientes:

1. Clasificar las señales de sonidos respiratorios en normales y CAS.
2. Explorar el uso del número de CAS como biomarcador digital para evaluar la respuesta broncodilatadora.

### 1.3 Dataset

El conjunto de datos comprende 28 sujetos: 23 pacientes diagnosticados de asma (P1–P23) y 5 controles sanos (C1–C5). Todos los sujetos realizaron 6 maniobras respiratorias controladas: las tres primeras en condición basal (pre-broncodilatación, pre-BD) y las tres últimas tras la administración del broncodilatador (post-broncodilatación, post-BD). Las señales de audio fueron adquiridas simultáneamente con 2 canales de micrófono por maniobra. El dataset resultante contiene un total de 14.900 segmentos de audio individuales.

La distribución de los sujetos según la etiqueta de respuesta broncodilatadora es la siguiente:

| Grupo | N.º sujetos | N.º segmentos | % del total |
|---|---|---|---|
| Pacientes BDR+ | 9 | 4.940 | 33,2 % |
| Pacientes BDR− | 14 | 7.076 | 47,5 % |
| Controles sanos | 5 | 2.884 | 19,3 % |
| **Total** | **28** | **14.900** | **100,0 %** |

Los nueve pacientes clasificados como BDR+ son: P2, P6, P7, P8, P9, P10, P11, P12 y P14. Los controles sanos no presentan patología bronquial y están clasificados como BDR−.

### 1.4 Sistema de adquisición

Las señales acústicas se adquirieron mediante micrófonos piezoeléctricos colocados en la espalda del paciente, en dos posiciones anatómicas distintas del hemitórax izquierdo:

- **Canal 1 — Canal inferior** (*RS bottom left*): situado en la zona baja del pulmón izquierdo, sensible a los sonidos generados en las vías aéreas inferiores.
- **Canal 2 — Canal superior** (*RS top left*): situado en la zona alta del pulmón izquierdo, sensible a los sonidos generados en las vías aéreas superiores.

El sistema de adquisición empleado es el software LabChart, que captura las señales digitalizadas a una frecuencia de muestreo de 12.500 Hz. Las señales son exportadas en formato `.mat` compatible con MATLAB, conteniendo en un único archivo continuo los datos de todos los canales y bloques de grabación, junto con las matrices de índices que delimitan el inicio y el fin de cada segmento y los metadatos de calibración.

---

## 2. Pipeline de procesamiento

El procesamiento de las señales sigue un pipeline de cuatro etapas secuenciales, implementadas en cuatro módulos de Python independientes ubicados en el directorio `src/`. Cada módulo expone funciones de biblioteca que son importadas por el módulo siguiente, de forma que la cadena completa puede ejecutarse tanto de manera progresiva como de forma integrada desde el script de construcción del dataset. La arquitectura modular facilita la reutilización de cada etapa de forma aislada en análisis exploratorios o notebooks.

```
  PX.mat / CX.mat
       │
       ▼
  ┌─────────────────┐
  │ Step 1          │  read_signals()
  │ Lectura         │  → signals [2 × 6]
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Step 2          │  preprocess_signal()
  │ Preprocesado    │  → señal filtrada @ 4.000 Hz
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Step 3          │  load_markers() + segment_signal()
  │ Segmentación    │  → {inspiracion, espiracion}
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Step 4          │  build_dataset()
  │ Dataset         │  → 14.900 segmentos + 4 vectores
  └─────────────────┘
```

---

## 3. Step 1 — Lectura de señales

### 3.1 Descripción

El módulo `step1_read_signals.py` implementa la función `read_signals()`, encargada de cargar los archivos `.mat` generados por LabChart y extraer de forma estructurada las señales de cada canal y bloque de grabación. La función recibe como argumento la ruta absoluta al archivo de señales de un sujeto y devuelve un diccionario con las señales organizadas por canal y maniobra, junto con los metadatos del archivo.

### 3.2 Estructura del archivo .mat

Los archivos `.mat` exportados desde LabChart contienen un vector continuo de muestras (`data`) con todos los canales concatenados, junto con matrices de índices que delimitan el inicio y el fin de cada segmento. La función `read_signals()` devuelve un diccionario con los siguientes campos:

| Campo | Tipo | Descripción |
|---|---|---|
| `signals` | `list[list[np.ndarray]]` | Señales organizadas como `[canal][bloque]`; cada elemento es un array 1D de muestras en float64. |
| `nchannels` | `int` | Número de canales de audio (2 en este dataset). |
| `nblocks` | `int` | Número de bloques de grabación o maniobras (6 en este dataset). |
| `samplerate` | `np.ndarray (nch × nbl)` | Frecuencia de muestreo de cada segmento en Hz. |
| `titles` | `list[str]` | Nombre de cada canal según la configuración de LabChart. |
| `unittext` | `list[str]` | Unidades físicas disponibles en el archivo. |
| `unittextmap` | `np.ndarray (nch × nbl)` | Índice en `unittext` correspondiente a cada segmento. |

### 3.3 Implementación

El archivo `.mat` se carga mediante `scipy.io.loadmat` con el parámetro `squeeze_me=False` para preservar las dimensiones originales de todos los arrays. La extracción de las señales individuales requiere convertir los índices de inicio (`datastart`) y fin (`dataend`) del sistema de referencia de MATLAB —basado en 1 e inclusivo en ambos extremos— al sistema de índices de Python —basado en 0 y con extremo derecho exclusivo—. La conversión se realiza de la siguiente forma:

```
inicio_python = datastart[i, j] - 1
fin_python    = dataend[i, j]        # equivale a dataend[i, j] − 1 + 1
```

Los campos de texto del archivo (`titles`, `unittext`) pueden estar codificados como arrays de caracteres individuales o como celdas de texto de MATLAB, dependiendo de la versión exportada desde LabChart. La función auxiliar `_parse_string_field()` unifica ambos formatos en una lista de cadenas de texto limpias, mientras que `_flatten_string()` se encarga de desanidar los arrays de objetos que `loadmat` puede introducir al procesar celdas MATLAB.

### 3.4 Ejemplo de salida

Al procesar el archivo `Data/P1.mat`, el script imprime el siguiente resumen:

```
Canales : 2
Bloques : 6
Frecuencias de muestreo únicas: [12500.] Hz

Canal 1: 'RS bottom left'  [V]
  Bloque 1: ~625.000 muestras  (~50 s)
  Bloque 2: ~625.000 muestras  (~50 s)
  ...
Canal 2: 'RS top left'  [V]
  Bloque 1: ~625.000 muestras  (~50 s)
  ...
```

Cada bloque corresponde a una maniobra respiratoria completa de aproximadamente 50 a 60 segundos de duración.

---

## 4. Step 2 — Preprocesado

### 4.1 Descripción

El módulo `step2_preprocessing.py` implementa la cadena completa de preprocesado de las señales de sonido respiratorio. El objetivo es eliminar las componentes indeseadas —ruido de movimiento corporal e interferencia eléctrica de red— mientras se preserva íntegra la banda de frecuencias en la que se encuentran los sonidos de interés clínico. La función pública `preprocess_signal()` encapsula los tres pasos de la cadena y puede importarse directamente por los módulos posteriores del pipeline.

### 4.2 Cadena de preprocesado

El preprocesado se compone de tres etapas aplicadas secuencialmente sobre cada señal cruda.

**1. Remuestreo de 12.500 Hz a 4.000 Hz**

La señal original se remuestrea de 12.500 Hz a 4.000 Hz mediante `scipy.signal.resample_poly` con factores enteros irreducibles `up = 8` y `down = 25`, obtenidos a partir de la fracción 4000/12500 = 8/25. Este método aplica internamente un filtro antialiasing antes del diezmado, garantizando la ausencia de solapamiento espectral. La reducción de la frecuencia de muestreo está justificada porque los CAS se encuentran en la banda 100–1.000 Hz; una frecuencia de 4.000 Hz proporciona una frecuencia de Nyquist de 2.000 Hz, suficiente para representar fielmente las componentes de interés con amplio margen.

**2. Filtro paso banda Butterworth**

Se aplica un filtro paso banda Butterworth de orden 8 con frecuencias de corte en 70 Hz y 1.900 Hz, implementado en forma de secciones de segundo orden (SOS) y aplicado con fase cero mediante `scipy.signal.sosfiltfilt`. La banda de paso elimina las componentes de muy baja frecuencia (< 70 Hz) debidas al movimiento corporal y al ruido mecánico de la adquisición, así como las componentes de alta frecuencia (> 1.900 Hz) no relacionadas con los sonidos respiratorios. El orden elevado (8) proporciona una transición espectral abrupta que minimiza la distorsión dentro de la banda de paso.

**3. Filtro comb notch (interferencia de red eléctrica)**

La interferencia de la red eléctrica a 50 Hz (estándar europeo) y sus armónicos se elimina mediante un banco de filtros notch IIR aplicados secuencialmente, uno por cada armónico presente en la banda útil. Para cada frecuencia f₀ = k × 50 Hz con k = 1, 2, …, 39 (abarcando el rango de 50 Hz hasta el límite de Nyquist de 2.000 Hz), se diseña un filtro notch de segundo orden con ancho de banda constante de 1 Hz mediante `scipy.signal.iirnotch`, lo que equivale a un factor de calidad Q = f₀/BW = k × 50. El filtro se convierte a forma SOS y se aplica con fase cero. En total se aplican 39 filtros notch de forma encadenada.

### 4.3 Parámetros técnicos

| Parámetro | Valor | Descripción |
|---|---|---|
| `FS_ORIGINAL` | 12.500 Hz | Frecuencia de muestreo de adquisición original |
| `FS_TARGET` | 4.000 Hz | Frecuencia de muestreo tras el remuestreo |
| `BP_LOW` | 70 Hz | Límite inferior del filtro paso banda |
| `BP_HIGH` | 1.900 Hz | Límite superior del filtro paso banda |
| `BP_ORDER` | 8 | Orden del filtro Butterworth |
| `NOTCH_FUND` | 50 Hz | Frecuencia fundamental del banco comb notch |
| `NOTCH_BW` | 1 Hz | Ancho de banda por componente notch |

### 4.4 Resultados visuales

El módulo genera cinco figuras guardadas en `outputs/figures/step2/`:

- **`fig1_senal_cruda_vs_preprocesada.png`**: Comparativa temporal de los primeros 5 s de la señal cruda (12.500 Hz) frente a la señal preprocesada (4.000 Hz) para un sujeto BDR+ (P1) y un sujeto BDR− (P19), dispuestos en una cuadrícula 2×2.
- **`fig2_pasos_preprocesado_BDRpos.png`**: Evolución temporal de la señal del sujeto BDR+ (P1) tras cada una de las cuatro etapas del preprocesado, mostrando el efecto acumulativo de cada paso sobre la morfología de la señal.
- **`fig3_pasos_preprocesado_BDRneg.png`**: Misma representación para el sujeto BDR− (P19).
- **`fig4_psd_antes_despues.png`**: Densidad espectral de potencia (PSD) en dB calculada por el método de Welch, comparando la señal cruda y la señal preprocesada para ambos sujetos. Se superponen líneas de referencia en las frecuencias de corte del paso banda y en los armónicos eliminados por el comb notch.
- **`fig5_espectrograma_antes_despues.png`**: Espectrograma tiempo-frecuencia en escala logarítmica (ventana de 256 muestras, solapamiento del 50 %), antes y después del preprocesado, para los mismos dos sujetos.

---

## 5. Step 3 — Segmentación

### 5.1 Descripción

El módulo `step3_segmentation.py` implementa la segmentación de las señales preprocesadas en ciclos respiratorios individuales, separando los segmentos de inspiración y espiración. La segmentación se realiza a partir de los archivos de marcadores temporales `tPX.mat` (pacientes) y `tCX.mat` (controles), que contienen los instantes de inicio y fin de cada fase respiratoria en cada maniobra, determinados durante la adquisición.

### 5.2 Estructura de los archivos de marcadores

Cada archivo de marcadores contiene la variable `seg_t`, un array de celdas MATLAB de dimensión (1, 6), en el que cada celda almacena los marcadores temporales de una de las seis maniobras. Los marcadores de cada maniobra presentan la forma (n_ciclos, 4) con las siguientes columnas:

| Columna | Contenido | Unidad |
|---|---|---|
| 0 | Instante de inicio de la inspiración | s |
| 1 | Instante de fin de la inspiración | s |
| 2 | Instante de inicio de la espiración | s |
| 3 | Instante de fin de la espiración | s |

La carga de los marcadores se realiza mediante la función `load_markers()`, que gestiona las posibles capas de anidamiento que `scipy.io.loadmat` introduce al procesar arrays de celdas MATLAB con varias dimensiones de objeto.

### 5.3 Conversión tiempo → índice de muestra

La conversión de los tiempos de marcador (expresados en segundos) a índices de muestra de la señal preprocesada se realiza mediante la expresión:

```
idx = int(round(t_segundos × FS_TARGET))
```

donde `FS_TARGET = 4.000 Hz`. El resultado se pinza al intervalo `[0, N−1]`, siendo `N` la longitud total de la señal preprocesada, para garantizar que los índices de borde no superen los límites del array en presencia de pequeñas discrepancias de redondeo entre la duración del marcador y la duración real de la señal.

### 5.4 Estadísticas de segmentación

El número de ciclos por maniobra y la duración de los segmentos presentan variabilidad inter-sujeto relevante, debida a las diferencias en el patrón ventilatorio de cada individuo. Las estadísticas agregadas derivadas del dataset completo son las siguientes:

| Métrica | Media | Desv. típ. | Rango |
|---|---|---|---|
| Ciclos por maniobra (por sujeto) | 22,2 | 2,3 | 18–25 |
| Duración media de segmento (ms) | 1.361 | 245 | 901–1.721 |
| Desv. típ. intra-sujeto de duración (ms) | 398 | — | 178–614 |

El número total de ciclos en el dataset es 3.725 (suma de todos los ciclos de todos los sujetos y maniobras). Multiplicado por los dos canales de adquisición y las dos fases respiratorias (inspiración y espiración), se obtienen los 14.900 segmentos del dataset (3.725 × 2 canales × 2 fases = 14.900).

### 5.5 Resultados visuales

El módulo genera seis figuras guardadas en `outputs/figures/step3/`:

- **`fig1_segmentacion_maniobra_BDRpos.png`**: Señal completa de la primera maniobra del sujeto BDR+ (P2) con sombreado verde para los segmentos de inspiración y naranja para los de espiración, ilustrando la correspondencia entre los marcadores temporales y la forma de onda.
- **`fig2_segmentacion_maniobra_BDRneg.png`**: Misma representación para el sujeto BDR− (P16).
- **`fig3_segmentos_individuales_BDRpos.png`**: Cuadrícula 2×4 con los primeros cuatro segmentos individuales de inspiración (fila superior) y espiración (fila inferior) extraídos de la primera maniobra del sujeto BDR+ (P2), con indicación de la duración de cada segmento en milisegundos.
- **`fig4_segmentos_individuales_BDRneg.png`**: Misma representación para el sujeto BDR− (P16).
- **`fig5_prebd_vs_postbd_BDRpos.png`**: Comparativa de la maniobra 2 (pre-BD) frente a la maniobra 5 (post-BD) del sujeto BDR+ (P2), que ilustra el posible cambio en la morfología de los ciclos respiratorios tras la administración del broncodilatador.
- **`fig6_prebd_vs_postbd_BDRneg.png`**: Misma representación para el sujeto BDR− (P16).

---

## 6. Step 4 — Construcción del dataset

### 6.1 Descripción

El módulo `step4_dataset.py` integra las tres etapas anteriores para procesar la totalidad de los 28 sujetos y construir el dataset completo de 14.900 segmentos. Para cada segmento se registran cuatro vectores de metadatos que codifican el origen y las condiciones experimentales de la muestra. Estos vectores resultan imprescindibles para las etapas posteriores del proyecto —extracción de características y clasificación—, ya que permiten realizar análisis estratificados por sujeto, condición de broncodilatación, canal de audio y fase respiratoria, así como asegurar una validación cruzada correctamente independizada a nivel de sujeto.

### 6.2 Los cuatro vectores de metadatos

| Vector | Valores posibles | Descripción |
|---|---|---|
| `v_subject` | 1–23 (pacientes), 24–28 (controles) | Identificador numérico del sujeto |
| `v_bd` | 1 = pre-BD, 2 = post-BD | Condición broncodilatadora de la maniobra |
| `v_channel` | 1 = canal inferior, 2 = canal superior | Canal de audio de adquisición |
| `v_phase` | 1 = inspiración, 2 = espiración | Fase del ciclo respiratorio |

Todos los vectores tienen longitud 14.900 y tipo `numpy.int32`.

### 6.3 Orden de construcción

Los segmentos se almacenan siguiendo el orden de iteración especificado a continuación, que garantiza que los segmentos del mismo ciclo sean contiguos en el array y que el recorrido sea determinista y reproducible:

```
para cada sujeto (P1 → P23, C1 → C5):
  para cada canal (0 = inferior, 1 = superior):
    para cada maniobra (0 → 5):
      para cada ciclo en esa maniobra:
        → añadir segmento de inspiración  (v_phase = 1)
        → añadir segmento de espiración   (v_phase = 2)
```

### 6.4 Estadísticas finales del dataset

El dataset completo contiene 14.900 segmentos distribuidos de forma equilibrada entre las condiciones experimentales:

| Condición | N.º segmentos | Porcentaje |
|---|---|---|
| Total | 14.900 | 100,0 % |
| Pacientes BDR+ | 4.940 | 33,2 % |
| Pacientes BDR− | 7.076 | 47,5 % |
| Controles sanos | 2.884 | 19,3 % |
| Pre-BD (maniobras 1–3) | 7.448 | 50,0 % |
| Post-BD (maniobras 4–6) | 7.452 | 50,0 % |
| Canal inferior | 7.450 | 50,0 % |
| Canal superior | 7.450 | 50,0 % |
| Inspiración | 7.450 | 50,0 % |
| Espiración | 7.450 | 50,0 % |

La tabla siguiente recoge el resumen por sujeto exportado en `outputs/results/step4/dataset_summary.csv`:

| Sujeto | Tipo | BDR | Segm. | Pre-BD | Post-BD | Insp. | Esp. | CH1 | CH2 | Dur. media (ms) | Desv. típ. (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | Paciente | BDR− | 440 | 208 | 232 | 220 | 220 | 220 | 220 | 1.302,7 | 283,1 |
| P2 | Paciente | BDR+ | 472 | 232 | 240 | 236 | 236 | 236 | 236 | 1.720,6 | 534,5 |
| P3 | Paciente | BDR− | 608 | 312 | 296 | 304 | 304 | 304 | 304 | 901,1 | 215,7 |
| P4 | Paciente | BDR− | 544 | 276 | 268 | 272 | 272 | 272 | 272 | 1.039,3 | 248,2 |
| P5 | Paciente | BDR− | 568 | 272 | 296 | 284 | 284 | 284 | 284 | 1.110,5 | 271,0 |
| P6 | Paciente | BDR+ | 564 | 280 | 284 | 282 | 282 | 282 | 282 | 1.401,5 | 497,6 |
| P7 | Paciente | BDR+ | 580 | 292 | 288 | 290 | 290 | 290 | 290 | 1.579,4 | 540,2 |
| P8 | Paciente | BDR+ | 496 | 236 | 260 | 248 | 248 | 248 | 248 | 1.242,5 | 394,9 |
| P9 | Paciente | BDR+ | 572 | 288 | 284 | 286 | 286 | 286 | 286 | 1.564,1 | 489,1 |
| P10 | Paciente | BDR+ | 576 | 288 | 288 | 288 | 288 | 288 | 288 | 1.565,1 | 481,0 |
| P11 | Paciente | BDR+ | 576 | 288 | 288 | 288 | 288 | 288 | 288 | 1.613,5 | 434,7 |
| P12 | Paciente | BDR+ | 576 | 288 | 288 | 288 | 288 | 288 | 288 | 1.590,4 | 534,2 |
| P13 | Paciente | BDR− | 540 | 272 | 268 | 270 | 270 | 270 | 270 | 1.230,0 | 396,4 |
| P14 | Paciente | BDR+ | 528 | 252 | 276 | 264 | 264 | 264 | 264 | 1.394,8 | 371,5 |
| P15 | Paciente | BDR− | 576 | 288 | 288 | 288 | 288 | 288 | 288 | 1.604,0 | 482,5 |
| P16 | Paciente | BDR− | 576 | 288 | 288 | 288 | 288 | 288 | 288 | 1.595,3 | 437,3 |
| P17 | Paciente | BDR− | 464 | 236 | 228 | 232 | 232 | 232 | 232 | 1.440,6 | 286,9 |
| P18 | Paciente | BDR− | 464 | 224 | 240 | 232 | 232 | 232 | 232 | 1.087,0 | 324,7 |
| P19 | Paciente | BDR− | 492 | 256 | 236 | 246 | 246 | 246 | 246 | 973,3 | 177,6 |
| P20 | Paciente | BDR− | 420 | 204 | 216 | 210 | 210 | 210 | 210 | 1.126,8 | 290,7 |
| P21 | Paciente | BDR− | 440 | 224 | 216 | 220 | 220 | 220 | 220 | 1.115,6 | 248,2 |
| P22 | Paciente | BDR− | 468 | 256 | 212 | 234 | 234 | 234 | 234 | 1.099,0 | 340,7 |
| P23 | Paciente | BDR− | 476 | 248 | 228 | 238 | 238 | 238 | 238 | 1.060,2 | 287,3 |
| C1 | Control | BDR− | 572 | 284 | 288 | 286 | 286 | 286 | 286 | 1.608,2 | 440,9 |
| C2 | Control | BDR− | 576 | 288 | 288 | 288 | 288 | 288 | 288 | 1.381,0 | 435,1 |
| C3 | Control | BDR− | 580 | 292 | 288 | 290 | 290 | 290 | 290 | 1.602,5 | 613,6 |
| C4 | Control | BDR− | 580 | 288 | 292 | 290 | 290 | 290 | 290 | 1.572,9 | 552,5 |
| C5 | Control | BDR− | 576 | 288 | 288 | 288 | 288 | 288 | 288 | 1.594,8 | 523,6 |

### 6.5 Resultados visuales

El módulo genera cuatro figuras guardadas en `outputs/figures/step4/`:

- **`fig1_segmentos_por_sujeto.png`**: Diagrama de barras con el número de segmentos por sujeto (28 barras), codificado por color según el grupo clínico (verde = BDR+, azul = BDR−, gris = control). Una línea horizontal de trazos indica la media global.
- **`fig2_distribucion_vectores.png`**: Cuadrícula 2×2 de diagramas de sectores que muestra la distribución de los cuatro vectores de metadatos: pre-BD frente a post-BD, canal inferior frente a canal superior, inspiración frente a espiración, y pacientes frente a controles.
- **`fig3_duracion_segmentos.png`**: Dos subgráficos apilados verticalmente con la duración media de los segmentos de inspiración (superior) y espiración (inferior) por sujeto, con barras de error para la desviación típica y codificación de color por grupo clínico.
- **`fig4_heatmap_segmentos.png`**: Mapa de calor de dimensión 28 × 6 (sujetos × maniobras) con el número de segmentos totales por celda, colormap YlOrRd y anotación numérica en cada celda. Una línea vertical de trazos separa las maniobras pre-BD (1–3) de las post-BD (4–6).

---

## 7. Estructura del repositorio

La estructura de directorios del proyecto es la siguiente:

```
Project/
├── Data/                           # Señales y marcadores (no incluidos en el repositorio)
│   ├── P1.mat  …  P23.mat          # Señales de los 23 pacientes asmáticos
│   ├── C1.mat  …  C5.mat           # Señales de los 5 controles sanos
│   ├── tP1.mat … tP23.mat          # Marcadores temporales de pacientes
│   ├── tC1.mat … tC5.mat           # Marcadores temporales de controles
│   └── database/
│       └── subject_metadata.csv    # Metadatos de los 28 sujetos (BDR, sexo, tipo)
├── docs/                           # Documentación de referencia del proyecto
│   ├── Base de datos proyecto.pdf  # Descripción de la base de datos
│   ├── Presentación proyecto.pdf   # Presentación del proyecto
│   └── read_signals.m              # Script MATLAB de referencia para lectura de señales
├── notebooks/                      # Notebooks de exploración y análisis interactivo
├── outputs/
│   ├── figures/
│   │   ├── step2/                  # 5 figuras del preprocesado
│   │   ├── step3/                  # 6 figuras de la segmentación
│   │   └── step4/                  # 4 figuras del dataset
│   └── results/
│       └── step4/
│           ├── dataset.npz         # Vectores de metadatos en formato comprimido NumPy
│           ├── dataset_summary.csv # Resumen estadístico por sujeto
│           └── segment_lengths.npy # Longitud en muestras de cada uno de los 14.900 segmentos
├── src/
│   ├── step1_read_signals.py       # Lectura de archivos .mat exportados desde LabChart
│   ├── step2_preprocessing.py      # Remuestreo, filtro paso banda y filtro comb notch
│   ├── step3_segmentation.py       # Segmentación de ciclos respiratorios mediante marcadores
│   └── step4_dataset.py            # Construcción del dataset completo con vectores de metadatos
├── tests/                          # Suite de pruebas unitarias
├── pytest.ini                      # Configuración de pytest y cobertura de código
├── README.md                       # Descripción básica del repositorio
├── DOCUMENTACION.md                # Este documento
└── requirements.txt                # Dependencias Python del proyecto
```

---

## 8. Requisitos e instalación

### 8.1 Requisitos del sistema

El proyecto requiere Python 3.10 o superior. Las dependencias del entorno virtual son las siguientes:

| Paquete | Versión mínima | Uso principal |
|---|---|---|
| `numpy` | ≥ 1.26 | Arrays numéricos y operaciones vectoriales |
| `scipy` | ≥ 1.12 | Lectura de `.mat`, filtrado digital, remuestreo y estadística |
| `matplotlib` | ≥ 3.8 | Generación de todas las figuras |
| `seaborn` | ≥ 0.13 | Visualizaciones estadísticas complementarias |
| `pandas` | ≥ 2.1 | Manipulación de datos tabulares |
| `scikit-learn` | ≥ 1.4 | Clasificación y validación cruzada (etapas posteriores) |
| `imbalanced-learn` | ≥ 0.12 | Tratamiento de clases desequilibradas (etapas posteriores) |
| `jupyter` | ≥ 1.0 | Notebooks de exploración interactiva |
| `pytest` | ≥ 8.0 | Ejecución de la suite de pruebas unitarias |
| `pytest-cov` | ≥ 5.0 | Informe de cobertura de código |
| `ruff` | ≥ 0.4 | Análisis estático y formato de código |

### 8.2 Instalación

```bash
git clone <url-del-repositorio>
cd Project
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 8.3 Ejecución

Cada script del pipeline puede ejecutarse de forma independiente desde el directorio raíz del proyecto. Los scripts `step1`, `step2` y `step3` procesan sujetos de ejemplo predefinidos con fines de visualización y verificación. El script `step4_dataset.py` procesa los 28 sujetos completos, guarda los resultados numéricos en `outputs/results/step4/` y las figuras en `outputs/figures/step4/`.

```bash
python src/step1_read_signals.py
python src/step2_preprocessing.py
python src/step3_segmentation.py
python src/step4_dataset.py
```

Las rutas a los archivos de datos se definen como constantes al inicio de cada módulo, por lo que pueden modificarse sin necesidad de alterar la lógica principal del procesamiento. Todos los directorios de salida se crean automáticamente si no existen en el momento de la ejecución.

---

## 9. Notas sobre los datos

Los archivos de señales en formato `.mat` no están incluidos en el repositorio por motivos de privacidad, al tratarse de datos de pacientes. Para poder ejecutar el pipeline completo, los archivos de señales y marcadores deben ubicarse en la carpeta `Data/` del directorio raíz del proyecto, siguiendo estrictamente la nomenclatura indicada a continuación:

- Señales de pacientes: `Data/P1.mat`, `Data/P2.mat`, …, `Data/P23.mat`
- Señales de controles: `Data/C1.mat`, `Data/C2.mat`, …, `Data/C5.mat`
- Marcadores de pacientes: `Data/tP1.mat`, `Data/tP2.mat`, …, `Data/tP23.mat`
- Marcadores de controles: `Data/tC1.mat`, `Data/tC2.mat`, …, `Data/tC5.mat`
- Metadatos de sujetos: `Data/database/subject_metadata.csv`

Las figuras generadas se guardan automáticamente en `outputs/figures/` y los resultados numéricos en `outputs/results/`. Los archivos de salida más relevantes para las etapas posteriores del proyecto son `outputs/results/step4/dataset.npz` (vectores de metadatos), `outputs/results/step4/segment_lengths.npy` (longitudes de segmento en muestras) y `outputs/results/step4/dataset_summary.csv` (resumen estadístico por sujeto).
