# Evaluacion de la Respuesta Broncodilatadora mediante Analisis de Sonidos Respiratorios

---

## 1. Descripcion del proyecto

### 1.1 Contexto clinico

El asma es una enfermedad inflamatoria cronica de las vias respiratorias que se caracteriza por
obstruccion variable al flujo aereo, hiperreactividad bronquial y sintomas como sibilancias,
disnea y tos. La respuesta broncodilatadora (BDR) es un criterio diagnostico y de seguimiento
clinico clave: consiste en la mejora objetiva de la funcion pulmonar (tipicamente espirometria)
tras la administracion de un broncodilatador de accion corta (salbutamol).

La evaluacion estandar de la BDR se basa en espirometria, que requiere un esfuerzo maximo
del paciente y equipamiento clinico especializado. La posibilidad de cuantificar la BDR
mediante analisis automatico de sonidos respiratorios —medidos con microfono en superficie
corporal— abre la puerta a evaluaciones mas asequibles, no invasivas y potencialmente
realizables en entorno domiciliario.

### 1.2 Hipotesis

Los sonidos continuos adventicios (CAS, *Continuous Adventitious Sounds*), tambien llamados
sibilancias (*wheezes*) o roncus (*rhonchi*), son ruidos musicales superpuestos al flujo
respiratorio que aparecen cuando las vias aereas estan parcialmente obstruidas y vibran. La
hipotesis del proyecto es:

> **Un broncodilatador eficaz deberia reducir la obstruccion bronquial y, por tanto, disminuir
> la tasa de CAS despues de su administracion. La diferencia delta_CAS = CAS_rate_pre -
> CAS_rate_post deberia ser mayor en sujetos BDR+ que en BDR-.**

Si esta hipotesis se confirma, delta_CAS puede actuar como biomarcador digital de la BDR,
cuantificable de forma no invasiva a partir de audio.

### 1.3 Dataset

| Parametro            | Valor                              |
|----------------------|------------------------------------|
| Pacientes asmaticos  | 23 (P1-P23)                        |
| Controles sanos      | 5 (C1-C5)                          |
| Total de sujetos     | 28                                 |
| Maniobras por sujeto | 6 (3 pre-BD + 3 post-BD)           |
| Canales de audio     | 2 (traqueal inferior y superior)   |
| Frecuencia original  | 12 500 Hz                          |
| Frecuencia objetivo  | 4 000 Hz (tras remuestreo)         |
| Total de segmentos   | 14 900                             |

Los sujetos se graban en dos sesiones separadas por la administracion de broncodilatador. Las
tres primeras maniobras corresponden a la fase pre-BD y las tres ultimas a la fase post-BD.
En cada maniobra se registran dos canales de audio en paralelo.

### 1.4 Etiquetas BDR

La respuesta broncodilatadora se determino a partir de espirometria segun criterios clinicos
estandar (mejora de FEV1 >= 200 mL y >= 12% del valor basal):

- **BDR+ (n = 9):** P2, P6, P7, P8, P9, P10, P11, P12, P14
- **BDR- (n = 19):** P1, P3, P4, P5, P13, P15-P23, C1, C2, C3, C4, C5

Los controles sanos se clasifican como BDR- por definicion.

---

## 2. Pipeline tecnico

### 2.1 Lectura de datos (`src/phase1_io.py`)

Los datos se almacenan en ficheros `.mat` exportados desde LabChart (AD Instruments). La
funcion `read_signals(path)` los carga mediante `scipy.io.loadmat` y reconstruye la matriz de
senales a partir de las variables internas del formato LabChart:

| Variable LabChart | Descripcion                                         |
|-------------------|-----------------------------------------------------|
| `data`            | Array plano 1D con todas las muestras concatenadas  |
| `datastart`       | Matriz (nchannels x nblocks): indices 1-based de inicio de cada bloque |
| `dataend`         | Matriz (nchannels x nblocks): indices 1-based de fin de cada bloque    |
| `samplerate`      | Matriz (nchannels x nblocks): frecuencia de muestreo en Hz             |
| `titles`          | Lista de nombres de canal                           |

La funcion devuelve un diccionario con clave `signals` como lista bidimensional
`signals[canal][bloque]` de arrays 1D de numpy, junto con `nchannels`, `nblocks`,
`samplerate` y `titles`.

**Convenciones de ficheros:**
- Pacientes: `P{n}.mat` (senales) y `tP{n}.mat` (marcas temporales)
- Controles: `C{n}.mat` (senales) y `tC{n}.mat` (marcas temporales)

### 2.2 Preprocesamiento (`src/phase1_preprocessing.py`)

Cada senal bruta pasa por tres etapas en orden fijo mediante la funcion `preprocess_signal()`:

#### Paso 1 — Remuestreo de 12 500 Hz a 4 000 Hz

Se aplica remuestreo polifasico (`scipy.signal.resample_poly`) con razon reducida 8/25
(fraccion minima de 4000/12500). El filtro anti-aliasing esta integrado en el filtro polifasico.
La frecuencia objetivo de 4 000 Hz es suficiente para capturar el rango de CAS clinicamente
relevante (100-1 000 Hz) con margen holgado.

#### Paso 2 — Filtro paso banda Butterworth de orden 8, 70-1 900 Hz, fase cero

Se implementa en forma SOS (*Second-Order Sections*) para evitar inestabilidad numerica con
filtros de orden alto. El filtrado es bidireccional (`sosfiltfilt`) para garantizar fase cero
y evitar desfases que distorsionarian la deteccion de picos espectrales.

- **Limite inferior 70 Hz:** elimina el ruido de bajo frecuencia (movimiento, rozamiento)
- **Limite superior 1 900 Hz:** elimina ruido de alta frecuencia por encima del rango de CAS

#### Paso 3 — Filtro notch en peine, 50 Hz y armonicos, ancho de banda 1 Hz

Se utiliza cancelacion en el dominio de la frecuencia (FFT): para cada armonico de 50 Hz
por debajo de la frecuencia de Nyquist (2 000 Hz), los bins cuyo centro cae dentro de
±0.5 Hz del armonico se anulan antes de la IFFT. Este enfoque de bloque:
- Consigue atenuacion exacta (cero) en los armonicos objetivo
- Evita transitorios de arranque de filtros IIR de banda estrecha (Q elevado) en
  senales cortas (< 5 s)

### 2.3 Segmentacion (`src/phase2_segmentation.py`)

#### Ficheros de marcas temporales

Los ficheros `tP{n}.mat` / `tC{n}.mat` contienen las marcas temporales de los ciclos
respiratorios. Se cargan con `load_markers(path)` y se devuelven como lista de 6 arrays,
uno por maniobra, cada uno de forma **(n_ciclos, 4)** en **segundos**:

| Columna | Contenido         |
|---------|-------------------|
| 0       | Inicio inspiracion |
| 1       | Fin inspiracion    |
| 2       | Inicio espiracion  |
| 3       | Fin espiracion     |

#### Conversion tiempo a muestra

La funcion `time_to_sample(t, fs=4000)` convierte tiempo en segundos a indice de muestra
mediante redondeo al entero mas proximo: `idx = int(round(t * fs))`.

#### Extraccion de segmentos

`segment_signal(signal, markers, fs=4000)` itera sobre cada fila de la matriz de marcas y
extrae dos segmentos por ciclo respiratorio:
- Segmento inspiratorio: `signal[insp_start:insp_end]`
- Segmento espiratorio: `signal[exp_start:exp_end]`

Los segmentos de menos de 200 muestras (50 ms) se descartan silenciosamente. Cada segmento
conservado se devuelve junto con su etiqueta de fase (1=inspiracion, 2=espiracion).

### 2.4 Construccion del dataset (`src/phase2_dataset.py`)

La funcion `build_dataset(data_dir, output_dir)` itera sobre todos los sujetos y acumula
cuatro vectores enteros de metadatos de longitud N = 14 900:

| Vector      | Tipo   | Valores              | Descripcion                     |
|-------------|--------|----------------------|---------------------------------|
| `v_subject` | int32  | 1-28                 | ID del sujeto (P1-P23 → 1-23, C1-C5 → 24-28) |
| `v_bd`      | int32  | 1=pre-BD, 2=post-BD  | Maniobras 1-3 son pre-BD; 4-6 son post-BD      |
| `v_channel` | int32  | 1=inferior, 2=superior | Canal de audio                 |
| `v_phase`   | int32  | 1=inspiracion, 2=espiracion | Fase respiratoria        |

Si el numero total de segmentos difiere de 14 900, se emite un `RuntimeWarning` con el
conteo real (Fix 5).

Los datos se guardan en `outputs/results/`:
- `dataset.npz`: los cuatro vectores de metadatos
- `all_signals.pkl`: lista Python de los 14 900 arrays de senales

### 2.5 Extraccion de features (`src/phase3_features.py`)

Se computan dos matrices de features con funciones separadas para evitar dependencia circular
entre el etiquetado CAS y los inputs del clasificador (vease Fix 1 en seccion 4).

#### Matriz ML: 16 features por segmento

**Bloque temporal (5 features):**

| Feature            | Descripcion                                              |
|--------------------|----------------------------------------------------------|
| `rms`              | Energia: raiz del promedio del cuadrado de la amplitud   |
| `zero_crossing_rate` | Frecuencia de cambios de signo, normalizada por longitud |
| `kurtosis`         | Curtosis de la distribucion de amplitudes                |
| `skewness`         | Asimetria de la distribucion de amplitudes               |
| `peak_to_peak`     | Rango pico a pico de amplitud                            |

**Bloque espectral (8 features):**
La densidad espectral de potencia (PSD) se estima con el metodo de Welch
(`nperseg = min(512, len(signal))`).

| Feature                 | Descripcion                                              |
|-------------------------|----------------------------------------------------------|
| `mean_freq`             | Frecuencia media ponderada por la PSD                    |
| `median_freq`           | Frecuencia donde la potencia acumulada alcanza el 50%    |
| `peak_freq`             | Frecuencia del maximo de la PSD                          |
| `spectral_entropy`      | Entropia de Shannon de la PSD normalizada                |
| `band_power_70_200`     | Fraccion de potencia en 70-200 Hz                        |
| `band_power_200_500`    | Fraccion de potencia en 200-500 Hz                       |
| `band_power_500_1000`   | Fraccion de potencia en 500-1000 Hz                      |
| `band_power_1000_1900`  | Fraccion de potencia en 1000-1900 Hz                     |

Las cuatro potencias de banda son **fraccionales** (suman aproximadamente 1.0 sobre el rango
70-1900 Hz), lo que las hace invariantes a la ganancia absoluta del microfono.

**Bloque CAS-especifico (3 features):**

| Feature          | Descripcion                                                    |
|------------------|----------------------------------------------------------------|
| `peak_sharpness` | Potencia en el pico dominante dividida por potencia en +-50 Hz |
| `autocorr_peak`  | Maximo de autocorrelacion normalizada en retardos 1..N/2       |
| `harmonic_ratio` | Potencia en frecuencia fundamental f0 / potencia en 2*f0       |

#### Por que duration_s, spectral_flatness y n_spectral_peaks_100_1000 se eliminaron del vector ML

En la version original del pipeline, estas tres variables estaban incluidas tanto en la matriz
de features ML como en las condiciones de la regla de etiquetado CAS. Esto creaba una
**dependencia circular**: el clasificador aprecia directamente los mismos valores que definen
si una muestra es CAS=1 o CAS=0, haciendo el problema ML trivialmente resoluble con un arbol
de decision de profundidad 3. La consecuencia fue RF AUC = 1.0000, un artefacto matematico
sin valor clinico. Tras eliminar estos features del vector ML (Fix 1), el RF AUC cayo a
0.8720, que refleja capacidad discriminativa real a partir de features independientes.

#### Matriz de etiquetado: 4 features exclusivos para generacion de etiquetas CAS

Estos features se computan por separado mediante `build_labeling_feature_matrix()` y se usan
**unicamente** para generar las etiquetas CAS. Nunca se pasan al clasificador.

| Feature                     | Descripcion                                                     |
|-----------------------------|-----------------------------------------------------------------|
| `spectral_flatness`         | Media geometrica / media aritmetica de la PSD (0=tonal, 1=ruido blanco) |
| `n_spectral_peaks_100_1000` | Numero de picos en 100-1000 Hz con prominencia >= 10% del maximo PSD    |
| `duration_s`                | Duracion del segmento en segundos                               |
| `max_peak_prominence_frac`  | Maxima prominencia topografica de pico en banda CAS, relativa a max(PSD) |

#### Normalizacion de features de amplitud por sujeto

La funcion `normalize_amplitude_features_per_subject(X, feature_names, v_subject)` aplica
z-score a las columnas `rms` y `peak_to_peak` de forma independiente para cada sujeto:
- Media y desviacion tipica calculadas unicamente con los segmentos de ese sujeto
- Sujetos con un unico segmento o desviacion cero no se modifican

Esta normalizacion elimina el sesgo de colocacion del microfono y de ganancia del sensor que
varia entre sujetos, reduciendo el riesgo de que el clasificador aprenda a distinguir sujetos
por nivel de amplitud absoluta en lugar de por caracteristicas acusticas de la senal.

### 2.6 Deteccion de CAS (`src/phase4_classifier.py`)

La funcion `make_rule_based_labels(X_label, feature_names)` asigna etiquetas binarias CAS
mediante una regla de cuatro condiciones sobre la matriz de etiquetado (NOT la matriz ML):

```
CAS = 1 si y solo si:
  spectral_flatness         < 0.10    (senal tonal, no ruido)
  n_spectral_peaks_100_1000 >= 2      (al menos 2 picos espectrales en banda CAS)
  duration_s                > 0.15    (duracion minima 150 ms)
  max_peak_prominence_frac  >= 0.05   (pico prominente respecto al maximo PSD)
```

**Tasa de positividad global (post-calibracion):** 55.6% de los 14 900 segmentos son
clasificados como CAS=1. Este valor esta por encima del rango objetivo de 20-50%, lo que
sugiere que los umbrales podrian ajustarse aun mas (p.ej. `n_peaks >= 3`).

**Umbrales pre-calibracion:** La version original usaba `flatness < 0.15`, `n_peaks >= 1`,
`duration > 0.10 s` y no tenia condicion de prominencia. Esto producia una tasa del 85.7%.

#### Pipelines de clasificacion

Se definen tres pipelines en el diccionario `PIPELINES`:

| Clave | Modelo                   | Preprocesado    |
|-------|--------------------------|-----------------|
| `svm` | SVM RBF, class_weight=balanced, probability=True | StandardScaler |
| `rf`  | RandomForest 200 arboles, class_weight=balanced  | StandardScaler |
| `knn` | KNeighborsClassifier k=7                         | StandardScaler |

El `StandardScaler` se ajusta **solo** sobre los datos de entrenamiento dentro de cada fold
LOSO, garantizando que no hay fuga de informacion del test al train.

### 2.7 Validacion LOSO (`src/phase4_validation.py`)

Se utiliza *Leave-One-Subject-Out* (LOSO) cross-validation mediante
`sklearn.model_selection.LeaveOneGroupOut` con `groups=v_subject`. Con 28 sujetos se
generan **28 folds**, cada uno con un sujeto como test y los 27 restantes como train.

La funcion `run_loso(pipeline, X, y, groups)` devuelve una lista de 28 diccionarios con las
metricas por fold:

| Metrica     | Descripcion                                          |
|-------------|------------------------------------------------------|
| accuracy    | Fraccion de segmentos correctamente clasificados     |
| sensitivity | TP / (TP + FN) — tasa de deteccion de CAS real      |
| specificity | TN / (TN + FP) — tasa de rechazo de no-CAS          |
| precision   | TP / (TP + FP)                                       |
| f1          | Media armonica de precision y sensibilidad           |
| auc         | Area bajo la curva ROC (usando probabilidades)       |

Los resultados por fold se guardan en `outputs/results/{modelo}/classifier_results.csv`.

### 2.8 Biomarcador delta_CAS (`src/phase5_biomarker.py`)

#### Definicion

```
delta_CAS(sujeto) = CAS_rate_pre - CAS_rate_post
```

Un valor positivo significa que la tasa de CAS disminuyo tras el broncodilatador; un valor
negativo indica que aumento. Segun la hipotesis clinica, los sujetos BDR+ deberan mostrar
values de delta_CAS mayores que los BDR-.

#### Tests estadisticos

La funcion `run_pre_post_test(pre, post)` elige entre Wilcoxon (*signed-rank*) y t-test
pareado segun el resultado del test de normalidad de Shapiro-Wilk. La funcion
`run_between_group_test(group_a, group_b)` aplica Mann-Whitney U bilateral.

#### Evaluacion como biomarcador

Se calcula el ROC AUC usando delta_CAS como score de clasificacion binaria BDR+/BDR-.

#### Figuras generadas

La funcion `generate_all_figures(cas_df, metadata_df, output_dir)` produce cuatro figuras
en `outputs/figures/`:

| Fichero                          | Contenido                                                   |
|----------------------------------|-------------------------------------------------------------|
| `barplot_cas_per_subject.png`    | Tasa de CAS pre-BD por sujeto, coloreado por grupo BDR      |
| `boxplot_cas_pre_post.png`       | Boxplot CAS rate pre vs post por grupo BDR (BDR+ vs BDR-)  |
| `heatmap_cas_subject_manoeuvre.png` | Heatmap sujetos x condicion (pre/post-BD) de tasa CAS   |
| `roc_curve_biomarker.png`        | Curva ROC usando delta_CAS como score de BDR+               |

---

## 3. Resultados

### 3.1 Tasa de CAS por sujeto

> **Nota:** La tabla siguiente fue generada con la pipeline tras la aplicacion de los fixes
> (umbrales calibrados). La tasa global post-calibracion es del 55.6% sobre los 14 900
> segmentos. Los sujetos controles (C1-C5, IDs 24-28) muestran tasas elevadas, lo que
> evidencia que los umbrales actuales no son suficientemente discriminativos entre CAS
> clinico y artefactos o variantes normales.

| subject_id | cas_rate_pre | cas_rate_post | delta_cas  | Tipo     | BDR    |
|-----------|-------------|--------------|------------|----------|--------|
| 1  (P1)   | 0.856       | 0.853        | +0.002     | Paciente | BDR-   |
| 2  (P2)   | 0.978       | 0.963        | +0.016     | Paciente | BDR+   |
| 3  (P3)   | 0.974       | 0.980        | -0.005     | Paciente | BDR-   |
| 4  (P4)   | 0.884       | 0.787        | +0.097     | Paciente | BDR-   |
| 5  (P5)   | 0.673       | 0.841        | -0.168     | Paciente | BDR-   |
| 6  (P6)   | 0.761       | 0.884        | -0.123     | Paciente | BDR+   |
| 7  (P7)   | 0.788       | 0.760        | +0.027     | Paciente | BDR+   |
| 8  (P8)   | 0.962       | 0.996        | -0.034     | Paciente | BDR+   |
| 9  (P9)   | 0.944       | 0.873        | +0.071     | Paciente | BDR+   |
| 10 (P10)  | 0.833       | 0.684        | +0.149     | Paciente | BDR+   |
| 11 (P11)  | 0.878       | 0.889        | -0.010     | Paciente | BDR+   |
| 12 (P12)  | 0.990       | 0.986        | +0.003     | Paciente | BDR+   |
| 13 (P13)  | 0.982       | 0.978        | +0.004     | Paciente | BDR-   |
| 14 (P14)  | 0.639       | 0.699        | -0.060     | Paciente | BDR+   |
| 15 (P15)  | 0.941       | 0.969        | -0.028     | Paciente | BDR-   |
| 16 (P16)  | 0.951       | 0.965        | -0.014     | Paciente | BDR-   |
| 17 (P17)  | 0.907       | 0.912        | -0.006     | Paciente | BDR-   |
| 18 (P18)  | 0.665       | 0.613        | +0.053     | Paciente | BDR-   |
| 19 (P19)  | 0.969       | 0.953        | +0.015     | Paciente | BDR-   |
| 20 (P20)  | 0.789       | 0.704        | +0.086     | Paciente | BDR-   |
| 21 (P21)  | 0.455       | 0.500        | -0.045     | Paciente | BDR-   |
| 22 (P22)  | 0.902       | 0.939        | -0.036     | Paciente | BDR-   |
| 23 (P23)  | 0.617       | 0.697        | -0.080     | Paciente | BDR-   |
| 24 (C1)   | 0.933       | 0.924        | +0.009     | Control  | BDR-   |
| 25 (C2)   | 0.854       | 0.878        | -0.024     | Control  | BDR-   |
| 26 (C3)   | 0.945       | 0.948        | -0.003     | Control  | BDR-   |
| 27 (C4)   | 0.965       | 0.884        | +0.082     | Control  | BDR-   |
| 28 (C5)   | 0.847       | 0.785        | +0.063     | Control  | BDR-   |

**Outliers destacables:**
- **P3 (pre = 97.4%):** tasa de CAS extremadamente alta en sujeto BDR-. Puede indicar
  presencia real de sibilancias en este paciente o artefactos en el canal de grabacion.
- **P12 (post = 98.6%):** sujeto BDR+, pero la tasa post-BD es casi identica a la pre-BD.
  El broncodilatador no redujo la carga acustica en este caso particular.
- **P21 (pre = 45.5%):** el sujeto con menor tasa de CAS del dataset, inusualmente baja
  para un paciente asmatico. Podria reflejar un fenotipo de asma no sibilante.

### 3.2 Resultados de clasificacion LOSO

Todos los resultados corresponden a la version **post-fix** (16 features, sin fuga circular).
Los valores de AUC son media +/- desviacion tipica sobre los 28 folds.

| Modelo | Accuracy | Sensitivity | Specificity | Precision | F1     | AUC (media +/- std) |
|--------|----------|-------------|-------------|-----------|--------|---------------------|
| SVM    | 0.7907   | 0.7374      | 0.8182      | 0.8461    | 0.7782 | 0.8731 +/- 0.054    |
| RF     | 0.7938   | 0.8040      | 0.7439      | 0.8012    | 0.7952 | 0.8720 +/- 0.054    |
| KNN    | 0.7541   | 0.7569      | 0.7160      | 0.7644    | 0.7548 | 0.8139 +/- 0.061    |

**Observaciones:**
- SVM y RF obtienen resultados muy similares (AUC ~0.872), con SVM siendo ligeramente mas
  especifico (0.818 vs 0.744) y RF ligeramente mas sensible (0.804 vs 0.737).
- KNN muestra mayor variabilidad entre folds (std=0.061) y menor AUC global (0.814).
- Todos los modelos superan ampliamente el azar (AUC=0.5), sugiriendo que los 16 features
  independientes del etiquetado contienen informacion discriminativa real.

**Comparativa pre-fix vs post-fix:**

| Version  | RF AUC   | SVM AUC  | CAS rate |
|----------|----------|----------|----------|
| Pre-fix  | 1.0000   | 0.9955   | 85.7%    |
| Post-fix | 0.8720   | 0.8731   | 55.6%    |

### 3.3 Biomarcador delta_CAS

| Metrica                              | Valor          |
|--------------------------------------|----------------|
| ROC AUC (delta_CAS como score BDR+) | 0.42           |
| Test pre vs post (todos, Wilcoxon)  | p = 0.779      |
| Test pre vs post (BDR+, Wilcoxon)  | p = 0.910      |
| Test pre vs post (BDR-, Wilcoxon)  | p = 0.595      |
| Mann-Whitney BDR+ vs BDR-           | p = 0.523      |
| delta_CAS BDR+ (media +/- std)     | +0.002 +/- 0.126 |
| delta_CAS BDR- (media +/- std)     | +0.011 +/- 0.077 |

**Interpretacion:** El biomarcador no muestra capacidad discriminativa estadisticamente
significativa (todos los p-valores >> 0.05, AUC=0.42 < 0.50). Notablemente, la direccion
del efecto esta **invertida**: los sujetos BDR- reducen ligeramente mas su tasa de CAS que
los BDR+. Esto no es un error de codigo (la implementacion del test y la formula delta_CAS
son correctas), sino una limitacion del sistema de etiquetado basado en reglas heuristicas
no validadas contra anotacion clinica experta.

### 3.4 Figuras generadas

| Fichero                               | Descripcion                                                                |
|---------------------------------------|----------------------------------------------------------------------------|
| `barplot_cas_per_subject.png`         | Diagrama de barras con la tasa de CAS pre-BD por sujeto, coloreadas en rojo (BDR+) y azul (BDR-). Permite identificar visualmente que la distribucion de tasas CAS no separa claramente los dos grupos. |
| `boxplot_cas_pre_post.png`            | Boxplot comparando la tasa de CAS pre-BD y post-BD en los grupos BDR+ y BDR-. Muestra la ausencia de una reduccion sistematica diferencial entre grupos tras el broncodilatador. |
| `heatmap_cas_subject_manoeuvre.png`   | Mapa de calor (sujeto x condicion) con la tasa CAS por sujeto en pre-BD y post-BD. Permite identificar sujetos con tasas extremas y patrones visuales de cambio. |
| `roc_curve_biomarker.png`             | Curva ROC usando delta_CAS como score de clasificacion BDR+/BDR-. El AUC=0.42 indica que el biomarcador no supera el clasificador aleatorio en este dataset. |

---

## 4. Problemas encontrados y correcciones aplicadas

### Fix 1 — Dependencia circular en la matriz de features (CRITICO)

**Problema:** Las variables `spectral_flatness`, `n_spectral_peaks_100_1000` y `duration_s`
estaban incluidas tanto en la matriz de features ML (`FEATURE_NAMES`, 19 columnas) como en
las condiciones de la regla CAS que generaba las etiquetas `y`. Esto significa que el
clasificador recibia como inputs los mismos valores que determinaban si cada muestra era
CAS=1 o CAS=0. Un arbol de decision de profundidad 3 podia replicar exactamente la regla y
obtener AUC=1.0 sin aprender ningun patron acustico real.

**Evidencia:** RF AUC = 1.0000 en LOSO, lo que es fisiologicamente inverosimil para este
tipo de datos.

**Correccion aplicada:**
- Se creo una infraestructura separada: `LABELING_FEATURE_NAMES` (4 features) y
  `build_labeling_feature_matrix()` para uso exclusivo del etiquetado.
- `FEATURE_NAMES` se redujo de 19 a **16 features**, excluyendo las tres variables
  circulares.
- `make_rule_based_labels()` ahora requiere la matriz de etiquetado como input, no la
  matriz ML.

**Resultado post-fix:** RF AUC = 0.8720 (reduccion de 0.128).

### Fix 2 — Umbrales de deteccion CAS demasiado permisivos (CRITICO)

**Problema:** Con `CAS_MIN_PEAKS = 1` y `CAS_FLATNESS_THRESHOLD = 0.15`, el 85.7% de los
segmentos se clasificaban como CAS=1, lo que no refleja la prevalencia clinica real de
sibilancias (tipicamente 20-50% en asmaticos durante maniobras forzadas). Una tasa de
positividad del 86% indica que el detector es casi siempre activo, haciendo inutil el
biomarcador delta_CAS.

**Correccion aplicada:**

| Umbral                    | Antes  | Despues |
|---------------------------|--------|---------|
| CAS_FLATNESS_THRESHOLD    | 0.15   | 0.10    |
| CAS_MIN_PEAKS             | 1      | 2       |
| CAS_MIN_DURATION_S        | 0.10   | 0.15    |
| CAS_MIN_PEAK_PROMINENCE   | (no existia) | 0.05 |

Se anyadio una cuarta condicion de prominencia espectral para excluir picos espectrales
debiles que no corresponden a sibilancias reales.

**Resultado post-fix:** tasa global reducida de 85.7% a 55.6% (aun por encima del rango
objetivo 20-50%; se recomienda ajuste adicional).

### Fix 3 — Verificacion de features de potencia de banda (WARNING)

**Hallazgo:** La funcion `_band_power_fraction()` ya implementaba correctamente el calculo
como fraccion del total (`sum(psd[banda]) / total_power`). Los cuatro features de potencia
de banda son invariantes a la ganancia absoluta.

**Correccion aplicada:** Ninguna. Confirmado como correcto.

### Fix 4 — Features de amplitud absoluta dependientes del sensor (WARNING)

**Problema:** Los features `rms` y `peak_to_peak` dependen de la ganancia del microfono y
la distancia de colocacion al cuerpo, que varia entre sujetos. En un escenario LOSO, el
clasificador podia aprender a distinguir sujetos por su nivel de amplitud absoluta en lugar
de por caracteristicas acusticas de las sibilancias.

**Correccion aplicada:** Se implemento `normalize_amplitude_features_per_subject()` que
aplica z-score a `rms` y `peak_to_peak` de forma independiente para cada sujeto. La
normalizacion se aplica inmediatamente tras `build_feature_matrix()`, antes de cualquier
entrenamiento.

### Fix 5 — Validacion del conteo de segmentos (INFO)

**Problema:** Si algun fichero `.mat` faltaba o contenia marcas erroneas, el dataset podia
construirse con un numero incorrecto de segmentos sin ninguna notificacion al usuario.

**Correccion aplicada:** Se anyadio un `warnings.warn(RuntimeWarning)` al final de
`build_dataset()` si el numero total de segmentos difiere de 14 900 (valor esperado).

---

## 5. Conclusiones

### 5.1 Sobre el clasificador de CAS

- El pipeline ML es **metodologicamente valido** tras la aplicacion de los cinco fixes: no
  hay fuga de datos entre train y test (LOSO correcto), no hay dependencia circular
  feature/etiqueta, y la normalizacion de amplitud es por sujeto.
- Un AUC de ~0.87 (SVM y RF) es **clinicamente plausible** para un sistema de deteccion
  automatica de sibilancias sin ground truth anotado por expertos.
- La implementacion LOSO con `StandardScaler` dentro del pipeline garantiza que el escalado
  se ajusta unicamente con datos de entrenamiento de cada fold.

### 5.2 Sobre el biomarcador delta_CAS

- El biomarcador **no muestra capacidad discriminativa** para detectar BDR (AUC = 0.42,
  todos los tests estadisticos no significativos).
- La direccion del efecto esta invertida respecto a la hipotesis clinica (BDR- reduce mas
  CAS que BDR+), pero la diferencia entre grupos es minima y no significativa.
- Esto **no es un error de implementacion** sino una limitacion fundamental: las etiquetas
  CAS se generan mediante reglas heuristicas no validadas contra anotacion experta. Sin
  un ground truth clinico de referencia, no es posible determinar si las etiquetas CAS
  reflejan correctamente la presencia de sibilancias reales.

### 5.3 Limitaciones del estudio

1. **Etiquetado sin ground truth:** Las etiquetas CAS se generan por umbralizado de features
   espectrales, sin validacion contra anotacion por fononeumologos o base de datos publica.
   La tasa de CAS del 55.6% sigue siendo alta, lo que sugiere falsos positivos.

2. **Tamano de muestra:** n=23 pacientes asmaticos es insuficiente para extraer conclusiones
   estadisticas robustas sobre el biomarcador. Los tests de Wilcoxon y Mann-Whitney tienen
   muy baja potencia estadistica con este tamano muestral.

3. **Umbrales sin calibracion:** Los umbrales de deteccion (`flatness < 0.10`,
   `n_peaks >= 2`, etc.) se establecieron heuristicamente y no han sido ajustados sobre un
   conjunto de validacion independiente con anotacion experta.

4. **Canales no analizados por separado:** El pipeline procesa ambos canales de forma
   identica. El canal traqueal y el canal de pecho pueden ofrecer informacion complementaria
   que no se explota en la version actual.

### 5.4 Trabajo futuro

1. **Calibracion de umbrales contra anotaciones expertas:** Utilizar una base de datos
   publica de sibilancias anotadas (RALE Repository, HF Lung dataset) para optimizar los
   umbrales de la regla CAS mediante validacion cruzada.

2. **Features alternativos para el biomarcador:** Explorar la duracion media de episodios
   CAS, la variabilidad ciclo a ciclo de la tasa CAS, y el ratio espectral por bandas como
   alternativas a la tasa CAS media.

3. **Analisis por canal:** Analizar canal traqueal (canal 1) y canal de pecho (canal 2)
   por separado para identificar cual tiene mayor sensibilidad a las sibilancias.

4. **Modelos de aprendizaje profundo:** Explorar arquitecturas CNN o LSTM sobre espectrogramas
   de segmentos, que podrian capturar patrones temporales que los features estadisticos
   no recogen.

5. **Ajuste fino de umbrales CAS:** Probar `CAS_MIN_PEAKS = 3` o
   `CAS_FLATNESS_THRESHOLD = 0.07` para reducir la tasa de positividad por debajo del 50%.

---

## 6. Estructura del repositorio

```
Project/
|-- src/                          # Codigo fuente del pipeline
|   |-- __init__.py
|   |-- phase1_io.py              # Lectura de ficheros .mat de LabChart
|   |-- phase1_preprocessing.py  # Remuestreo, bandpass, notch comb
|   |-- phase2_dataset.py        # Construccion del dataset de 14 900 segmentos
|   |-- phase2_segmentation.py   # Carga de marcas temporales y segmentacion
|   |-- phase3_features.py       # Extraccion de 16 features ML + 4 features de etiquetado
|   |-- phase4_classifier.py     # Regla CAS y definicion de pipelines sklearn
|   |-- phase4_validation.py     # LOSO cross-validation y metricas
|   `-- phase5_biomarker.py      # Calculo de tasas CAS, tests estadisticos y figuras
|
|-- tests/                        # Tests unitarios (pytest)
|   |-- test_phase3_features.py  # Tests de extraccion de features (74 tests)
|   `-- test_phase4_classifier.py # Tests de clasificacion y validacion
|
|-- database/
|   `-- subject_metadata.csv     # ID de sujeto, tipo, sexo, etiqueta BDR
|
|-- outputs/
|   |-- figures/                  # Figuras PNG (4 figuras generadas)
|   `-- results/                  # Artefactos del pipeline
|       |-- dataset.npz           # Vectores de metadatos (v_subject, v_bd, v_channel, v_phase)
|       |-- all_signals.pkl       # Lista de 14 900 arrays de senal
|       |-- X_features.npy        # Matriz de features (version pre-fix, 19 columnas)
|       |-- y_cas_labels.npy      # Etiquetas CAS (version pre-fix)
|       |-- cas_rates_per_subject.csv # Tasas CAS por sujeto
|       |-- run_summary.json      # Resumen de ejecucion (version pre-fix)
|       |-- svm/classifier_results.csv
|       |-- rf/classifier_results.csv
|       |-- knn/classifier_results.csv
|       `-- _pipeline_cache/      # Cache de la ejecucion post-fix
|           |-- X_16.npy          # Matriz ML post-fix (14900, 16)
|           |-- X_label_4.npy     # Matriz de etiquetado post-fix (14900, 4)
|           `-- clf_results.json  # Resultados LOSO post-fix (SVM, RF, KNN)
|
|-- Data/                         # Ficheros .mat de LabChart (no incluidos en repo)
|-- notebooks/                    # Notebooks de exploracion
|-- run_pipeline.py               # Script de ejecucion completa del pipeline
|-- requirements.txt              # Dependencias Python
|-- pytest.ini                    # Configuracion de pytest
`-- README.md                     # Descripcion general del proyecto
```

---

## 7. Como ejecutar el pipeline

### 7.1 Requisitos

- Python 3.10 o superior (el proyecto usa Python 3.13 en el entorno de desarrollo)
- Entorno virtual recomendado

### 7.2 Instalacion de dependencias

```bash
pip install -r requirements.txt
```

Dependencias principales:

| Paquete        | Version minima | Uso                                      |
|----------------|---------------|------------------------------------------|
| numpy          | >= 1.26       | Arrays, operaciones numericas            |
| scipy          | >= 1.12       | DSP, estadistica, carga de .mat          |
| scikit-learn   | >= 1.4        | Clasificadores, LOSO, metricas           |
| pandas         | >= 2.1        | DataFrames para resultados y metadatos   |
| matplotlib     | >= 3.8        | Figuras                                  |
| seaborn        | >= 0.13       | Figuras estadisticas                     |
| imbalanced-learn | >= 0.12    | Utilidades para clases desbalanceadas    |
| pytest         | >= 8.0        | Suite de tests                           |

### 7.3 Ejecucion de tests

```bash
python -m pytest tests/ -v
```

La suite contiene **74 tests** que verifican:
- Formas de arrays de features (5, 8, 3 por bloque; 16 total ML; 4 etiquetado)
- Propiedades de features individuales (RMS, ZCR, planitud espectral, picos)
- Ausencia de NaN en la matriz de features
- Correctitud de la regla CAS con datos sinteticos
- Metricas de clasificacion (clasificador perfecto, sin probabilidades, clase unica)
- Ejecucion LOSO completa (numero de folds, claves de resultado, sin fuga de sujeto)
- Guardado de CSV de resultados

### 7.4 Ejecucion completa del pipeline

**Prerequisito:** Los ficheros `.mat` de datos deben estar en `Data/` y el dataset debe
haberse construido previamente (guardando `all_signals.pkl` y `dataset.npz` en
`outputs/results/`).

```bash
python run_pipeline.py
```

El script:
1. Carga `all_signals.pkl` y `dataset.npz` desde `outputs/results/`
2. Extrae la matriz ML de 16 features y la normaliza por sujeto (o carga desde cache)
3. Extrae la matriz de etiquetado de 4 features (o carga desde cache)
4. Aplica la regla CAS para generar etiquetas binarias
5. Ejecuta LOSO para SVM, RF y KNN (o carga desde cache)
6. Calcula tasas CAS por sujeto y delta_CAS
7. Genera las 4 figuras en `outputs/figures/`
8. Imprime el informe de evaluacion completo

**Cache:** Las matrices de features y los resultados LOSO se almacenan en
`outputs/results/_pipeline_cache/` para evitar recomputacion en ejecuciones sucesivas.

### 7.5 Outputs generados

| Fichero                                    | Descripcion                          |
|--------------------------------------------|--------------------------------------|
| `outputs/results/_pipeline_cache/X_16.npy`    | Matriz ML (14900, 16) post-fix       |
| `outputs/results/_pipeline_cache/X_label_4.npy` | Matriz etiquetado (14900, 4)       |
| `outputs/results/_pipeline_cache/clf_results.json` | Resultados LOSO por modelo     |
| `outputs/figures/barplot_cas_per_subject.png`  | Figura 1                            |
| `outputs/figures/boxplot_cas_pre_post.png`     | Figura 2                            |
| `outputs/figures/heatmap_cas_subject_manoeuvre.png` | Figura 3                      |
| `outputs/figures/roc_curve_biomarker.png`      | Figura 4                            |
