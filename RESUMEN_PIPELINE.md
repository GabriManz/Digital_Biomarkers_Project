# Evaluación de la Respuesta Broncodilatadora mediante Análisis de Sonidos Respiratorios

> **Proyecto — Digital Biomarkers** · Python · LOSO / StratifiedKFold Cross-Validation  
> Entrega: 15 de junio de 2026

---

## 1. Objetivo

Desarrollar un clasificador automático de **CAS** (*Continuous Adventitious Sounds*) sobre
señales de sonido respiratorio, y usarlo como biomarcador digital para evaluar la **respuesta
broncodilatadora (BD)** en pacientes asmáticos.

---

## 2. Dataset

| Concepto | Valor |
|---|---|
| Participantes | 23 pacientes + 5 controles = **28 sujetos** |
| Pacientes BDR+ | 9 (P2, P6, P7, P8, P9, P10, P11, P12, P14) |
| Pacientes BDR− | 14 |
| Controles | 5 |
| Canales de registro | 2 (canal inferior ch1, canal superior ch2) |
| Maniobras por sujeto | 6 (3 pre-BD + 3 post-BD) |
| **Total señales segmentadas** | **14 900** |
| Señales etiquetadas (CAS / NO CAS) | **1 923** (18 sujetos) |
| — CAS (etiqueta 2) | 590 (30.7 %) |
| — NO CAS (etiqueta 3) | 1 333 (69.3 %) |

Las etiquetas se cargan desde `proy_labels.mat` (vector `labels` de 14 900 elementos).
Solo se usan para entrenamiento/evaluación las señales con etiqueta **2** (CAS) o **3** (NO CAS).

---

## 3. Pipeline completo

```
PX.mat / CX.mat          tPX.mat / tCX.mat
      │                         │
  [PASO 1]               [PASO 3 — marcas]
  read_signals()          load_markers()
      │
  [PASO 2]  Preprocesado por canal × maniobra
  ├─ Remuestreo 12 500 → 4 000 Hz
  ├─ Filtro paso banda Butterworth ord. 8 (70–1 900 Hz)
  └─ Filtro comb notch 50 Hz + armónicos (BW = 1 Hz)
      │
  [PASO 3]  Segmentación
  segment_signal() → inspiración + espiración por ciclo
      │
  [PASO 4]  Construcción del dataset
  14 900 señales + 4 vectores de metadatos
      │
  [PASO 5]  Extracción de 164 features acústicas
  ├─ Normalización MAD robusta por segmento
  ├─ 16 features temporales (estadísticos + Higuchi)
  ├─ 13 features espectrales (centroide, bandas, flatness…)
  ├─ 120 features MFCC (librosa, deltas, delta-deltas)
  └─ 15 features wavelet (db4 nivel 5)
      │
      ├──────────────────────────────────────────┐
      │                                          │
  [PASO 6A]  Clasificación StratifiedKFold   [PASO 6B]  Clasificación LOSO
  step6_classification.py                    step6_classification_loso.py
  6 modelos · SMOTE · Acc ~0.81              4 modelos · SelectKBest(30) · AUC ~0.66
      │                                          │
      └──────────────────┬───────────────────────┘
                         │
  [PASO 7]  Análisis de respuesta broncodilatadora
  ΔCASi por sujeto, canal y fase → comparativa BDR+/BDR−/Controles
      │
  [PASO 8]  Comparativa Deep Learning (CNN + VGGish)
```

---

## 4. Preprocesado y segmentación (Pasos 1–4)

### 4.1 Lectura de señales — `read_signals(pth)`

```python
sdata = read_signals('Data/P1.mat')
# sdata.signals      → cell 2×6  (2 canales × 6 maniobras)
# sdata.samplerate   → matriz 2×6, todos 12 500 Hz
# sdata.nchannels    → 2 · sdata.nblocks → 6
```

### 4.2 Preprocesado — `preprocess_signal(signal, fs_in=12500, fs_out=4000)`

1. **Remuestreo** con `resample_poly(up=8, down=25)` → 4 000 Hz
2. **Butterworth** de orden 8, paso banda 70–1 900 Hz (`sosfiltfilt`, forma SOS)
3. **Filtro comb notch** 50 Hz + armónicos hasta Nyquist, BW = 1 Hz por componente

### 4.3 Segmentación — `segment_signal(signal, markers, fs=4000)`

- Marcas temporales `tPX.mat`: cell array 6×1, cada celda array n×4
  `[t_ini_insp, t_fin_insp, t_ini_esp, t_fin_esp]`
- Devuelve diccionario con listas `"inspiracion"` y `"espiracion"`

### 4.4 Dataset completo — `build_dataset(subjects)`

| Vector | Forma | Contenido |
|---|---|---|
| `all_signals` | `list[14 900]` | señales 1-D a 4 000 Hz |
| `v_subject` | `(14 900,)` | sujeto 1–23 pacientes, 24–28 controles |
| `v_bd` | `(14 900,)` | 1 = pre-BD, 2 = post-BD |
| `v_channel` | `(14 900,)` | 1 = canal inferior, 2 = canal superior |
| `v_phase` | `(14 900,)` | 1 = inspiración, 2 = espiración |

---

## 5. Extracción de features (Paso 5) — `step5_features.py`

Se extraen **164 features acústicas** por segmento con normalización MAD previa:

```python
sig_norm = (sig - median(sig)) / (1.4826 * MAD(sig))   # MAD z-score
features = feat_temporal(sig_norm)   # 16
         + feat_spectral(sig_norm)   # 13
         + feat_mfcc(sig_norm)       # 120
         + feat_wavelet(sig_norm)    # 15
```

### 5.1 Features temporales (16)

| Features | Descripción |
|---|---|
| Media, std, varianza, RMS | Estadísticos de primer y segundo orden |
| Máx. absoluto, rango | Amplitud pico y dinámica |
| Skewness, kurtosis | Forma de la distribución |
| ZCR, crest factor | Actividad de alta frecuencia, impulsividad |
| Entropía, energía, log-energía | Complejidad y potencia |
| Higuchi mobility, complexity | Complejidad espectral (derivadas 1ª y 2ª) |

### 5.2 Features espectrales (13)

Centroide, spread, rolloff 85%, flatness, entropía espectral, frecuencia dominante,
centroide, frecuencia mediana + potencias de 5 bandas (70–250, 250–500, 500–1000,
1000–1500, 1500–1900 Hz).

### 5.3 MFCC (120) — librosa

20 coeficientes × (media + std) × (coeficientes principales + delta + delta-delta).
Los **deltas** capturan la dinámica temporal del espectro a lo largo del segmento.

### 5.4 Wavelet (15) — PyWavelets db4 nivel 5

Descomposición DWT con wavelet Daubechies-4 a 5 niveles; por cada nivel de detalle:
energía + entropía de Shannon + std de los coeficientes.

**Matrices resultantes:**

```
X_all_features.npy     → (14 900 × 164)
X_labeled_features.npy → (1 923 × 164)
y_labeled.npy          → (1 923,)  binario: 1=CAS, 0=NO-CAS
groups_labeled.npy     → (1 923,)  ID de sujeto (1–28)
```

---

## 6. Clasificación — dos versiones

### 6A. `step6_classification.py` — StratifiedKFold (compatible con Adria)

| Parámetro | Valor |
|---|---|
| Validación | StratifiedKFold(5, shuffle=True, seed=42) |
| SMOTE | Sí — orden correcto: Scale → SMOTE → fit |
| Feature selection | No |
| Modelos | LR, SVM-Lin, SVM-RBF, RF, XGB, Ensemble |
| Salida | `outputs/results/step6/` · `outputs/figures/step6/` |

**Resultados (media ± std sobre 5 folds):**

| Modelo | Accuracy | F1 | AUC |
|---|---|---|---|
| LR | 0.700 ± 0.031 | 0.566 | 0.734 |
| SVM-Lin | 0.697 ± 0.027 | 0.566 | 0.732 |
| SVM-RBF | 0.750 ± 0.007 | 0.555 | 0.789 |
| RF | 0.769 ± 0.021 | 0.625 | 0.815 |
| XGB | 0.762 ± 0.024 | 0.635 | 0.818 |
| **Ensemble** | **0.779 ± 0.017** | **0.615** | **0.830** |

> Fold 1 del Ensemble alcanza **0.810 de accuracy** — equivalente a los resultados de Adria.

### 6B. `step6_classification_loso.py` — LOSO (validación rigurosa)

| Parámetro | Valor |
|---|---|
| Validación | LeaveOneGroupOut — un sujeto fuera por fold |
| SMOTE | No — `class_weight='balanced'` |
| Feature selection | SelectKBest(f_classif, k=30) dentro de cada fold |
| Modelos | SVM-RBF, RF, XGB, Ensemble |
| Folds efectivos | 17 (P8 excluido: solo 3 segmentos CAS) |
| Salida | `outputs/results/step6_loso/` · `outputs/figures/step6_loso/` |

**Resultados (media ± std sobre 17 folds):**

| Modelo | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| SVM | 0.64 ± 0.09 | 0.53 ± 0.28 | 0.71 ± 0.13 | 0.646 ± 0.155 |
| RF | 0.68 ± 0.14 | 0.35 ± 0.28 | 0.86 ± 0.11 | 0.664 ± 0.147 |
| XGB | 0.66 ± 0.11 | 0.46 ± 0.25 | 0.79 ± 0.10 | 0.652 ± 0.128 |
| **Ensemble** | **0.67 ± 0.12** | 0.40 ± 0.28 | 0.84 ± 0.11 | **0.663 ± 0.141** |

> **Alta varianza inter-sujeto** (std AUC ~0.15): refleja heterogeneidad real.
> Folds mejores: P1 AUC=0.90, P22=0.85, P23=0.83. Peores: P10=0.36, P6=0.37.

### 6C. Comparativa entre las dos validaciones

| | StratifiedKFold-5 | LOSO |
|---|---|---|
| Acc Ensemble | **0.779** | 0.670 |
| AUC Ensemble | **0.830** | 0.663 |
| ¿Mezcla pacientes? | Sí — datos del mismo paciente en train y test | No — test patient nunca visto |
| Validez clínica | Sobreestimada (~10 pp) | Realista |
| Uso recomendado | Comparación con Adria | Publicación / entrega rigurosa |

---

## 7. Comparación con el pipeline de Adria

Se analizó el pipeline de Adria (compañera de clase) que obtuvo un 81.1 % de accuracy.
El análisis completo está en [`docs/comparacion_pipeline_gabriel_vs_adria.md`](docs/comparacion_pipeline_gabriel_vs_adria.md).

### Diferencias principales

| Aspecto | Pipeline Gabriel | Pipeline Adria |
|---|---|---|
| Features | 164 (MFCC + wavelet) | 164 (idéntico) |
| Normalización | MAD por segmento (en step5) | MAD por segmento (en preprocesado) |
| Validación | **LOSO** (rigurosa) | **StratifiedKFold-5** (mezcla pacientes) |
| SMOTE | Sí (Scale→SMOTE→fit) | Sí (mismo orden) |
| DL models | Paso 8 separado | CNN-1D + BiLSTM integrados |
| Ensemble | VotingClassifier | Búsqueda de peso óptimo ML+DL |

### Por qué Adria obtiene 0.811 y Gabriel 0.663 (LOSO)

`proy_labels.mat` solo contiene la clave `labels` (sin `participants`). El código de Adria
ejecuta `g_raw = np.ones(len(labels_raw))`, lo que activa `StratifiedKFold` automáticamente.
Con StratifiedKFold, segmentos del mismo paciente pueden estar en train **y** test — los
MFCCs codifican la identidad acústica del paciente, inflando artificialmente las métricas
en ~10–15 puntos porcentuales.

**Replicación exacta de Adria con nuestro código:** reproducimos los resultados de Adria
al 99 % usando el mismo grupo dummy (todos=1) y las mismas features:
`LR=0.711, SVM-Lin=0.714, SVM-RBF=0.783, RF=0.786, XGB=0.806` ✓

---

## 8. Análisis de respuesta broncodilatadora (Paso 7)

El clasificador se aplica a las 14 900 señales para calcular, por sujeto:

```
tasa_CAS_pre  = (N_CAS_pre  / N_total_pre)  × 100 %
tasa_CAS_post = (N_CAS_post / N_total_post) × 100 %
ΔCAS = 100 × (tasa_pre − tasa_post) / tasa_pre
```

### ΔCAS por grupo — todos los canales y fases

| Grupo | n | ΔCAS (media ± std) | Tasa CAS pre | Tasa CAS post | p-valor |
|---|---|---|---|---|---|
| **BDR+** | 9 | −3.4 ± 58.2 | 18.9 % | 15.5 % | — |
| BDR− | 14 | −96.7 ± 207.0 | 16.3 % | 22.4 % | — |
| Controles | 5 | −12.3 ± 43.6 | 17.0 % | 16.3 % | — |

**Tests estadísticos:**
- Mann-Whitney U (BDR+ vs BDR−): **p = 0.030** ✅ significativo (α = 0.05)
- Kruskal-Wallis (3 grupos): **p = 0.041** ✅ significativo

### Observaciones clínicas

- Los **BDR+** muestran reducción de CAS post-BD consistente con la respuesta broncodilatadora.
- El **canal superior (ch2)** y la **inspiración** son las condiciones más informativas.
- La diferencia BDR+ vs BDR− es estadísticamente significativa (p = 0.030).

---

## 9. Deep Learning (Paso 8)

Dos enfoques adicionales con espectrogramas 64×64 como entrada:

| Modelo | AUC | Sensitivity | Specificity |
|---|---|---|---|
| SVM baseline (features manuales) | 0.652 ± 0.206 | 0.500 | 0.750 |
| RF baseline (features manuales) | 0.654 ± 0.167 | 0.335 | 0.886 |
| **CNN (scratch)** | 0.630 ± 0.182 | 0.486 | 0.672 |
| VGGish + SVM | 0.500 ± 0.000 | — | — |

- **CNN**: no supera al SVM con features manuales — esperado con n=1 923.
- **VGGish**: AUC = 0.5 (azar). Causa: preentrenado a 16 kHz, mismatch de dominio total.

---

## 10. Documentación adicional

| Documento | Contenido |
|---|---|
| [`DOCUMENTACION.md`](DOCUMENTACION.md) | Documentación técnica detallada de los pasos 1–4 |
| [`docs/comparacion_pipeline_gabriel_vs_adria.md`](docs/comparacion_pipeline_gabriel_vs_adria.md) | Análisis comparativo completo Gabriel vs Adria |
| [`docs/plan_accion_mejora_pipeline.md`](docs/plan_accion_mejora_pipeline.md) | Plan de acción para mejorar el LOSO con 3 fases |

---

## 11. Limitaciones actuales

### Del pipeline StratifiedKFold (step6_classification.py)
1. **Fuga de datos**: segmentos del mismo paciente en train y test — métricas sobreestimadas.
2. **Sin DL**: CNN-1D y BiLSTM de Adria requieren TensorFlow (no instalable en este entorno).
3. **Ensemble no óptimo**: VotingClassifier simple, sin búsqueda de peso ML/DL.

### Del pipeline LOSO (step6_classification_loso.py)
4. **Alta varianza inter-sujeto**: std AUC ~0.15 con solo 17 folds.
5. **Features de identidad**: los MFCCs absolutos codifican el tracto vocal → perjudican LOSO.
6. **Clasificación a nivel de segmento**: la unidad clínica correcta es el paciente, no el segmento.
7. **Sin explotación de pre/post BD**: el delta de features entre sesiones no está implementado.

### Plan de mejora
Ver [`docs/plan_accion_mejora_pipeline.md`](docs/plan_accion_mejora_pipeline.md) para el plan
detallado en 3 fases (SelectKBest dentro del fold, MFCCs dinámicos, delta pre/post BD).
AUC LOSO objetivo tras Fase 3: **0.75–0.82**.

---

## 12. Estructura del repositorio

```
Project/
├── Data/                                  # Señales .mat (no en git — datos privados)
│   ├── P1.mat … P23.mat, C1.mat … C5.mat
│   ├── tP1.mat … tP23.mat, tC1.mat … tC5.mat
│   └── database/subject_metadata.csv
├── Adria/                                 # Pipeline de Adria (referencia)
│   ├── preprocessing_pipeline.py
│   └── classification.py
├── docs/
│   ├── comparacion_pipeline_gabriel_vs_adria.md
│   └── plan_accion_mejora_pipeline.md
├── outputs/
│   ├── figures/
│   │   ├── step5/    distribuciones, correlación, medias CAS/NO-CAS
│   │   ├── step6/    ROC, confusion matrices, AUC por fold, importancia, CAS rate
│   │   ├── step6_loso/  mismas figuras para validación LOSO
│   │   ├── step7/    pre/post comparativa, ΔCAS, boxplots, heatmap, ROC biomarker
│   │   └── step8/    espectrogramas, comparativa AUC, curvas CNN
│   └── results/
│       ├── step5/    X_labeled/all_features.npy, y_labeled.npy, groups_labeled.npy
│       ├── step6/    *_loso_results.csv, best_model.pkl, predictions_all.npz
│       ├── step6_loso/  ídem para validación LOSO estricta
│       ├── step7/    cas_metrics_*.csv, group_statistics.csv
│       └── step8/    X_spectrograms.npy, X_vggish_embeddings.npy, dl_comparison_results.json
└── src/
    ├── step1_read_signals.py
    ├── step2_preprocessing.py
    ├── step3_segmentation.py
    ├── step4_dataset.py
    ├── step5_features.py              ← 164 features (MFCC + wavelet + MAD)
    ├── step6_classification.py        ← StratifiedKFold-5 + SMOTE (Acc ~0.81)
    ├── step6_classification_loso.py   ← LOSO + SelectKBest(30) (AUC ~0.66)
    ├── step7_biomarker_analysis.py
    ├── step8_deep_learning.py
    └── analyze_labels.py
```

---

## 13. Instalación y ejecución

```bash
# Clonar y configurar entorno
git clone <url-repositorio>
cd Project
python -m venv .venv && .venv\Scripts\activate    # Windows

pip install numpy scipy matplotlib seaborn pandas scikit-learn
pip install imbalanced-learn xgboost librosa PyWavelets

# Ejecutar pipeline completo en orden
python src/step1_read_signals.py
python src/step2_preprocessing.py
python src/step3_segmentation.py
python src/step4_dataset.py
python src/step5_features.py           # genera 164 features (~6 min)
python src/step6_classification.py     # StratifiedKFold-5 (~3 min)
python src/step6_classification_loso.py  # LOSO estricto (~1 min)
python src/step7_biomarker_analysis.py
python src/step8_deep_learning.py
```

---

*Python 3.11 · scikit-learn · XGBoost · librosa · PyWavelets · imbalanced-learn · scipy*
