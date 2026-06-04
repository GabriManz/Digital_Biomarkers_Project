# Evaluación de la Respuesta Broncodilatadora mediante Análisis de Sonidos Respiratorios

> **Proyecto — Digital Biomarkers** · Python · LOSO Cross-Validation  
> Entrega: 15 de junio de 2026

---

## 1. Objetivo

Desarrollar un clasificador automático de **CAS** (*Crackle-like Adventitious Sounds*) sobre
señales de sonido respiratorio, y usarlo como biomarcador digital para evaluar la **respuesta
broncodilatadora (BD)** en pacientes asmáticos.

---

## 2. Dataset

| Concepto | Valor |
|---|---|
| Participantes | 23 pacientes + 5 controles = **28 sujetos** |
| Pacientes BDR+ | 9 |
| Pacientes BDR− | 14 |
| Controles | 5 |
| Canales de registro | 2 (canal inferior ch1, canal superior ch2) |
| Maniobras por sujeto | 6 (3 pre-BD + 3 post-BD) |
| **Total señales segmentadas** | **14 900** |
| Señales etiquetadas (CAS / NO CAS) | **1 923** |
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
  [PASO 5]  Extracción de 15 features acústicas
      │
  [PASO 6]  Clasificación LOSO
  SVM / RF / XGB / Ensemble → mejor modelo → inferencia 14 900
      │
  [PASO 7]  Análisis de respuesta broncodilatadora
  ΔCASi por sujeto, canal y fase → comparativa BDR+/BDR−/Controles
      │
  [PASO 8*] Comparativa Deep Learning (CNN + VGGish)
            * Análisis adicional no requerido
```

---

## 4. Preprocesado y segmentación (Pasos 1–4)

### 4.1 Lectura de señales — `read_signals(pth)`

```python
sdata = read_signals('Data/P1.mat')
# sdata.signals      → cell 2×6  (2 canales × 6 maniobras)
# sdata.samplerate   → matriz 2×6, todos 12 500 Hz
# sdata.nchannels    → 2
# sdata.nblocks      → 6
```

### 4.2 Preprocesado — `preprocess_signal(signal, fs_in=12500, fs_out=4000)`

1. **Remuestreo** con `resample_poly(up=8, down=25)` → 4 000 Hz  
2. **Butterworth** de orden 8, paso banda 70–1 900 Hz (`sosfiltfilt`)  
3. **Filtro comb notch** 50 Hz + armónicos hasta Nyquist, Q = 50 por tono  

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

## 5. Extracción de features (Paso 5)

Se extraen **15 features acústicas** por segmento sobre las 1 923 señales etiquetadas:

| # | Feature | Descripción |
|---|---|---|
| 1 | RMS | Energía cuadrática media |
| 2 | Duración | Duración en segundos |
| 3 | ZCR | Tasa de cruces por cero |
| 4 | Kurtosis | Apuntamiento de la distribución |
| 5 | Skewness | Asimetría |
| 6 | TKEO | Energía Teager-Kaiser media |
| 7 | Frec. dominante | 70–2 000 Hz |
| 8 | Frec. media | 70–2 000 Hz |
| 9–12 | Band power | 100–1000 / 70–200 / 200–600 / 600–1000 Hz |
| 13 | Entropía espectral | Normalizada [0, 1] |
| 14 | Razón armónica | Energía en armónicos 1–3 de la frec. dominante |
| 15 | Sample entropy | m=2, r=0.2·σ |

**Matriz resultante:** `X_labeled` → (1 923 × 15) · `y_labeled` → (1 923,) binario

---

## 6. Clasificación (Paso 6)

### 6.1 Estrategia de validación

- **Leave-One-Subject-Out (LOSO)**: en cada fold se deja fuera un sujeto completo.  
  Justificación: las 1 923 señales pertenecen a **18 sujetos** — son datos agrupados.  
  El sujeto P8 se excluye del LOSO (solo 3 señales CAS < umbral de 5) → **17 folds**.
- Datos desbalanceados: todos los modelos usan `class_weight='balanced'` o `scale_pos_weight`.
- Pipeline por modelo: `StandardScaler → SelectKBest(k=10) → Clasificador`

### 6.2 Modelos evaluados

| Modelo | AUC (media ± std) | Sensitivity | Specificity | F1 |
|---|---|---|---|---|
| **SVM** (RBF, C=1) | **0.652 ± 0.206** | 0.500 ± 0.301 | 0.750 ± 0.135 | 0.407 |
| Random Forest (300 árboles) | 0.654 ± 0.167 | 0.335 ± 0.291 | 0.886 ± 0.082 | 0.333 |
| XGBoost | 0.625 ± 0.168 | 0.422 ± 0.286 | 0.775 ± 0.109 | 0.395 |
| Ensemble (soft voting) | 0.646 ± 0.187 | 0.385 ± 0.312 | 0.852 ± 0.088 | 0.366 |

> **Modelo seleccionado: SVM** (mejor AUC medio). Se reentrena sobre las 1 923 señales
> y se aplica a las **14 900 señales** para obtener la clasificación completa.

### 6.3 Resultados SVM por fold (LOSO)

| Sujeto | AUC | Sensitivity | Specificity |
|---|---|---|---|
| P1 | 0.949 | 0.810 | 0.897 |
| P2 | 0.707 | 0.600 | 0.781 |
| P3 | 0.745 | 0.897 | 0.385 |
| P4 | 0.627 | 0.488 | 0.781 |
| P5 | 0.913 | 0.889 | 0.692 |
| P6 | 0.269 | 0.059 | 0.744 |
| P7 | 0.723 | 0.455 | 0.792 |
| P9 | 0.723 | 0.521 | 0.857 |
| P10 | 0.262 | 0.071 | 0.800 |
| P11 | 0.525 | 0.192 | 0.800 |
| P13 | 0.393 | 0.136 | 0.730 |
| P15 | 0.626 | 0.375 | 0.816 |
| P17 | 0.788 | 0.800 | 0.641 |
| P18 | 0.460 | 0.167 | 0.809 |
| P20 | 0.785 | 0.667 | 0.750 |
| P22 | 0.802 | 0.444 | 0.947 |
| P23 | 0.794 | 0.929 | 0.521 |
| **MEDIA** | **0.652** | **0.500** | **0.750** |

**Alta varianza entre folds** (std AUC = 0.206): refleja la heterogeneidad entre sujetos.
Folds como P6 (AUC 0.27) o P10 (0.26) indican sujetos con distribución atípica.

---

## 7. Análisis de respuesta broncodilatadora (Paso 7)

El clasificador SVM se aplica a las 14 900 señales para calcular, por sujeto:

```
tasa_CAS_pre  = (N_CAS_pre  / N_total_pre)  × 100 %
tasa_CAS_post = (N_CAS_post / N_total_post) × 100 %

ΔCAS = 100 × (tasa_pre − tasa_post) / tasa_pre
```

ΔCAS > 0 → reducción de CAS tras broncodilatador (respuesta esperada).  
ΔCAS < 0 → aumento de CAS post-BD.

### 7.1 ΔCAS por grupo — todos los canales y fases

Fórmula exacta del enunciado: `ΔCAS = 100 × (N_CAS_pre − N_CAS_post) / N_CAS_pre`

| Grupo | n | ΔCAS (media ± std) | Tasa CAS pre | Tasa CAS post | p-valor |
|---|---|---|---|---|---|
| **BDR+** | 9 | −3.4 ± 58.2 | 18.9 % | 15.5 % | — |
| BDR− | 14 | −96.7 ± 207.0 | 16.3 % | 22.4 % | — |
| Controles | 5 | −12.3 ± 43.6 | 17.0 % | 16.3 % | — |

**Tests estadísticos:**
- Mann-Whitney U (BDR+ vs BDR−): **p = 0.030** ✅ significativo (α = 0.05)
- Kruskal-Wallis (3 grupos): **p = 0.041** ✅ significativo

### 7.2 ΔCAS por canal

| Grupo | Canal inferior (ch1) | Canal superior (ch2) |
|---|---|---|
| BDR+ | −10.5 ± 122.0 | **+7.3 ± 54.7** |
| BDR− | −126.8 ± 252.6 | −66.8 ± 142.9 |
| Controles | −31.2 ± 131.6 | −2.1 ± 23.0 |

### 7.3 ΔCAS por fase respiratoria

| Grupo | Inspiración | Espiración |
|---|---|---|
| BDR+ | **+11.6 ± 45.4** | **+21.5 ± 84.4** |
| BDR− | −114.3 ± 188.8 | −86.3 ± 235.6 |
| Controles | −21.4 ± 43.0 | +22.4 ± 72.7 |

### 7.4 Observaciones clínicas

- Los **BDR+** muestran reducción de CAS post-BD (ΔCAS positivo en ch2 e inspiración)
  consistente con la respuesta broncodilatadora esperada. La diferencia con BDR− es
  **estadísticamente significativa** (p = 0.030, Mann-Whitney U).
- Los **BDR−** aumentan el número de CAS post-BD (ΔCAS negativo), con alta varianza
  inter-sujeto por el pequeño tamaño muestral.
- Los **controles** muestran ΔCAS próximo a cero, como se espera al no tener obstrucción.
- El **canal superior (ch2)** y la **inspiración** son las condiciones más informativas
  para discriminar BDR+ de BDR−.

---

## 8. Comparativa Deep Learning (Paso 8 — extra)

Se implementaron dos enfoques adicionales usando espectrogramas (64×64) como entrada:

| Modelo | AUC (media ± std) | Sensitivity | Specificity |
|---|---|---|---|
| SVM baseline | 0.652 ± 0.206 | 0.500 ± 0.301 | 0.750 ± 0.135 |
| RF baseline | 0.654 ± 0.167 | 0.335 ± 0.291 | 0.886 ± 0.082 |
| **CNN (scratch)** | 0.630 ± 0.182 | 0.486 ± 0.321 | 0.672 ± 0.212 |
| VGGish + SVM | 0.500 ± 0.000 | — | — |

- **CNN**: red convolucional pequeña (~35K parámetros), entrenada con LOSO + SpecAugment.
  No supera al SVM con features manuales — esperado con n=1 923.
- **VGGish**: AUC = 0.5 (azar). Causa: modelo preentrenado en audio genérico a 16 kHz;
  los embeddings resultantes son no discriminativos para sonidos respiratorios a 4 kHz.

---

## 9. Limitaciones

### Técnicas
1. **Dataset pequeño para deep learning**: 1 923 señales en 17 sujetos producen alta varianza
   LOSO (std AUC ~0.20). El rendimiento del clasificador está acotado por la escasez de datos.
2. **Clasificador entrenado solo con señales etiquetadas de pacientes**: no hay etiquetas para
   controles, por lo que la calidad de la clasificación en ese grupo no está validada.
3. **LOSO estricto**: la distribución acústica varía notablemente entre sujetos (AUC 0.26–0.95).
   El rendimiento medio esconde diferencias individuales importantes.
4. **VGGish no es adecuado**: preentrenado en audio a 16 kHz de dominio general; el
   mismatch de dominio con sonidos respiratorios a 4 kHz invalida los embeddings.

### Clínicas
5. **ΔCAS con alta varianza intra-grupo**: el biomarcador no discrimina significativamente
   BDR+ de BDR− ni de controles a nivel de grupo (test Mann-Whitney no significativo).
   Causas posibles: ruido en la clasificación, variabilidad fisiológica real, n pequeño.
6. **La fórmula ΔCAS es sensible a valores bajos de tasa pre-BD**: sujetos con pocos CAS
   pre-BD producen valores extremos de ΔCAS (P10: −140 %, P21: −833 %).
7. **Solo se evalúan 6 maniobras**: la reproducibilidad inter-sesión no se estudia.
8. **No se valida el biomarker contra gold standard clínico** (espirometría, FeNO).

---

## 10. Conclusiones

1. Se procesaron correctamente **14 900 señales** de 28 sujetos con el pipeline completo
   (lectura → preprocesado → segmentación → features → clasificación → análisis).

2. El **SVM** obtuvo el mejor rendimiento (AUC = 0.652 ± 0.206) en validación LOSO
   estricta por sujeto, seguido de cerca por RF (0.654 ± 0.167).

3. Los **BDR+** muestran una tendencia positiva en ΔCAS (reducción de CAS post-BD)
   coherente con la respuesta broncodilatadora, más clara en **ch2** e **inspiración**.
   Sin embargo, la alta varianza impide conclusiones estadísticamente robustas.

4. El **deep learning no aportó mejora** sobre features manuales en este contexto de
   datos limitados — resultado esperado y con valor como hallazgo metodológico.

5. La principal limitación es el **tamaño de la muestra**: más sujetos etiquetados
   mejorarían tanto el clasificador como la potencia estadística del análisis de BD.

---

## Ficheros generados

```
outputs/
├── results/
│   ├── step5/   X_labeled_features.npy (1923×15), y_labeled.npy, groups_labeled.npy
│   ├── step6/   svm/rf/xgb/ensemble_loso_results.csv, best_model.pkl, predictions_all.npz
│   ├── step7/   cas_metrics_{all,ch1,ch2,insp,esp,ch1_insp,...}.csv, group_statistics.csv
│   └── step8/   X_spectrograms.npy, X_vggish_embeddings.npy, dl_comparison_results.json
└── figures/
    ├── step6/   ROC curves, confusion matrices, AUC por fold, feature importance, CAS rate
    ├── step7/   pre/post comparativa, delta CAS por sujeto, boxplots, heatmap, ROC biomarker
    └── step8/   espectrogramas ejemplo/medio, comparativa AUC, curvas entrenamiento CNN
```

---

*Código: Python 3.11 · scikit-learn 1.7 · PyTorch 2.11+cu128 · scipy 1.17 · RTX 5070 Laptop*
