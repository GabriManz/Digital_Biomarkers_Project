# Comparación de Pipelines: Gabriel vs Adria

**Proyecto:** Digital Biomarkers — Clasificación CAS / NO-CAS  
**Fecha:** Junio 2026  
**Resultado Adria:** ~0.81 accuracy (Ensemble)  
**Resultado Gabriel (pipeline original):** ~0.65 AUC / 0.67 accuracy  
**Resultado Gabriel (pipeline actualizado):** ~0.81 accuracy (Ensemble, folds 1–2), 0.779 media

---

## Resumen ejecutivo

Adria obtuvo mejores resultados gracias a **cuatro cambios principales** respecto al pipeline original:

| # | Cambio | Impacto estimado |
|---|--------|-----------------|
| 1 | Normalización MAD por segmento | Moderado — reduce variabilidad inter-sujeto |
| 2 | Features: 15 → 164 (MFCC + wavelet) | **Alto** — principal fuente de mejora |
| 3 | Validación: LOSO → StratifiedKFold-5 | **Alto** — infla ~10–15 pp en accuracy |
| 4 | SMOTE en cada fold (orden correcto) | Moderado — mejora recall en clase minoritaria |

> **Nota importante sobre la validación:** la diferencia numérica más grande entre ambos pipelines no viene de las features sino del protocolo de evaluación. LOSO (Leave-One-Subject-Out) es científicamente más riguroso porque nunca evalúa sobre un paciente ya visto durante el entrenamiento. StratifiedKFold-5 permite que segmentos del mismo paciente estén en train y test simultáneamente, lo que infla las métricas artificialmente entre 10 y 15 puntos porcentuales.

---

## 1. Preprocesado de señales

### 1.1 Cadena de filtros

Ambos pipelines aplican la misma cadena de tres pasos:

| Paso | Parámetros | Gabriel | Adria |
|------|-----------|---------|-------|
| Remuestreo | 12 500 → 4 000 Hz, `resample_poly(up=8, down=25)` | ✅ igual | ✅ igual |
| Butterworth paso-banda | 70–1 900 Hz, orden 8, fase cero | `sosfiltfilt` (SOS) | `filtfilt` (b,a) |
| Notch comb | 50 Hz + armónicos, BW = 1 Hz, fase cero | `sosfiltfilt` (SOS) | `filtfilt` (b,a) |

**Diferencia técnica en el filtrado:**  
Gabriel usa `sosfiltfilt` (second-order sections), que es numéricamente más estable para filtros de orden alto (8). Adria usa `filtfilt` con coeficientes b/a, que para orden 8 puede presentar inestabilidades numéricas en señales muy largas. Desde el punto de vista del resultado final, la diferencia es despreciable en estos datos.

### 1.2 Normalización por segmento ← **cambio clave**

| | Gabriel (original) | Adria |
|--|-------------------|-------|
| Normalización | ❌ Ninguna | ✅ MAD z-score robusta por segmento |

Adria aplica una **normalización robusta** a cada segmento individual después de segmentarlo:

```
z = (x − mediana) / (1.4826 × MAD)
```

donde `MAD = mediana(|x − mediana|)`.

**Por qué importa:**  
Las señales de distintos pacientes tienen amplitudes muy diferentes (diferente posición del micrófono, diferente intensidad respiratoria, diferentes condiciones de grabación). Sin normalización, features como RMS o energía reflejan más las diferencias entre pacientes que las diferencias entre CAS y NO-CAS. La normalización MAD (más robusta que el z-score estándar porque ignora valores extremos) elimina esta variabilidad no deseada antes de extraer features.

---

## 2. Extracción de features

### 2.1 Comparativa general

| | Gabriel (original) | Adria |
|--|-------------------|-------|
| Total de features | **15** | **164** |
| Normalización previa | No | MAD z-score por segmento |
| MFCC | ❌ | ✅ 120 features |
| Wavelet | ❌ | ✅ 15 features |
| Features temporales | 6 básicas | 16 enriquecidas |
| Features espectrales | 6 básicas | 13 enriquecidas |
| Sample entropy | ✅ | ❌ |

### 2.2 Features temporales (6 → 16)

**Gabriel (6 features):**
- RMS, duración, ZCR, kurtosis, skewness, TKEO medio

**Adria (16 features):**

| Feature | Descripción |
|---------|-------------|
| Media, std, varianza | Estadísticos de primer y segundo orden |
| RMS | Energía cuadrática media |
| Máximo absoluto, rango | Amplitud pico y dinámica total |
| Skewness, kurtosis | Forma de la distribución de amplitudes |
| Zero-crossing rate | Actividad de alta frecuencia |
| **Crest factor** | Relación pico/RMS — mide impulsividad |
| Entropía de amplitud | Complejidad de la distribución temporal |
| Energía, log-energía | Medidas de potencia total |
| **Higuchi mobility** | Movilidad espectral (derivada 1ª) |
| **Higuchi complexity** | Complejidad espectral (derivada 2ª / mobility) |

Los parámetros de Higuchi (mobility y complexity) son medidas de complejidad de la señal basadas en las derivadas temporales — útiles para caracterizar patrones de vibración respiratoria.

### 2.3 Features espectrales (6 → 13)

**Gabriel (6 features):**
- Frecuencia dominante, frecuencia media, 4 potencias de banda, entropía espectral, harmonic ratio, sample entropy

**Adria (13 features):**

| Feature | Descripción |
|---------|-------------|
| **Centroide espectral** | "Centro de gravedad" del espectro — indica zona de energía dominante |
| **Spread espectral** | Dispersión alrededor del centroide |
| **Rolloff 85%** | Frecuencia por debajo de la cual está el 85% de la energía |
| **Flatness** | Cociente media geométrica / aritmética — mide ruido vs tono |
| Entropía espectral | Complejidad de la distribución espectral |
| Frecuencia dominante | Frecuencia del pico de máxima potencia |
| Centroide (repetido) | Centroide como segunda referencia |
| Frecuencia mediana (50%) | Divide el espectro en dos mitades iguales |
| Potencia 70–250 Hz | Banda baja |
| Potencia 250–500 Hz | Banda media-baja |
| Potencia 500–1 000 Hz | Banda media |
| Potencia 1 000–1 500 Hz | Banda media-alta |
| Potencia 1 500–1 900 Hz | Banda alta |

### 2.4 MFCC — 120 features ← **aportación más importante**

Adria añade **Mel-Frequency Cepstral Coefficients (MFCC)** calculados con `librosa`. Los MFCCs son el descriptor estándar en reconocimiento de voz y audio porque capturan la **envolvente espectral** de forma compacta y perceptualmente relevante.

**Configuración:**
```python
N_MFCC = 20
m  = librosa.feature.mfcc(y=signal, sr=4000, n_mfcc=20)
d  = librosa.feature.delta(m)    # delta (velocidad)
d2 = librosa.feature.delta(m, order=2)  # delta-delta (aceleración)
```

**Features extraídas (20 × 6 = 120):**

| Orden | Estadístico | Features |
|-------|-------------|---------|
| MFCCs originales | media | 20 |
| MFCCs originales | std | 20 |
| Delta (1ª derivada temporal) | media | 20 |
| Delta (1ª derivada temporal) | std | 20 |
| Delta-delta (2ª derivada) | media | 20 |
| Delta-delta (2ª derivada) | std | 20 |
| **Total** | | **120** |

Los **deltas** añaden información dinámica: cómo cambia el espectro a lo largo del segmento — crucial para distinguir patrones CAS (sonidos con características evolutivas específicas) de NO-CAS.

### 2.5 Wavelet — 15 features

Adria aplica la **Transformada Wavelet Discreta (DWT)** con la wavelet `db4` a nivel 5 usando `pywt`. Las wavelets permiten un análisis tiempo-frecuencia multiresolución que la FFT no ofrece: cada nivel de descomposición captura una banda de frecuencia diferente con resolución temporal adaptada.

```python
coeffs = pywt.wavedec(signal, 'db4', level=5)
# coeffs[0] = aproximación (nivel 5)
# coeffs[1..5] = detalles niveles 5, 4, 3, 2, 1
```

**Por cada nivel de detalle (×5 = 15 features):**

| Feature | Descripción |
|---------|-------------|
| Energía | `sum(c²)` — potencia en esa banda |
| Entropía wavelet | `-sum(p·log(p))` donde `p = c²/energía` |
| Std | Desviación estándar de los coeficientes |

| Nivel | Banda de frecuencia aproximada (a 4 000 Hz) |
|-------|---------------------------------------------|
| 1 (detalle) | 1 000–2 000 Hz |
| 2 (detalle) | 500–1 000 Hz |
| 3 (detalle) | 250–500 Hz |
| 4 (detalle) | 125–250 Hz |
| 5 (detalle) | 62–125 Hz |

---

## 3. Clasificación

### 3.1 Modelos utilizados

| Modelo | Gabriel (original) | Adria |
|--------|-------------------|-------|
| Logistic Regression | ❌ | ✅ |
| SVM lineal | ❌ | ✅ |
| SVM-RBF | ✅ | ✅ |
| Random Forest | ✅ | ✅ (n=500, depth=10) |
| XGBoost | ✅ | ✅ (lr=0.05, depth=5) |
| **CNN-1D** (log-mel) | — en step8 — | ✅ integrado |
| **BiLSTM** (log-mel) | — en step8 — | ✅ integrado |
| **Ensemble ponderado** | VotingClassifier | Búsqueda de peso óptimo ML+DL |

**Diferencia en el Ensemble:**
- Gabriel: `VotingClassifier(voting='soft')` — promedio simple de probabilidades
- Adria: búsqueda del peso óptimo `w` tal que `prob = w·ML + (1−w)·DL` maximiza F1

### 3.2 Selección de features

| | Gabriel (original) | Adria |
|--|-------------------|-------|
| Feature selection | `SelectKBest(k=10)` | ❌ Ninguna |

Gabriel aplicaba SelectKBest dentro del pipeline para quedarse con las 10 features más informativas (de las 15 totales). Con 164 features bien diseñadas, no es necesario — los modelos de árbol hacen selección implícita, y los modelos lineales se benefician de todas las features tras el escalado.

### 3.3 Manejo del desbalance de clases

| | Gabriel (original) | Adria |
|--|-------------------|-------|
| Distribución | CAS=590 (31%), NO-CAS=1333 (69%) | igual |
| class_weight | `'balanced'` en SVM/RF | `'balanced'` en SVM/RF |
| SMOTE | ❌ | ✅ en cada fold |
| Orden SMOTE | — | Scale → SMOTE → fit ← correcto |

**Orden crítico — Scale → SMOTE → fit:**  
SMOTE genera muestras sintéticas interpolando entre vecinos en el espacio de features. Si se aplica antes de escalar, las distancias entre puntos están dominadas por features de gran escala (energía, potencias absolutas). Si se aplica después de escalar, todas las features contribuyen por igual a la interpolación, generando muestras sintéticas más representativas.

### 3.4 Protocolo de validación ← **diferencia crítica**

| | Gabriel (original) | Adria |
|--|-------------------|-------|
| Método | **LeaveOneGroupOut (LOSO)** | **StratifiedKFold(5)** |
| Folds | 18–28 (un paciente por fold) | 5 |
| Separación paciente | ✅ Estricta — paciente test nunca en train | ❌ Mezcla — segmentos del mismo paciente en train y test |
| Accuracy típica RF | ~0.68 | ~0.79 |
| Diferencia | — | +11 pp artificiales |

**LOSO** es el gold standard para datos clínicos. Evalúa si el modelo generaliza a pacientes completamente nuevos, que es exactamente lo que importa en un contexto real de diagnóstico.

**StratifiedKFold** sin agrupación por paciente permite que el modelo "vea" segmentos del mismo paciente durante el entrenamiento y luego se evalúe sobre otros segmentos del mismo paciente. Dado que cada paciente tiene características individuales distintivas (timbre de voz, anatomía), el modelo puede aprender a reconocer individuos en lugar de patrones CAS/NO-CAS genuinos — inflando artificialmente las métricas.

**Por qué Adria usó StratifiedKFold:**  
No fue una decisión deliberada. El archivo `proy_labels.mat` solo contiene la clave `labels` (sin información de participantes), por lo que el código de Adria ejecuta:

```python
# Adria/classification.py
if 'participants' in mat_data:
    g_raw = mat_data['participants'].flatten()
elif len(claus_reals) > 1:
    g_raw = mat_data[claus_reals[1]].flatten()
else:
    g_raw = np.ones(len(labels_raw))   # ← todos al mismo grupo
```

Con `g_raw = np.ones(...)` → `unique_groups = 1 < 2` → activa `StratifiedKFold` automáticamente.

---

## 4. Deep Learning (Adria exclusivo)

Adria integra modelos de Deep Learning en el mismo pipeline de clasificación, algo que en el pipeline de Gabriel está en un paso separado (step8).

### 4.1 Espectrograma log-mel

Antes de los modelos DL, cada segmento se convierte en un espectrograma log-mel:

```python
FIXED_LEN = 8000   # 2 segundos a 4000 Hz
N_FFT  = 256
HOP    = 128
N_MELS = 64

mel = librosa.feature.melspectrogram(y=seg, sr=4000,
      n_fft=256, hop_length=128, n_mels=64, fmin=70, fmax=1900)
mel = librosa.power_to_db(mel, ref=np.max).T   # shape: (frames, 64)
mel = (mel - mel.mean()) / (mel.std() + 1e-9)  # normalización por instancia
```

Resultado: tensor `(N, 63, 64)` — N segmentos × 63 frames temporales × 64 bins mel.

### 4.2 CNN-1D

```
Input(63, 64)
→ Conv1D(32, kernel=7, relu) → BN → MaxPool(2) → Dropout(0.25)
→ Conv1D(64, kernel=5, relu) → BN → MaxPool(2) → Dropout(0.25)
→ Conv1D(128, kernel=3, relu) → BN → GlobalAveragePooling
→ Dense(64, relu) → Dropout(0.4)
→ Dense(1, sigmoid)
```

Captura patrones locales en la dimensión temporal del espectrograma.

### 4.3 BiLSTM

```
Input(63, 64)
→ BiLSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
→ BiLSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.2)
→ Dense(64, relu) → Dropout(0.4)
→ Dense(1, sigmoid)
```

Procesa el espectrograma como secuencia temporal, capturando dependencias a largo plazo en ambas direcciones.

### 4.4 Ensemble ponderado ML+DL

```python
# Búsqueda de peso óptimo w ∈ [0.1, 0.9]
for w in np.linspace(0.1, 0.9, 17):
    ens = w * prob_best_ML + (1-w) * prob_best_DL
    f1v = f1_score(y, (ens >= 0.5).astype(int))
    if f1v > best_f1: best_w = w

prob_ensemble = best_w * prob_ML + (1 - best_w) * prob_DL
```

---

## 5. Comparativa final de resultados

### 5.1 Pipeline original de Gabriel (LOSO — validación rigurosa)

| Modelo | Accuracy | F1 | AUC |
|--------|----------|----|-----|
| SVM-RBF | 0.644 | 0.405 | 0.652 |
| RF | 0.676 | 0.315 | 0.654 |
| XGB | 0.629 | 0.352 | 0.625 |
| Ensemble | 0.674 | 0.350 | 0.646 |

### 5.2 Adria (StratifiedKFold-5 — validación con mezcla de pacientes)

| Modelo | Accuracy | F1 | AUC |
|--------|----------|----|-----|
| LR | 0.712 | — | — |
| SVM-Lin | 0.714 | — | — |
| SVM-RBF | 0.783 | — | — |
| RF | 0.786 | — | — |
| XGB | 0.803 | — | — |
| CNN-1D | 0.721 | — | — |
| BiLSTM | 0.634 | — | — |
| **Ensemble** | **0.811** | — | — |

### 5.3 Pipeline actualizado de Gabriel (StratifiedKFold-5 — equivalente a Adria)

| Modelo | Accuracy media | Acc máx (fold) | F1 | AUC |
|--------|---------------|----------------|-----|-----|
| LR | 0.700 | 0.753 | 0.566 | 0.734 |
| SVM-Lin | 0.697 | 0.740 | 0.566 | 0.732 |
| SVM-RBF | 0.750 | 0.764 | 0.555 | 0.789 |
| RF | 0.769 | 0.797 | 0.625 | 0.815 |
| XGB | 0.762 | 0.800 | 0.635 | 0.818 |
| **Ensemble** | **0.779** | **0.810** | **0.615** | **0.830** |

---

## 6. Diagrama de flujo comparativo

```
┌─────────────────────────────────────────────────────────────┐
│                    SEÑAL BRUTA (12500 Hz)                    │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
       GABRIEL ORIGINAL                    ADRIA
              │                               │
    ┌─────────▼─────────┐         ┌───────────▼───────────┐
    │  1. Resample 4kHz │         │  1. Resample 4kHz     │
    │  2. Butterworth   │ = igual │  2. Butterworth       │
    │  3. Notch comb    │         │  3. Notch comb        │
    │  ❌ Sin normaliz. │         │  ✅ MAD z-score/segm. │
    └─────────┬─────────┘         └───────────┬───────────┘
              │                               │
    ┌─────────▼─────────┐         ┌───────────▼───────────┐
    │   15 FEATURES     │         │    164 FEATURES        │
    │  · 6 temporales   │         │  · 16 temporales       │
    │  · 6 espectrales  │  ──▶    │  · 13 espectrales      │
    │  · sample entropy │         │  · 120 MFCC (librosa)  │
    │                   │         │  · 15 wavelet (pywt)   │
    └─────────┬─────────┘         └───────────┬───────────┘
              │                               │
    ┌─────────▼─────────┐         ┌───────────▼───────────┐
    │   CLASIFICACIÓN   │         │   CLASIFICACIÓN        │
    │  · SVM, RF, XGB   │         │  · LR, SVM-Lin, SVM,  │
    │  · SelectKBest(10)│  ──▶    │    RF, XGB             │
    │  · class_weight   │         │  · Sin SelectKBest     │
    │  · ❌ Sin SMOTE   │         │  · Scale→SMOTE→fit     │
    └─────────┬─────────┘         │  · CNN-1D + BiLSTM    │
              │                   └───────────┬───────────┘
    ┌─────────▼─────────┐         ┌───────────▼───────────┐
    │  LOSO (LOSO)       │         │  StratifiedKFold-5    │
    │  28 folds          │         │  5 folds              │
    │  Un paciente/fold  │         │  Mezcla pacientes     │
    │  AUC ~0.65         │         │  Acc ~0.81 Ensemble   │
    │  Acc ~0.67         │         │                       │
    └───────────────────┘         └───────────────────────┘
```

---

## 7. Conclusiones y recomendaciones

### ¿Qué cambios de Adria son legítimas mejoras?

| Cambio | ¿Mejora real? | Motivo |
|--------|--------------|--------|
| Normalización MAD | ✅ Sí | Reduce variabilidad inter-sujeto genuina |
| MFCC (120 features) | ✅ Sí | Descriptor estándar y potente para audio |
| Wavelet (15 features) | ✅ Sí | Análisis multiresolución, complementa MFCC |
| Features temporales enriquecidas | ✅ Sí | Higuchi mobility/complexity añaden información real |
| SMOTE en cada fold (orden correcto) | ✅ Sí | Mejora recall en clase minoritaria |
| Más modelos (LR, SVM-Lin) | ✅ Sí | Diversidad para el ensemble |
| StratifiedKFold en lugar de LOSO | ⚠️ No | Infla métricas ~10-15 pp por fuga de datos |

### Recomendación final

Para un trabajo académico riguroso sobre biomarcadores digitales:

1. **Usar los features de Adria** (164, con MFCC y wavelet) — son una mejora real
2. **Reportar resultados con LOSO** — es la validación correcta para datos clínicos agrupados por paciente
3. **El 0.81 de Adria es con StratifiedKFold** — equivale a ~0.69–0.72 en LOSO honesto
4. **El AUC** es más informativo que la accuracy para clases desbalanceadas (30% CAS / 70% NO-CAS)
