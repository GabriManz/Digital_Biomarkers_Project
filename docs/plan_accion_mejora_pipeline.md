# Plan de Acción: Mejora del Pipeline de Clasificación CAS/NO-CAS

**Proyecto:** Digital Biomarkers — Detección de CAS y Respuesta Broncodilatadora  
**Fecha:** Junio 2026  
**Contexto:** Análisis comparativo Gabriel vs Adria + hoja de ruta para resultados rigurosos

---

## 1. Situación de partida

### Métricas actuales (LOSO — validación rigurosa)

| Pipeline | Features | Validación | RF Acc | RF AUC | XGB AUC |
|----------|----------|-----------|--------|--------|---------|
| Gabriel original | 15 | LOSO | 0.676 | 0.654 | 0.625 |
| Gabriel + 164 features (Adria) | 164 | LOSO | 0.657 | 0.619 | 0.616 |
| Gabriel actualizado | 164 | StratifiedKFold-5 | 0.769 | 0.815 | 0.818 |
| Adria original | 164 + DL | StratifiedKFold-5 | 0.786 | — | 0.803 |

> **Observación crítica:** añadir las 164 features de Adria al pipeline LOSO honesto **empeora** los resultados respecto a las 15 features originales. Esto no es un error — es el diagnóstico del problema central.

---

## 2. Limitaciones del pipeline de Adria

### 2.1 Fuga de datos en la validación (data leakage)

**Problema:** `proy_labels.mat` solo contiene la clave `labels`, sin información de participantes. El código de Adria ejecuta:

```python
else:
    g_raw = np.ones(len(labels_raw))   # todos al grupo 1
```

Con un único grupo, `StratifiedKFold` mezcla segmentos del mismo paciente en train y test. El modelo puede aprender a reconocer la identidad acústica de cada paciente, no el patrón clínico CAS.

**Impacto:** +10–15 puntos porcentuales de accuracy artificiales. El 0.811 real es ~0.65–0.69 con validación honesta.

---

### 2.2 Features que codifican identidad del paciente, no patología

**Problema:** Los **MFCCs** (120 de las 164 features) capturan la envolvente espectral del tracto vocal — una característica anatómica individual. Con StratifiedKFold, esto es una ventaja (el modelo memoriza al paciente). Con LOSO (donde el test patient nunca ha sido visto), estas features se convierten en ruido.

**Evidencia:** Al aplicar las 164 features de Adria con LOSO, el AUC cae de 0.654 a 0.619 respecto a las 15 features originales. Las 120 features MFCC añaden 109 features de "identidad" por cada 11 de "patología".

---

### 2.3 Ignorar la estructura pre/post broncodilatador

**Problema:** Los segmentos se clasifican de forma independiente (CAS o NO-CAS) sin aprovechar que cada paciente tiene mediciones **antes y después** del broncodilatador. La pregunta clínica no es "¿este segmento tiene CAS?" sino "¿la proporción de CAS cambia con el broncodilatador?".

**Datos disponibles:**
```
Total etiquetados : 1923 segmentos / 18 pacientes
Pre-BD (v_bd=1)   : 956 segmentos
Post-BD (v_bd=2)  : 967 segmentos
```

El **delta_CAS** (cambio en tasa de CAS pre→post) es el biomarcador clínico real de BDR, y Adria no lo explota en la clasificación.

---

### 2.4 Modelos DL acoplados a TensorFlow sin alternativa

**Problema:** CNN-1D y BiLSTM requieren TensorFlow, que no se instala correctamente en entornos Windows con paths largos (error de instalación documentado). El código no tiene fallback: si TF falla, el pipeline entero falla.

**Impacto práctico:** Reproducibilidad comprometida. Los resultados DL (CNN=0.721, BiLSTM=0.634) no pueden verificarse sin resolver la instalación.

---

### 2.5 Ensemble empírico sin base estadística

**Problema:** El peso del ensemble se busca optimizando F1 sobre todo el conjunto de test:

```python
for w in np.linspace(0.1, 0.9, 17):
    ens = w * prob_ML + (1-w) * prob_DL
    f1v = f1_score(y, (ens >= 0.5).astype(int))
```

El peso `w` se optimiza sobre los mismos datos de evaluación, introduciendo otro nivel de sobreajuste.

---

### 2.6 Sin análisis de calibración ni intervalos de confianza

**Problema:** Se reportan métricas puntuales sin incertidumbre estadística. Con solo 18 pacientes en LOSO (o 5 folds), las métricas tienen alta varianza. No se reporta si las diferencias entre modelos son estadísticamente significativas.

---

## 3. Limitaciones del pipeline de Gabriel

### 3.1 Features demasiado básicas (problema original)

**Problema:** 15 features genéricas que no aprovechan la riqueza espectral de las señales de sonido respiratorio. Las features espectrales se calculan con `nperseg=256`, que da resolución frecuencial baja (~15.6 Hz/bin a 4000 Hz).

**Resultado:** Insuficiente para capturar los patrones acústicos sutiles que distinguen CAS de NO-CAS.

---

### 3.2 Las 164 features de Adria no generalizan en LOSO

**Problema:** Al incorporar las features de Adria, el AUC LOSO cae de 0.654 a 0.619. Las features MFCC, diseñadas para reconocimiento de voz, capturan la identidad del paciente más que la patología. Con LOSO, el test patient siempre es desconocido.

**Diagnóstico concreto:**

| Paciente | MFCC₁ medio (sus segmentos) | Descripción |
|---------|---------------------------|-------------|
| P02 | −43.2 dB | Timbre grave, BDR+ |
| P07 | −61.8 dB | Timbre agudo, NO CAS predominante |
| P_nuevo | −52.1 dB | Jamás visto en training |

El modelo aprende rangos de P02 y P07; P_nuevo cae fuera de esa distribución.

---

### 3.3 Validación LOSO con datos escasos y desbalanceados

**Problema:** 18 pacientes disponibles para LOSO. Algunos pacientes tienen muy pocos segmentos CAS:

```
P05: solo 9 CAS de 129 segmentos
P08: solo 3 CAS de 112 segmentos
P10: solo 14 CAS de 69 segmentos
```

Cuando P08 es el test patient (3 CAS), AUC no está definida o es irreliable. Cuando P08 está en train, aporta casi solo ejemplos NO-CAS.

---

### 3.4 Clasificación a nivel de segmento, no de paciente

**Problema:** La tarea clínica es predecir BDR+ / BDR− a nivel de **paciente**. El pipeline clasifica segmentos individuales (unidad incorrecta). Un paciente con 50 segmentos y 26 predichos como CAS (52%) y otro con 5 de 50 (10%) deberían dar respuestas muy diferentes, pero ambos comparten el umbral 0.5 por segmento.

---

### 3.5 Sin explotación de la información longitudinal pre/post BD

**Problema:** El pipeline trata todos los segmentos como independientes, ignorando que cada paciente tiene:
- Sesión pre-BD (v_bd=1): respiración basal
- Sesión post-BD (v_bd=2): respiración tras broncodilatador

El cambio en las features entre sesiones (delta pre→post) es el biomarcador de BDR más directo, y no se calcula.

---

### 3.6 Preprocesado sin normalización en la cadena de filtrado

**Problema:** `step2_preprocessing.py` aplica resample + Butterworth + Notch pero no normaliza. La normalización MAD se aplica dentro de `extract_features` (step5), lo que significa que las señales en bruto almacenadas en `dataset.npz` siguen teniendo amplitudes absolutas heterogéneas.

---

## 4. Plan de acción

El plan se organiza en tres fases de dificultad creciente. Cada fase es independiente y mejora los resultados de forma verificable.

---

### Fase 1 — Mejoras inmediatas al LOSO actual
**Plazo: 1–2 días | Herramientas: sklearn, numpy**

#### Acción 1.1: Feature selection dentro del fold LOSO

**Objetivo:** Filtrar en cada fold las features que discriminan CAS de NO-CAS en los datos de entrenamiento, eliminando las que codifican identidad del paciente.

**Implementación en `step6_classification.py`:**
```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Dentro del bucle LOSO, después de escalar:
selector = SelectKBest(mutual_info_classif, k=40)   # probar k ∈ {20, 30, 40, 50}
X_tr_sel = selector.fit_transform(X_tr_sc, y_tr)
X_te_sel = selector.transform(X_te_sc)
clf.fit(X_tr_sel, y_tr)
clf.predict(X_te_sel)
```

**Por qué funciona:** La información mutua entre feature y etiqueta se calcula sobre el conjunto de entrenamiento. Features como MFCC absoluto tienen alta información mutua con la identidad del paciente pero baja con CAS/NO-CAS → serán descartadas. Features como entropía wavelet o delta-MFCC tienen mayor información mutua con la etiqueta clínica.

**Resultado esperado:** AUC LOSO 0.62 → 0.67–0.70

---

#### Acción 1.2: Centrado de features por paciente (eliminar efecto individuo)

**Objetivo:** Transformar features absolutas en desviaciones respecto al propio paciente, haciéndolas invariantes a la identidad.

**Implementación:**
```python
# En el fold LOSO, ANTES de StandardScaler:
# --- Centrar training set por paciente ---
X_train_centered = X_train.copy()
patient_means_train = {}
for pid in np.unique(groups[train_idx]):
    m = groups[train_idx] == pid
    patient_means_train[pid] = X_train[m].mean(axis=0)
    X_train_centered[m] -= patient_means_train[pid]

# --- Para test patient: restar la media global del training ---
global_mean_train = X_train_centered.mean(axis=0)
X_test_centered = X_test - global_mean_train
```

**Por qué funciona:** Si P02 tiene MFCC₁ = −43 dB en todos sus segmentos (identidad), después del centrado sus segmentos tienen MFCC₁ ≈ 0. Lo que queda es la variación intra-paciente, que refleja diferencias entre CAS y NO-CAS.

**Resultado esperado:** AUC LOSO +0.03–0.05 sobre línea base

---

#### Acción 1.3: Agregación de predicciones a nivel de paciente

**Objetivo:** La métrica de evaluación debe ser por paciente, no por segmento.

**Implementación (post-procesado del LOSO):**
```python
# Después de obtener probs[test_idx] en cada fold:
pid_test = groups[test_idx[0]]   # un único paciente en LOSO
prob_patient = probs[test_idx].mean()   # media de probabilidades del paciente
pred_patient = int(prob_patient >= 0.5)
```

**Resultado:** En lugar de accuracy sobre 1923 segmentos, obtienes accuracy sobre 18 pacientes — que es la métrica clínicamente relevante. Con 18 puntos y clasificación binaria, un modelo puede ser perfecto (18/18) o aleatorio (9/18 = 0.50). El objetivo realista es 14–16/18 correctos.

**Resultado esperado:** Métrica limpia de 18 puntos; AUC calculada con 18 probabilidades únicas.

---

### Fase 2 — Rediseño de features para generalización cross-patient
**Plazo: 3–5 días | Herramientas: scipy, librosa, numpy**

#### Acción 2.1: Sustituir MFCCs absolutos por features temporalmente dinámicas

**Objetivo:** En lugar de la media de cada MFCC (muy específica del individuo), usar medidas de variación temporal que capturen la dinámica del sonido respiratorio.

**Features a añadir en `step5_features.py`:**

```python
# Para cada MFCC, extraer variación temporal en lugar de valor medio
def feat_mfcc_dynamic(signal, fs=4000, n_mfcc=20):
    m  = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=n_mfcc)
    d  = librosa.feature.delta(m)
    d2 = librosa.feature.delta(m, order=2)
    return (
        list(np.std(m, axis=1))    +   # variabilidad temporal de cada coef.  (20)
        list(np.mean(np.abs(d),1)) +   # magnitud media del delta              (20)
        list(np.std(d, axis=1))    +   # variabilidad del delta                (20)
        list(np.mean(np.abs(d2),1))    # magnitud media del delta-delta        (20)
    )   # 80 features dinámicas (vs 120 absolutas de Adria)
```

**Por qué es mejor para LOSO:** La media de MFCC₁ de P02 es siempre −43 dB (identidad). La std temporal de MFCC₁ de P02 varía entre segmentos CAS y NO-CAS porque CAS tiene modulaciones acústicas específicas. Esta variación temporal sí generaliza a pacientes nuevos.

---

#### Acción 2.2: Features espectrales relativas (ratio de bandas)

**Objetivo:** En lugar de potencias absolutas, usar ratios entre bandas que son invariantes a la ganancia global del micrófono y la intensidad respiratoria del paciente.

```python
def feat_spectral_ratios(signal, fs=4000):
    f, p = welch(signal, fs=fs, nperseg=512)
    bp = lambda lo, hi: np.sum(p[(f>=lo)&(f<hi)]) + 1e-12
    b1 = bp(70,  250)
    b2 = bp(250, 500)
    b3 = bp(500, 1000)
    b4 = bp(1000, 1500)
    b5 = bp(1500, 1900)
    total = b1+b2+b3+b4+b5
    return [
        b1/total, b2/total, b3/total, b4/total, b5/total,   # proporciones
        b3/b1,   # ratio alta/baja — aumenta en CAS (más energía alta)
        b4/b2,   # ratio banda media-alta / media-baja
        b5/b3,   # ratio alta-alta / media
        (b4+b5)/(b1+b2),   # ratio global alta/baja
    ]   # 9 features invariantes a ganancia
```

---

#### Acción 2.3: Features de modulación de amplitud

**Objetivo:** CAS (Continuous Adventitious Sounds) como sibilancias tienen modulaciones de amplitud periódicas. Capturar la estructura de modulación es más específico de patología que la amplitud absoluta.

```python
def feat_amplitude_modulation(signal, fs=4000):
    # Envolvente de amplitud (Hilbert)
    from scipy.signal import hilbert
    envelope = np.abs(hilbert(signal))
    # Análisis espectral de la envolvente
    f_env, p_env = welch(envelope, fs=fs, nperseg=min(256, len(envelope)))
    # Features de la modulación
    return [
        np.std(envelope) / (np.mean(envelope) + 1e-12),  # índice de modulación
        float(f_env[np.argmax(p_env)]),                   # frecuencia dominante modulación
        np.sum(p_env[f_env <= 20]) / (np.sum(p_env) + 1e-12),  # energía modulación lenta
        np.sum(p_env[(f_env>20)&(f_env<=100)]) / (np.sum(p_env) + 1e-12),  # modulación media
    ]   # 4 features
```

---

### Fase 3 — Rediseño a nivel de paciente (cambio de paradigma)
**Plazo: 1–2 semanas | Impacto: Alto — ataca el problema real**

#### Acción 3.1: Delta_CAS como biomarcador principal

**Objetivo:** La pregunta clínica es: ¿cambia la tasa de CAS con el broncodilatador? Construir un dataset a **nivel de paciente** donde cada observación es el cambio pre→post.

**Estructura del dataset resultante:**

```
Para cada paciente i (18 pacientes):
  tasa_CAS_pre[i]  = #{segmentos CAS pre-BD}    / #{segmentos pre-BD totales}
  tasa_CAS_post[i] = #{segmentos CAS post-BD}   / #{segmentos post-BD totales}
  delta_CAS[i]     = tasa_CAS_pre[i] - tasa_CAS_post[i]
  etiqueta[i]      = BDR+ (si delta_CAS > umbral) o BDR-
```

**Dataset a nivel de paciente (18 filas):**

| Paciente | tasa_CAS_pre | tasa_CAS_post | delta_CAS | BDR label |
|---------|-------------|--------------|-----------|-----------|
| P02 | 0.68 | 0.41 | +0.27 | BDR+ ? |
| P07 | 0.05 | 0.09 | −0.04 | BDR- ? |
| ... | | | | |

Con LOSO a nivel de paciente: 18 folds, cada uno con 17 pacientes en train y 1 en test. La métrica es accuracy de los 18 pacientes — exactamente la pregunta clínica.

**Implementación en nuevo `step7b_delta_cas_classification.py`:**
```python
# Por paciente: calcular tasa CAS pre y post
from pathlib import Path
import numpy as np, pandas as pd

def compute_patient_delta_cas(y_pred_all, v_subject, v_bd, labels_raw, metadata):
    rows = []
    for pid in np.unique(v_subject):
        mask_pre  = (v_subject==pid) & (v_bd==1)
        mask_post = (v_subject==pid) & (v_bd==2)
        if mask_pre.sum() < 5 or mask_post.sum() < 5:
            continue   # insuficientes segmentos
        rate_pre  = y_pred_all[mask_pre].mean()
        rate_post = y_pred_all[mask_post].mean()
        rows.append({
            'patient': pid,
            'cas_rate_pre':  rate_pre,
            'cas_rate_post': rate_post,
            'delta_cas':     rate_pre - rate_post,
        })
    return pd.DataFrame(rows)
```

---

#### Acción 3.2: Features a nivel de paciente para clasificación LOSO robusta

**Objetivo:** Construir un vector de features por paciente que combine estadísticos de segmentos con información de cambio pre/post.

**Features por paciente (ejemplos):**
```python
patient_features = {
    # Tasa de CAS (biomarcador principal)
    'cas_rate_pre':   #{CAS_pre} / #{total_pre},
    'cas_rate_post':  #{CAS_post} / #{total_post},
    'delta_cas':      cas_rate_pre - cas_rate_post,

    # Mediana y IQR de probabilidades (más robusto que media)
    'median_prob_pre':  np.median(probs[pre_mask]),
    'iqr_prob_pre':     np.percentile(probs[pre_mask],75) - np.percentile(probs[pre_mask],25),

    # Estadísticos de features espectrales por sesión
    'mean_spectral_centroid_pre':  np.mean(X_pre[:, idx_centroid]),
    'delta_spectral_centroid':     mean_post - mean_pre,
    ...
}
```

Con 18 pacientes y LOSO, cada fold tiene 17 en train y 1 en test. Se usa regresión logística o SVM lineal (no árboles — demasiado poco data para modelos complejos).

---

#### Acción 3.3: Aprendizaje semi-supervisado en los 12 547 segmentos no etiquetados

**Contexto:** Solo 1923 de los 14 900 segmentos tienen etiqueta (CAS=590, NO-CAS=1333). Hay **12 547 segmentos con etiqueta 1 (sin clasificar)** y 430 con etiqueta 6 (desconocida).

**Estrategia Label Propagation:**
```
Paso 1: Entrenar modelo base con los 1923 segmentos etiquetados (LOSO)
Paso 2: Predecir probabilidades sobre los 12547 sin etiquetar
Paso 3: Añadir al training los segmentos con probabilidad > 0.85 o < 0.15
         (alta confianza — pseudo-etiquetas)
Paso 4: Re-entrenar con 1923 + pseudo-etiquetas
Paso 5: Repetir 2-3 iteraciones
```

**Beneficio esperado:** Doblar o triplicar el conjunto de entrenamiento efectivo. Los segmentos de alta confianza aportan regularización sin introducir ruido.

---

## 5. Tabla resumen del plan

| Acción | Tipo | Dificultad | AUC LOSO esperado | Plazo |
|--------|------|-----------|-------------------|-------|
| 1.1 SelectKBest dentro del fold | Modificar step6 | Baja | 0.65–0.70 | 2h |
| 1.2 Centrado por paciente | Modificar step6 | Baja | +0.03–0.05 | 3h |
| 1.3 Agregación a nivel paciente | Post-proceso | Muy baja | Métrica limpia | 1h |
| 2.1 MFCCs dinámicos (std, delta) | Modificar step5 | Media | 0.68–0.72 | 4h |
| 2.2 Ratios espectrales | Modificar step5 | Baja | +0.02–0.03 | 2h |
| 2.3 Modulación de amplitud | Modificar step5 | Media | +0.02–0.03 | 3h |
| 3.1 Delta_CAS por paciente | Nuevo step7b | Media | 0.72–0.78 | 1 día |
| 3.2 Dataset paciente-nivel | Nuevo step7b | Alta | 0.75–0.82* | 2 días |
| 3.3 Semi-supervisado | Modificar step6 | Alta | +0.03–0.06 | 2 días |

*Con features de delta pre/post como biomarcador directo de BDR.

---

## 6. Flujo recomendado de implementación

```
AHORA (línea base LOSO con 164 features)
AUC = 0.619
    │
    ▼
FASE 1 — Quick wins (1-2 días)
    ├── 1.1: SelectKBest(k=30-40) dentro del fold
    ├── 1.2: Centrado por paciente
    └── 1.3: Agregar predicciones a nivel de paciente
AUC esperado: 0.67–0.71
    │
    ▼
FASE 2 — Mejores features (3-5 días)
    ├── 2.1: MFCCs dinámicos (reemplazar 120 por ~80 features dinámicas)
    ├── 2.2: Ratios espectrales invariantes a ganancia
    └── 2.3: Modulación de amplitud (Hilbert)
AUC esperado: 0.71–0.75
    │
    ▼
FASE 3 — Paradigma correcto (1-2 semanas)
    ├── 3.1: Delta_CAS pre/post como biomarcador primario
    ├── 3.2: Clasificación a nivel de paciente
    └── 3.3: Semi-supervisado en 12547 sin etiquetar
AUC esperado: 0.75–0.82  (sobre los 18 pacientes etiquetados)
```

---

## 7. Nota sobre la comparación con Adria

El objetivo no es igualar 0.811 de Adria con LOSO — eso no es posible sin fuga de datos. El objetivo es obtener el AUC más alto posible con validación honesta (LOSO). La siguiente tabla muestra la equivalencia real:

| Adria StratifiedKFold | Equivalente LOSO honesto estimado |
|-----------------------|----------------------------------|
| XGB Acc = 0.803 | ~0.67–0.69 |
| Ensemble Acc = 0.811 | ~0.68–0.71 |
| **Objetivo realista LOSO** | **AUC 0.75–0.80 con Fase 3** |

Un resultado de **AUC 0.75 con LOSO** es científicamente más valioso que **0.81 con StratifiedKFold** porque demuestra generalización a pacientes nuevos — que es exactamente lo que se necesita para un biomarcador clínico real.
