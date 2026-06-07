import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Define PDF Path
ROOT_DIR = r"c:\DATA\01_Proyectos\Master\Digital_Biomarkers\Project"
PDF_PATH = os.path.join(ROOT_DIR, "Guia_Presentacion_15min.pdf")

def create_presentation_guide():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#002B49'),
        alignment=1, # Center
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#5C768D'),
        alignment=1, # Center
        spaceAfter=20
    )
    
    h1_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#002B49'),
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SlideHeading',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=14.5,
        textColor=colors.HexColor('#008080'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )
    
    tip_style = ParagraphStyle(
        'TipText',
        parent=body_style,
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#7F8C8D')
    )
    
    story = []
    
    # Title & Header
    story.append(Paragraph("Guía de Defensa de Presentación de 15 Minutos", title_style))
    story.append(Paragraph("Análisis Acústico de Sonidos Respiratorios y Evaluación de la Respuesta Broncodilatadora (BDR)", subtitle_style))
    story.append(Spacer(1, 10))
    
    intro_text = (
        "Esta guía proporciona una estructura detallada diapositiva por diapositiva para la defensa "
        "del proyecto, minimizando el énfasis en la introducción y preprocesamiento (comunes a todos los grupos) "
        "y enfocándose firmemente en la justificación científica de los biomarcadores, los modelos de ML/DL usados "
        "y el análisis clínico final del biomarcador Delta CAS junto a las limitaciones metodológicas."
    )
    story.append(Paragraph(intro_text, body_style))
    story.append(Spacer(1, 10))
    
    # ---------------------------------------------------------
    # SECTION 1: TIME DISTRIBUTION
    # ---------------------------------------------------------
    story.append(Paragraph("1. Distribución y Cronograma del Tiempo", h1_style))
    
    time_data = [
        ["Diapositiva", "Título de la Slide", "Tiempo", "Enfoque Principal"],
        ["Slide 1 (I1)", "Portada y Contexto Clínico", "1:30 min", "Objetivo de BDR y biomarcador Delta CAS."],
        ["Slide 2 (I1)", "Preprocesamiento y Segmentación", "1:30 min", "Butterworth, Notch multiarmónico y segmentación."],
        ["Slide 3 (I2)", "Bloque SOTA: 4 Features Físicas", "1:30 min", "Lógica del TI, SPE, Curtosis y f50/f90."],
        ["Slide 4 (I2)", "Deep Learning en Espectrogramas", "1:30 min", "CNN-1D / BiLSTM y riesgo de overfitting en LOSO."],
        ["Slide 5 (I3)", "Características Clásicas (137)", "1:30 min", "Riqueza acústica (temporal, espectral, wavelets)."],
        ["Slide 6 (I3)", "Invarianza a la Identidad de Paciente", "1:30 min", "Eliminación de MFCC absolutos contra fugas de timbre."],
        ["Slide 7 (I4)", "Fusión Híbrida y Bucle LOSO", "1:30 min", "Set de 141 features y SelectKBest dinámico por fold."],
        ["Slide 8 (I4)", "Benchmark: Resultados de Segmento", "1:30 min", "Comparativa de 9 modelos segmentarios y ganadores."],
        ["Slide 9 (I5)", "Delta CAS y Ley de Grandes Números", "1:30 min", "Boxplots + Scatter Plots y cancelación estadística."],
        ["Slide 10 (I5)", "Conclusiones y Limitaciones", "1:30 min", "Lecciones metodológicas y aprendizaje semi-supervisado."]
    ]
    
    t = Table(time_data, colWidths=[70, 160, 55, 215])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#002B49')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F4F6F7')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDC3C7')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8.5),
    ]))
    story.append(t)
    
    story.append(PageBreak())
    
    # ---------------------------------------------------------
    # SECTION 2: SLIDE BY SLIDE DETAILS
    # ---------------------------------------------------------
    story.append(Paragraph("2. Guía Detallada de Diapositivas (10 diapositivas / 15 minutos)", h1_style))
    
    slides = [
        {
            "num": "Slide 1 (Integrante 1): Portada y Objetivo Clínico (1:30 min)",
            "key": "Definir directamente el objetivo del proyecto y la lógica de BDR mediante el biomarcador Delta CAS.",
            "points": [
                "Presentar al equipo e ir directo al grano: evaluar de forma pasiva y acústica la Respuesta Broncodilatadora (BDR).",
                "Definir el biomarcador digital: Delta CAS = Tasa CAS en estado basal (Pre) - Tasa CAS post-fármaco (Post).",
                "Explicar que un biomarcador de éxito debe registrar deltas significativamente positivos para respondedores (BDR+) y cercanos a cero para no-respondedores (BDR-) y controles."
            ],
            "figures": "Fórmula matemática y esquema del biomarcador Delta CAS.",
            "tips": "No te detengas en rodeos de introducción teórica general. El tribunal quiere ver el rigor técnico y tu biomarcador desde la primera diapositiva."
        },
        {
            "num": "Slide 2 (Integrante 1): Preprocesamiento de Señales y Segmentación (1:30 min)",
            "key": "Explicar de forma exprés la cadena de filtrado para limpiar ruidos antes del modelado.",
            "points": [
                "Remuestreo a 4000 Hz para limitar la frecuencia de Nyquist a 2000 Hz, cubriendo sibilancias y eliminando ruidos altos irrelevantes.",
                "Filtro Paso Banda Butterworth de orden 8 (70-1900 Hz) con fase cero ('sosfiltfilt') para no distorsionar las señales en el tiempo.",
                "Filtro Comb Notch multiarmónico en 50 Hz y armónicos (ancho de banda de 1 Hz) para eliminar el zumbido de la red eléctrica.",
                "Segmentación: obtención de 14 900 segmentos de inspiración/espiración a partir de las anotaciones clínicas."
            ],
            "figures": "Comparativa temporal antes/después de filtrado y densidad de potencia (PSD) Welch.",
            "tips": "Como este paso es común a todos los grupos, explícalo rápido. Céntrate en justificar las decisiones técnicas (fase cero, formato SOS)."
        },
        {
            "num": "Slide 3 (Integrante 2): Bloque SOTA - Los 4 Marcadores Físicos (1:30 min)",
            "key": "Justificación matemática de los descriptores físicos de sibilancias de la literatura médica.",
            "points": [
                "Índice Tonal (TI): Mide la predictibilidad de la fase y la amplitud (las sibilancias son tonos puros coherentes).",
                "Entropía de Picos Espectrales (SPE): Evalúa la concentración de potencia espectral en frecuencias armónicas específicas.",
                "Curtosis Espectral: Caracteriza el apuntamiento de la distribución espectral de la señal.",
                "Ratio f50/f90: Permite discriminar espectralmente entre roncus (graves) y sibilancias (agudas)."
            ],
            "figures": "Espectrograma mel con flechas indicando el dominio físico en el que actúa cada variable.",
            "tips": "Demuestra que entiendes la física que hay detrás de estas 4 variables clásicas de la literatura."
        },
        {
            "num": "Slide 4 (Integrante 2): Experimento de Deep Learning - CNN en Espectrogramas (1:30 min)",
            "key": "Mostrar la exploración realizada con modelos profundos y justificar científicamente por qué fallaron.",
            "points": [
                "Entrenamiento de redes profundas (CNN-1D y BiLSTM) alimentadas directamente con espectrogramas de Mel segmentados.",
                "Resultados segmentarios de DL limitados por sobreajuste (overfitting) sistemático.",
                "Conclusión metodológica: Las redes neuronales complejas requieren volúmenes de datos masivos. En validación cruzada LOSO con pocos pacientes, tienden a memorizar características del canal y el ruido del paciente, perdiendo generalización frente a modelos tradicionales regularizados."
            ],
            "figures": "Esquema de la arquitectura de la red CNN-1D / BiLSTM probada.",
            "tips": "Defender este fallo es un punto a favor: demuestra rigor metodológico y comprensión de las limitaciones de Deep Learning en datos clínicos pequeños."
        },
        {
            "num": "Slide 5 (Integrante 3): Ingeniería de Características Clásica (137 features) (1:30 min)",
            "key": "Explicar el origen del conjunto expandido de 137 variables clásicas para capturar la riqueza acústica.",
            "points": [
                "Ampliación del set: Fusión de las 4 variables físicas SOTA con 137 descriptores acústicos clásicos.",
                "Justificación clínica: Los sonidos adventicios tienen patrones transitorios y dinámicas que requieren análisis multirresolución (Wavelets Daubechies 4), envolventes espectrales y medidas temporales de complejidad.",
                "Objetivo: Aumentar la riqueza de features para detectar sibilancias sutiles que escapan a los descriptores físicos simples."
            ],
            "figures": "Diagrama categórico de la distribución del set de 137 características.",
            "tips": "Explica que este set rico permite al clasificador ir mucho más allá del análisis tonal simple de sibilancias."
        },
        {
            "num": "Slide 6 (Integrante 3): Modelos Clásicos y Resultados de Segmento (1:30 min)",
            "key": "Presentar los resultados del pipeline clásico y cómo roza la significación clínica.",
            "points": [
                "Resultados segmentarios: Los 9 modelos tradicionales mejoran sustancialmente con el conjunto expandido. Random Forest lidera con un AUC de segmento de ~0.68.",
                "Efecto en Delta CAS: Al acumular las tasas por paciente en el conjunto de 137 features, el biomarcador Delta CAS roza el umbral de significación clínica.",
                "Prueba estadística de Kruskal-Wallis: El p-valor desciende a **0.0504** al comparar BDR+, BDR- y Controles Sanos, demostrando que añadir riqueza espectral y temporal fue el camino correcto."
            ],
            "figures": "clasico_presentation_metrics.png y clasico_presentation_delta_cas.png (p = 0.0504).",
            "tips": "Muestra que el modelo clásico mejoró drásticamente a SOTA, justificando por qué decidimos dar el paso final de fusionar ambos en el bloque híbrido."
        },
        {
            "num": "Slide 7 (Integrante 4): Pipeline Híbrido Optimizado y Selección en LOSO (1:30 min)",
            "key": "Fusión total de variables (141 features) y estrategia rigurosa de selección para evitar fugas de información.",
            "points": [
                "Fusión de las 137 clásicas + 4 marcadores SOTA para crear un vector unificado de 141 características.",
                "Selección dinámica de variables: Implementación de SelectKBest (Mutual Information, k=40) de manera independiente dentro de cada fold del bucle LOSO.",
                "Esto asegura que la selección de variables nunca vea información del paciente que se está usando para el test (honestidad metodológica absoluta)."
            ],
            "figures": "Esquema metodológico del bucle LOSO con selección de características interna.",
            "tips": "Hacer la selección de variables dentro de cada fold de LOSO evita el sesgo optimista y garantiza la validez clínica externa."
        },
        {
            "num": "Slide 8 (Integrante 4): Resultados de Clasificación a Nivel de Segmento (1:30 min)",
            "key": "Presentar el benchmark de los 9 modelos sobre las variables segmentarias y justificar la elección del Híbrido.",
            "points": [
                "Comparativa de los 9 modelos (LR, SVM lineal, SVM RBF, RF, GBM, XGB, CNN, BiLSTM y Ensemble).",
                "Modelos ganadores: Random Forest y el Ensemble blando logran una especificidad segmentaria sobresaliente del 87.6%.",
                "Justificar que la alta especificidad es prioritaria en clínica para minimizar falsos positivos en diagnóstico respiratorio."
            ],
            "figures": "hibrido_presentation_metrics.png e hibrido_presentation_confusion_matrices.png.",
            "tips": "Aunque el AUC a nivel de segmento ronde el 0.60-0.68, explica que esto sirve como base óptima para el biomarcador clínico global del siguiente bloque."
        },
        {
            "num": "Slide 9 (Integrante 5): Biomarcador Clínico Delta CAS y Ley de los Grandes Números (1:30 min)",
            "key": "Defensa del biomarcador Delta CAS definitivo y la significación clínica obtenida con los tres grupos.",
            "points": [
                "Boxplot definitivo con Kruskal-Wallis: p-valor de **0.0035** (clínicamente muy significativo para BDR+, BDR- y Controles).",
                "Explicación de los Scatter Plots (Pre vs Post CAS): Los respondedores (BDR+) caen claramente por debajo de la diagonal (reducción de CAS), los no-respondedores (BDR-) caen por encima y los controles sanos se quedan estables cerca del origen.",
                "La reflexión del promediado: Aunque el clasificador segmentario individual falle bastante (AUC ~0.65), al promediar sobre cientos de ciclos respiratorios por paciente (~280 de media), los errores aleatorios se cancelan mutuamente (ley de los grandes números), logrando estimar la tasa global y su cambio temporal (Delta CAS) de forma altamente fiable y potente."
            ],
            "figures": "hibrido_presentation_delta_cas.png y hibrido_scatter_pre_post.png.",
            "tips": "Esta es la diapositiva estrella. Habla con calma y seguridad sobre cómo la ley de los grandes números y la acumulación de datos eliminan el ruido del modelo de base."
        },
        {
            "num": "Slide 10 (Integrante 5): Limitaciones del Modelo y Conclusiones (1:30 min)",
            "key": "Análisis autocrítico de las limitaciones y líneas de trabajo futuro.",
            "points": [
                "Limitaciones: Muestra pequeña (23 pacientes asmáticos útiles) que requiere validación en cohortes externas.",
                "Peligro de usar modelos complejos (Deep Learning) con pocos sujetos en validación LOSO.",
                "Inestabilidad matemática del Delta CAS ante valores basales pre-BD muy pequeños.",
                "Trabajo futuro: Explotar los más de 12 000 segmentos no etiquetados del dataset usando algoritmos de aprendizaje semi-supervisado (como Label Propagation)."
            ],
            "figures": "Esquema final del flujo clínico completo.",
            "tips": "El tribunal valora enormemente que seáis autocríticos con las limitaciones clínicas del estudio."
        }
    ]
    
    for s_idx, slide in enumerate(slides):
        story.append(Paragraph(slide["num"], h2_style))
        story.append(Paragraph(f"<b>Mensaje principal:</b> {slide['key']}", body_style))
        
        story.append(Paragraph("<b>Puntos clave del discurso:</b>", body_style))
        for bullet in slide["points"]:
            story.append(Paragraph(f"• {bullet}", bullet_style))
            
        story.append(Paragraph(f"<b>Figura a mostrar:</b> {slide['figures']}", body_style))
        story.append(Spacer(1, 2))
        story.append(Paragraph(f"<i>Consejo de defensa: {slide['tips']}</i>", tip_style))
        
        if s_idx < len(slides) - 1:
            story.append(Spacer(1, 10))
            
    doc.build(story)
    print("PDF Successfully generated!")

if __name__ == "__main__":
    create_presentation_guide()
