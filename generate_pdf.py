import os
import sys
from fpdf import FPDF

class PresentationPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(15, 15, 15)
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        if self.page_no() == 1:
            return
        # Header
        self.set_font("helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, "Guía de Exposición Colectiva (15 min - 5 Integrantes de 3 min) | Proyecto Digital Biomarkers", align="R")
        self.ln(6)
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.2)
        self.line(15, 22, 195, 22)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Página {self.page_no()}", align="C")

    def speaker_header(self, num, name, topic, time_range):
        self.set_font("helvetica", "B", 13)
        self.set_text_color(20, 80, 120)  # Dark Teal/Blue
        self.cell(0, 8, f"INTEGRANTE {num}: {name}", new_x="LMARGIN", new_y="NEXT")
        self.set_font("helvetica", "B", 10.5)
        self.set_text_color(40, 40, 40)
        self.cell(0, 5, f"Tema: {topic}  |  Tiempo: {time_range} (3 minutos exactos)", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def slide_title(self, title):
        self.set_font("helvetica", "B", 11)
        self.set_text_color(100, 110, 120)
        self.cell(0, 6, f">> Diapositiva: {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        
    def bullet_point(self, bold_text, normal_text):
        self.set_font("helvetica", "B", 8.5)
        self.set_text_color(40, 40, 40)
        self.write(4.5, "  *  " + bold_text + ": ")
        self.set_font("helvetica", "", 8.5)
        self.set_text_color(60, 60, 60)
        self.write(4.5, normal_text + "\n")
        
    def box_text(self, title, text_lines):
        self.set_fill_color(245, 248, 250)
        self.set_draw_color(20, 80, 120)
        self.set_line_width(0.3)
        
        # Calculate height
        h = 5 + len(text_lines) * 4.2
        self.cell(0, h, "", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        
        # Draw text inside box
        curr_y = self.get_y()
        self.set_y(curr_y - h + 1.5)
        self.set_x(18)
        self.set_font("helvetica", "B", 8.5)
        self.set_text_color(20, 80, 120)
        self.cell(0, 4, title, new_x="LMARGIN", new_y="NEXT")
        self.set_x(18)
        self.set_font("helvetica", "", 8)
        self.set_text_color(50, 50, 50)
        for line in text_lines:
            self.cell(0, 3.8, line, new_x="LMARGIN", new_y="NEXT")
            self.set_x(18)
        self.set_y(curr_y)
        self.ln(2)

def generate_presentation_guide():
    pdf = PresentationPDF()
    
    # -------------------------------------------------------------
    # PORTADA
    # -------------------------------------------------------------
    pdf.add_page()
    
    # Decorative Top Border
    pdf.set_fill_color(20, 80, 120)
    pdf.rect(0, 0, 210, 12, "F")
    
    pdf.ln(15)
    
    # Title
    pdf.set_font("helvetica", "B", 20)
    pdf.set_text_color(20, 80, 120)
    pdf.multi_cell(0, 8, "GUÍA DE PRESENTACIÓN COLECTIVA\n(15 MINUTOS - 5 INTEGRANTES DE 3 MINUTOS)", align="C")
    pdf.ln(3)
    
    # Subtitle
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 6, "Resumen del Pipeline Acústico de Detección de CAS y\nAnálisis Clínico de Respuesta Broncodilatadora (BDR)\n*Inicio Resumido + Mapa del Pipeline para Integrante 1*", align="C")
    
    pdf.ln(10)
    
    # Speaker Distribution Box
    pdf.set_fill_color(245, 245, 245)
    pdf.set_draw_color(220, 220, 220)
    pdf.rect(15, 65, 180, 95, "DF")
    
    pdf.set_y(68)
    pdf.set_font("helvetica", "B", 10.5)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 6, "DISTRIBUCIÓN DEL TIEMPO (3 MINUTOS EXACTOS POR PONENTE):", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    
    speakers = [
        ("Integrante 1", "Introducción, Dataset Común y Mapa del Pipeline", "Minuto 0:00 - 3:00 (3.0 min)"),
        ("Integrante 2", "Preprocesamiento de Señales y Filtrado", "Minuto 3:00 - 6:00 (3.0 min)"),
        ("Integrante 3", "Segmentación y Feature Engineering", "Minuto 6:00 - 9:00 (3.0 min)"),
        ("Integrante 4", "Clasificación y Validación LOSO", "Minuto 9:00 - 12:00 (3.0 min)"),
        ("Integrante 5", "Análisis Clínico de Biomarcadores y Conclusiones", "Minuto 12:00 - 15:00 (3.0 min)")
    ]
    
    for key, topic, val in speakers:
        pdf.set_x(20)
        pdf.set_font("helvetica", "B", 9)
        pdf.set_text_color(20, 80, 120)
        pdf.cell(30, 5, key + ": ")
        pdf.set_font("helvetica", "B", 8.5)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(95, 5, topic)
        pdf.set_font("helvetica", "", 8.5)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, val, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        
    pdf.ln(3)
    
    # Dataset Summary
    pdf.set_x(20)
    pdf.set_font("helvetica", "B", 8.5)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 5, "Dataset del Pipeline: 14 900 señales totales | 1 923 ciclos etiquetados", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(20)
    pdf.cell(0, 5, "Filtro de variables: SelectKBest (30 features) e invarianza de identidad vocal", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(25)
    
    # Footer Portada
    pdf.set_y(190)
    pdf.set_font("helvetica", "I", 9.5)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 5, "Máster en Bioingeniería / Biomarcadores Digitales", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "Guía de Soporte Colectivo (5 Ponentes de 3 Minutos)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "Fecha de entrega: 15 de junio de 2026", align="C", new_x="LMARGIN", new_y="NEXT")

    # -------------------------------------------------------------
    # PONENTES
    # -------------------------------------------------------------
    
    # -------------------------------------------------------------
    # INTEGRANTE 1 (3 MINUTOS)
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.speaker_header("1", "[Nombre Integrante 1]", "Introducción Rápida, Dataset y Mapa del Pipeline", "Minuto 0:00 - 3:00")
    
    pdf.slide_title("1. Introducción y Dataset Clínico Común")
    pdf.bullet_point("Objetivo clínico", "Detectar de forma pasiva y acústica los ruidos adventicios continuos (CAS: sibilancias/roncus) para medir la respuesta broncodilatadora (BDR) en asma.")
    pdf.bullet_point("Dataset Común de la clase", "Compartimos la base de datos estándar de 28 sujetos (23 asmáticos + 5 controles) grabados en 2 canales durante 6 maniobras (3 pre y 3 post-BD) a 12.5 kHz.")
    pdf.bullet_point("Ciclos etiquetados", "1 923 ciclos respiratorios etiquetados procedentes de 18 pacientes (590 CAS / 1 333 NO CAS).")
    pdf.ln(1.5)
    
    pdf.slide_title("2. Mapa General del Pipeline Diseñado")
    pdf.bullet_point("Paso 1-4: Ingeniería de Datos", "Lectura de señales crudas, preprocesado acústico en 3 filtros, segmentación por ciclos inspiración/espiración y ensamble del dataset (14 900 señales).")
    pdf.bullet_point("Paso 5: Extracción de Features", "Normalización MAD robusta y extracción de 137 características invariantes a identidad.")
    pdf.bullet_point("Paso 6-8: Clasificación y Clínica", "Validación LOSO con selección de variables, reentrenamiento e inferencia sobre 14 900 muestras, análisis estadístico de Delta CAS por grupo y comparación con Deep Learning.")
    pdf.ln(1.5)

    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(20, 80, 120)
    pdf.cell(0, 5, "FIGURA A MOSTRAR: outputs/figures/step4/fig1_segmentos_por_sujeto.png", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 4, "  (Ilustra la distribución y volumen de los datos extraídos para los 28 sujetos del dataset)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.5)
    
    pdf.box_text("Guión del Ponente (Integrante 1 - 3 MINUTOS):", [
        "\"Buenos días. Comenzaremos presentando el contexto clínico del proyecto. El diagnóstico del asma se basa en la espirometría,\"",
        "\"pero proponemos una alternativa acústica pasiva mediante la detección automática de sonidos adventicios continuos (CAS),\"",
        "\"es decir, sibilancias y roncus pulmonares. Como compartimos el mismo dataset común de 28 sujetos y 1 923 ciclos etiquetados\"",
        "\"que el resto de grupos, resumiremos esta introducción para enfocarnos en la estructura de nuestro pipeline.\"",
        "\"En la diapositiva podemos observar la distribución homogénea de segmentos de señal extraídos por participante y el mapa\"",
        "\"general de nuestro pipeline. Este consta de 8 pasos: desde la ingeniería de datos (lectura, preprocesado, segmentación y\"",
        "\"ensamble del dataset), pasando por la extracción de 137 features, hasta la fase de clasificación en validación LOSO, reentrenamiento\"",
        "\"e inferencia de las 14 900 señales para calcular el biomarcador Delta CAS. A continuación, mis compañeros detallarán cada paso.\""
    ])

    # -------------------------------------------------------------
    # INTEGRANTE 2
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.speaker_header("2", "[Nombre Integrante 2]", "Preprocesamiento de Señales y Filtrado", "Minuto 3:00 - 6:00")
    
    pdf.slide_title("3. Cadena de Preprocesado de Audio")
    pdf.bullet_point("Paso 1: Remuestreo a 4 000 Hz", "Se reduce el muestreo de 12 500 Hz a 4 000 Hz usando 'resample_poly' (razón entera 8/25). Esto limita la frecuencia de Nyquist a 2 000 Hz, eliminando frecuencias altas irrelevantes para sibilancias.")
    pdf.bullet_point("Paso 2: Filtro Paso Banda Butterworth", "Filtro de orden 8 entre 70 Hz y 1900 Hz, aplicado bidireccionalmente ('sosfiltfilt') en Secciones de Segundo Orden (SOS). La fase cero previene la distorsión temporal de la sibilancia y el formato SOS evita la inestabilidad numérica.")
    pdf.bullet_point("Paso 3: Filtro Comb Notch", "Filtro notch IIR multiarmónico en 50 Hz y todos sus armónicos (100, 150... 1950 Hz) con ancho de banda fijo de 1 Hz. Elimina de forma ultra-selectiva el zumbido de la corriente eléctrica.")
    pdf.ln(2)
    
    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(20, 80, 120)
    pdf.cell(0, 5, "FIGURAS A MOSTRAR (2):", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 8.5)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 4.5, "  1. outputs/figures/step2/fig1_senal_cruda_vs_preprocesada.png (Comparativa tiempo antes/después)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4.5, "  2. outputs/figures/step2/fig4_psd_antes_despues.png (Espectro de densidad de potencia Welch)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    pdf.box_text("Guión del Ponente (Integrante 2):", [
        "\"Mi sección detalla el preprocesamiento de las señales de audio, un paso crítico para eliminar ruidos externos y de red.\"",
        "\"El primer paso es el remuestreo de la señal a 4000 Hz. Posteriormente, aplicamos un filtro paso banda Butterworth de orden 8\"",
        "\"con frecuencias de corte de 70 y 1900 Hz. Este filtro se aplica en dos direcciones mediante 'sosfiltfilt' en formato SOS.\"",
        "\"Esto nos asegura que no hay alteración de la fase y evita errores de redondeo en filtros de orden alto. Por último,\"",
        "\"aplicamos un filtro comb notch en 50 Hz y todos sus armónicos con un ancho de banda constante de tan solo 1 Hz.\"",
        "\"Si observamos el espectro de potencia PSD de la diapositiva, se ve claramente cómo el filtro comb elimina los picos de la red\"",
        "\"eléctrica representados por las líneas rojas verticales, dejando el resto de la señal acústica intacta y limpia.\""
    ])

    # -------------------------------------------------------------
    # INTEGRANTE 3
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.speaker_header("3", "[Nombre Integrante 3]", "Segmentación y Feature Engineering", "Minuto 6:00 - 9:00")
    
    pdf.slide_title("4. Segmentación por Ciclos")
    pdf.bullet_point("Proceso", "Usando las marcas temporales en segundos, cortamos los tramos individuales de inspiración y espiración del audio preprocesado, obteniendo 14 900 señales listas para procesar.")
    pdf.bullet_point("Asociación de metadatos", "Se asocian a cada segmento sus metadatos correspondientes (sujeto, fase respiratoria, sesión pre/post-BD y canal superior/inferior).")
    pdf.ln(1)
    
    pdf.slide_title("5. Normalización y Extracción de Características (137 features)")
    pdf.bullet_point("Normalización MAD robusta", "Se normaliza la amplitud por segmento: z = (x - mediana) / (1.4826 * MAD). Esto elimina diferencias de ganancia de los micrófonos y volumen respiratorio de cada paciente.")
    pdf.bullet_point("Features calculadas", "- Temporales (16): RMS, crest factor, entropía, Higuchi Mobility y Complexity.")
    pdf.bullet_point(" ", "- Espectrales (13) y Wavelet db4 (15): Centroide, spread, rolloff y descomposición a 5 niveles.")
    pdf.bullet_point(" ", "- Ratios espectrales (9) y AM de Hilbert (4): Ratios de energía y envolvente de modulación.")
    pdf.bullet_point(" ", "- MFCC Dinámicos (80): Std de 20 coeficientes Mel, y medias y std de sus derivadas Delta y Delta-Delta.")
    pdf.bullet_point("Invarianza de Identidad", "Se eliminan los MFCC absolutos (valores medios) porque identifican el timbre anatómico del paciente. En su lugar se usan MFCC dinámicos, que miden la modulación temporal de la sibilancia y permiten generalizar en LOSO.")
    pdf.ln(1.5)
    
    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(20, 80, 120)
    pdf.cell(0, 5, "FIGURAS A MOSTRAR (2):", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 8.5)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 4.5, "  1. outputs/figures/step3/fig1_segmentacion_maniobra_BDRpos.png (Visualización de la segmentación)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4.5, "  2. outputs/figures/step5/fig3_feature_means_cas_vs_nocas.png (Medias normalizadas de features)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.5)
    
    pdf.box_text("Guión del Ponente (Integrante 3):", [
        "\"Yo explicaré la segmentación y extracción de características. Utilizando las marcas del profesor, extraemos las fases de\"",
        "\"inspiración y espiración de los 28 sujetos en ambos canales, construyendo el dataset con 14 900 segmentos.\"",
        "\"Antes de extraer características, aplicamos una normalización MAD robusta para anular diferencias en el volumen de respiración.\"",
        "\"Posteriormente, extraemos 137 características. Para garantizar la invarianza frente al paciente y que el modelo aprenda la\"",
        "\"enfermedad y no la identidad de la persona, eliminamos los MFCC absolutos y los reemplazamos por variables dinámicas (variación)\"",
        "\"junto con ratios de bandas y modulación AM. En la figura de la derecha podemos observar la media normalizada de las\"",
        "\"características, donde variables de Higuchi y wavelets muestran una separación clara entre CAS (en rojo) y NO CAS (en azul).\""
    ])

    # -------------------------------------------------------------
    # INTEGRANTE 4
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.speaker_header("4", "[Nombre Integrante 4]", "Clasificación y Validación LOSO", "Minuto 9:00 - 12:00")
    
    pdf.slide_title("6. Validación Leave-One-Subject-Out (LOSO)")
    pdf.bullet_point("El problema del agrupamiento", "El profesor solicitó explícitamente agrupar las señales por participante, ya que corresponden a 18 pacientes específicos.")
    pdf.bullet_point("Validación LOSO (Agrupada)", "Para evitar fuga de información inter-sujeto, usamos Leave-One-Subject-Out (LOSO) (18 folds). En cada fold se entrena con 17 pacientes y se evalúa en el paciente 18 (nunca visto). Es el estándar clínico de generalización real.")
    pdf.bullet_point("SelectKBest interno", "Se integra SelectKBest(mutual_info_classif, k=30) dentro de cada fold para evitar cualquier fuga de información.")
    pdf.ln(1.5)
    
    pdf.slide_title("7. Modelos y Rendimiento de Clasificación")
    pdf.bullet_point("Modelos evaluados", "SVM (kernel RBF), Random Forest, XGBoost y un Ensemble blando (Voting de los tres).")
    pdf.bullet_point("Resultados del Ensemble", "Accuracy: 0.67 ± 0.12 | Especificidad: 0.84 ± 0.11 | AUC: 0.663 ± 0.141.")
    pdf.bullet_point("Importancia de la Especificidad", "Un 84% de especificidad minimiza los falsos positivos de CAS en pacientes sanos.")
    pdf.bullet_point("Varianza inter-sujeto", "Se observa alta varianza en la exactitud por fold (desviación del AUC de 0.14), reflejando la heterogeneidad real del asma.")
    pdf.ln(1.5)
    
    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(20, 80, 120)
    pdf.cell(0, 5, "FIGURAS A MOSTRAR (2):", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 8.5)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 4.5, "  1. outputs/figures/step6_loso/fig1_roc_curves.png (Curvas ROC en validación LOSO)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4.5, "  2. outputs/figures/step6_loso/fig3_loso_auc_per_fold.png (Evolución de AUC por fold/paciente)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.5)
    
    pdf.box_text("Guión del Ponente (Integrante 4):", [
        "\"En esta sección analizamos el clasificador. Para entrenar y evaluar de manera honesta, implementamos una validación\"",
        "\"Leave-One-Subject-Out (LOSO) con 18 folds. Evaluamos modelos clásicos y un Ensemble Soft de SVM, Random Forest y XGBoost.\"",
        "\"El Ensemble logra un 67% de exactitud y un 84% de especificidad en segmentos, lo cual es muy valioso para evitar falsos\"",
        "\"positivos acústicos. En la curva ROC que vemos en la pantalla se comparan los modelos, situando al Ensemble y Random Forest\"",
        "\"como los mejores por su estabilidad de AUC. La segunda gráfica muestra el AUC por paciente individual: resalta la gran\"",
        "\"varianza inter-sujeto, donde pacientes como P1 se clasifican de forma excelente (AUC=0.90) y otros como P10 se quedan en el azar.\"",
        "\"Esto valida científicamente la variabilidad real de las sibilancias en pacientes asmáticos.\""
    ])

    # -------------------------------------------------------------
    # INTEGRANTE 5
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.speaker_header("5", "[Nombre Integrante 5]", "Análisis Clínico de Biomarcadores y Conclusiones", "Minuto 11:30 - 15:00")
    
    pdf.slide_title("8. Inferencia en el Dataset Completo (14 900 señales)")
    pdf.bullet_point("Tasa CAS", "Una vez validado el clasificador, se aplica a las 14 900 señales para calcular las tasas pre-BD y post-BD por sujeto.")
    pdf.bullet_point("Comparativa clínica", "- Grupo BDR+ (9 sujetos): Reducción de CAS tras el fármaco (18.89% pre-BD -> 15.54% post-BD).")
    pdf.bullet_point(" ", "- Grupo BDR- (14 sujetos): Incremento de la tasa de CAS (16.28% pre-BD -> 22.41% post-BD).")
    pdf.bullet_point(" ", "- Controles (5 sujetos): Tasas bajas y estables (17.04% pre-BD -> 16.28% post-BD).")
    pdf.bullet_point("Significancia Estadística", "Prueba de Mann-Whitney U para Delta CAS (BDR+ vs BDR-): **p-valor = 0.0298** ($p < 0.05$). La respuesta broncodilatadora medida acústicamente es significativamente diferente entre ambos grupos de pacientes.")
    pdf.ln(1)
    
    pdf.slide_title("9. Limitaciones y Trabajo Futuro")
    pdf.bullet_point("Limitaciones técnicas", "Entrenamiento del clasificador basado únicamente en el canal inferior (micrófono 1). El canal superior es una extrapolación.")
    pdf.bullet_point("Inestabilidad de Delta CAS", "La métrica porcentual sufre de inestabilidad con valores basales pre-BD pequeños (e.g., pasar de 1 a 3 CAS da delta de -200%).")
    pdf.bullet_point("Trabajo Futuro", "Aprendizaje semi-supervisado (Label Propagation) para pseudo-etiquetar y aprovechar las más de 12 000 señales sin etiqueta.")
    pdf.ln(1)
    
    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(20, 80, 120)
    pdf.cell(0, 5, "FIGURAS A MOSTRAR (2):", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 8.5)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 4.5, "  1. outputs/figures/step7/fig3_boxplot_delta_cas.png (Boxplot de Delta CAS y significancia p=0.0298)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4.5, "  2. outputs/figures/step7/fig4_heatmap_delta_cas.png (Heatmap de Delta CAS por sujeto y condición)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.5)
    
    pdf.box_text("Guión del Ponente (Integrante 5):", [
        "\"Para concluir, analizamos el biomarcador clínico Delta CAS sobre las 14 900 señales. El clasificador predice una reducción\"",
        "\"clara de CAS en el grupo respondedor BDR+, mientras que el grupo no respondedor BDR- exhibe un aumento de sibilancias.\"",
        "\"La prueba estadística de Mann-Whitney U para Delta CAS entre ambos grupos es significativa, con un p-valor de 0.0298.\"",
        "\"Esto valida clínicamente que las sibilancias estimadas por nuestro modelo se comportan como un biomarcador digital útil.\"",
        "\"Como limitaciones principales, el entrenamiento se basó en el canal inferior y la métrica Delta CAS es inestable ante valores\"",
        "\"basales muy pequeños. Proponemos el uso futuro de técnicas de aprendizaje semi-supervisado para explotar los miles de\"",
        "\"segmentos no etiquetados disponibles. Quedamos a su disposición para preguntas. Muchas gracias.\""
    ])
    
    # -------------------------------------------------------------
    # SAVE PDF
    # -------------------------------------------------------------
    output_path = os.path.join(_PROJECT_ROOT, "Guia_Presentacion_15min.pdf")
    pdf.output(output_path)
    print(f"PDF generado exitosamente en: {output_path}")

if __name__ == "__main__":
    from pathlib import Path
    # Detectar raíz del proyecto anclando en proy_labels.mat
    _HERE = Path(__file__).resolve().parent
    _PROJECT_ROOT = next(
        (p for p in [_HERE.parent, _HERE] if (p / "proy_labels.mat").exists()),
        _HERE.parent,
    )
    
    generate_presentation_guide()
