import os
import sys
from fpdf import FPDF

class ScriptsPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(20, 20, 20)
        self.set_auto_page_break(auto=True, margin=20)
        
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, "Guiones de Exposición de 15 Minutos (3 min por ponente) | Proyecto Digital Biomarkers", align="R")
        self.ln(6)
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.2)
        self.line(20, 26, 190, 26)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Página {self.page_no()}", align="C")

    def speaker_title(self, num, name, topic, time_range, words):
        self.set_font("helvetica", "B", 14)
        self.set_text_color(20, 80, 120)  # Dark Teal/Blue
        self.cell(0, 8, f"INTEGRANTE {num}: {name}", new_x="LMARGIN", new_y="NEXT")
        self.set_font("helvetica", "B", 10.5)
        self.set_text_color(40, 40, 40)
        self.cell(0, 5, f"Tema: {topic}", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 5, f"Tiempo: {time_range} (3 minutos exactos)  |  Longitud: ~{words} palabras", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        
    def script_paragraph(self, text):
        self.set_font("helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        parts = text.split("[")
        for i, part in enumerate(parts):
            if i == 0:
                self.write(5.5, part)
            else:
                subparts = part.split("]")
                cue = subparts[0]
                rest = subparts[1] if len(subparts) > 1 else ""
                
                self.set_font("helvetica", "B", 9.5)
                self.set_text_color(180, 40, 40)
                self.write(5.5, f" [{cue}] ")
                
                self.set_font("helvetica", "", 10)
                self.set_text_color(50, 50, 50)
                self.write(5.5, rest)
        self.ln(4.5)

    def figures_to_prepare(self, figs):
        self.ln(2)
        self.set_fill_color(245, 248, 250)
        self.set_draw_color(20, 80, 120)
        self.set_line_width(0.3)
        
        h = 6 + len(figs) * 4.5
        self.cell(0, h, "", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        
        curr_y = self.get_y()
        self.set_y(curr_y - h + 2)
        self.set_x(23)
        self.set_font("helvetica", "B", 9)
        self.set_text_color(20, 80, 120)
        self.cell(0, 4.5, "RECURSOS VISUALES Y FIGURAS A MOSTRAR EN ESTE BLOQUE:", new_x="LMARGIN", new_y="NEXT")
        self.set_x(23)
        self.set_font("helvetica", "", 8.5)
        self.set_text_color(60, 60, 60)
        for fig in figs:
            self.cell(0, 4, fig, new_x="LMARGIN", new_y="NEXT")
            self.set_x(23)
        self.set_y(curr_y)
        self.ln(2)

def generate_scripts_pdf():
    pdf = ScriptsPDF()
    
    # -------------------------------------------------------------
    # PORTADA
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.set_fill_color(20, 80, 120)
    pdf.rect(0, 0, 210, 12, "F")
    
    pdf.ln(15)
    
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(20, 80, 120)
    pdf.multi_cell(0, 8, "GUIONES ESCRITOS DE EXPOSICIÓN GRUPAL\nENFOQUE EN BIOMARCADORES, MODELOS Y DELTA CAS", align="C")
    pdf.ln(3)
    
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 6, "Guía de Soporte de Lectura y Ensayo para la Defensa de 15 Minutos\n*Discursos reestructurados con enfoque directo en la significación clínica y limitaciones*", align="C")
    
    pdf.ln(12)
    
    pdf.set_fill_color(245, 245, 245)
    pdf.set_draw_color(220, 220, 220)
    pdf.rect(15, 65, 180, 95, "DF")
    
    pdf.set_y(68)
    pdf.set_font("helvetica", "B", 10.5)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 6, "ASIGNACIÓN Y CRONOGRAMA DE TIEMPOS:", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    
    speakers = [
        ("Integrante 1", "Portada, Objetivo del Proyecto y el Biomarcador Delta CAS", "Minuto 0:00 - 3:00 (3.0 min)"),
        ("Integrante 2", "Dataset Común, Estrategia de Validación LOSO y Bloque SOTA", "Minuto 3:00 - 6:00 (3.0 min)"),
        ("Integrante 3", "Bloque Clásico (137 Variables) y Fuga de Timbre del Paciente", "Minuto 6:00 - 9:00 (3.0 min)"),
        ("Integrante 4", "Bloque Híbrido (141 Features) y Selección en el Bucle LOSO", "Minuto 9:00 - 12:00 (3.0 min)"),
        ("Integrante 5", "Resultados Diagnósticos, Limitaciones Metodológicas y Conclusión", "Minuto 12:00 - 15:00 (3.0 min)")
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
    
    pdf.set_x(20)
    pdf.set_font("helvetica", "B", 8.5)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 5, "Instrucciones de Ensayo: Cada guión tiene ~380-400 palabras.", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(20)
    pdf.cell(0, 5, "Esto equivale a hablar a un ritmo pausado y claro de 130 palabras/minuto.", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(25)
    
    pdf.set_y(190)
    pdf.set_font("helvetica", "I", 9.5)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 5, "Máster en Bioingeniería / Biomarcadores Digitales", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "Guiones de Exposición Adaptados (15 Minutos - 5 Ponentes)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "Fecha de entrega: 15 de junio de 2026", align="C", new_x="LMARGIN", new_y="NEXT")

    # -------------------------------------------------------------
    # INTEGRANTE 1 (Diapositivas 1 y 2)
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.speaker_title("1", "[Nombre Integrante 1]", "Portada, Objetivo Clínico y Preprocesamiento de Señales", "Minuto 0:00 - 3:00", 350)
    pdf.figures_to_prepare([
        "Diapositiva 1: Portada y esquema conceptual de la Respuesta Broncodilatadora (BDR).",
        "Diapositiva 2: Gráficas de señal cruda vs. preprocesada y densidad de potencia (PSD) Welch."
    ])
    pdf.ln(3)
    
    pdf.script_paragraph(
        "\"Buenos días a todos. [Inicio - Diapositiva 1] En esta exposición grupal os presentaremos el desarrollo de un pipeline "
        "acústico automatizado en Python diseñado para procesar sonidos respiratorios patológicos continuos (CAS), "
        "como sibilancias y roncus, con el fin de evaluar la Respuesta Broncodilatadora (BDR) en pacientes asmáticos. "
        "Nuestro biomarcador digital principal es **Delta CAS**, definido como la Tasa CAS Pre-BD menos la Tasa CAS Post-BD. "
        "Un modelo exitoso debe registrar deltas significativamente positivos para respondedores (BDR+) y cercanos a cero para no-respondedores (BDR-).\""
    )
    pdf.script_paragraph(
        "\"[Diapositiva 2] El preprocesamiento de la señal de audio es crítico. Primero remuestreamos la señal a 4000 Hz, lo que limita la frecuencia "
        "de Nyquist a 2000 Hz, cubriendo el rango de las sibilancias y eliminando frecuencias altas irrelevantes. Luego aplicamos un filtro paso "
        "banda Butterworth de orden 8 (70-1900 Hz) con fase cero mediante 'sosfiltfilt' para evitar distorsiones temporales. Finalmente, "
        "un filtro Comb Notch multiarmónico en 50 Hz con ancho de banda de 1 Hz elimina el zumbido de la corriente eléctrica. De la señal limpia, "
        "obtenemos 14 900 segmentos de inspiración/espiración listos para el modelado. [Paso a Integrante 2]\""
    )

    # -------------------------------------------------------------
    # INTEGRANTE 2 (Diapositivas 3 y 4)
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.speaker_title("2", "[Nombre Integrante 2]", "Bloque SOTA (4 Features Físicas) y Deep Learning (CNN)", "Minuto 3:00 - 6:00", 370)
    pdf.figures_to_prepare([
        "Diapositiva 3: Espectrograma de Mel y fórmulas de las 4 variables físicas SOTA.",
        "Diapositiva 4: Esquema de la red profunda CNN-1D + BiLSTM evaluada."
    ])
    pdf.ln(3)
    
    pdf.script_paragraph(
        "\"Muchas gracias. [Inicio - Diapositiva 3] En el primer bloque técnico evaluamos un enfoque minimalista basado en 4 características "
        "SOTA físicas descritas en la literatura: el Índice Tonal (TI), que mide la predictibilidad de la fase y la amplitud; la Entropía "
        "de Picos Espectrales (SPE), que caracteriza la concentración de la potencia espectral; la Curtosis Espectral, que describe el apuntamiento "
        "frecuencial; y la ratio f50/f90, que separa sibilancias agudas de roncus graves. Al entrenar 9 modelos tradicionales y redes con estas "
        "4 variables, los resultados fueron limitados (AUC de segmento ~0.58). El ruido en cada trozo distorsionó las tasas acumuladas del paciente, "
        "perdiendo significación clínica en Delta CAS.\""
    )
    pdf.script_paragraph(
        "\"[Diapositiva 4] Para superar esto, también experimentamos con Deep Learning, entrenando una red CNN-1D combinada con capas BiLSTM alimentada "
        "directamente con imágenes de espectrogramas Mel de cada segmento. Sin embargo, el modelo sufrió un severo sobreajuste (overfitting) en el "
        "bucle de validación Leave-One-Subject-Out (LOSO). Con un dataset clínico de 28 sujetos, las redes profundas tienden a memorizar el timbre y "
        "ruido particular de cada paciente en lugar del patrón respiratorio de la patología. Esto demostró que necesitábamos un enfoque clásico "
        "enriquecido y robusto. [Paso a Integrante 3]\""
    )

    # -------------------------------------------------------------
    # INTEGRANTE 3 (Diapositivas 5 y 6)
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.speaker_title("3", "[Nombre Integrante 3]", "Características Clásicas y Resultados de Modelado Clásico", "Minuto 6:00 - 9:00", 360)
    pdf.figures_to_prepare([
        "Diapositiva 5: Tabla categórica de las 137 variables acústicas clásicas.",
        "Diapositiva 6: clasico_presentation_metrics.png y clasico_presentation_delta_cas.png (p = 0.0504)."
    ])
    pdf.ln(3)
    
    pdf.script_paragraph(
        "\"Muchas gracias. [Inicio - Diapositiva 5] A partir del bloque anterior, decidimos enriquecer el modelo sumando a las 4 variables físicas SOTA un bloque de 137 descriptores acústicos clásicos. "
        "La justificación clínica es que los sonidos respiratorios son dinámicos y complejos. Al combinar variables de forma de onda temporal con descomposición Wavelet "
        "(para capturar transitorios de sonido rápidos) y ratios espectrales, permitimos al modelo analizar la textura del audio completa y detectar sibilancias sutiles que las variables físicas solas no veían.\""
    )
    pdf.script_paragraph(
        "\"[Diapositiva 6] Al evaluar los 9 clasificadores en este conjunto expandido de 137 features, el rendimiento en segmento mejoró sustancialmente, "
        "con Random Forest y el Ensemble logrando un AUC de ~0.68. Al acumular estas predicciones por sujeto, el biomarcador Delta CAS comenzó a tomar sentido "
        "clínico: al comparar los tres grupos (BDR+, BDR- y Controles), la prueba de Kruskal-Wallis arrojó un p-valor de **0.0504**. Esto roza la "
        "significación clínica convencional ($0.05$), demostrando la validez del modelado clásico y justificando el paso final de fusionar ambos en el bloque híbrido. [Paso a Integrante 4]\""
    )

    # -------------------------------------------------------------
    # INTEGRANTE 4 (Diapositivas 7 y 8)
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.speaker_title("4", "[Nombre Integrante 4]", "Fusión Híbrida, Bucle LOSO y Resultados de Clasificación", "Minuto 9:00 - 12:00", 365)
    pdf.figures_to_prepare([
        "Diapositiva 7: Diagrama de flujo del bucle de validación LOSO con SelectKBest interno.",
        "Diapositiva 8: Gráfica de métricas comparativas segmentarias (hibrido_presentation_metrics.png)."
    ])
    pdf.ln(3)
    
    pdf.script_paragraph(
        "\"Muchas gracias. [Inicio - Diapositiva 7] En el Bloque 3 decidimos crear el pipeline Híbrido, fusionando la riqueza de las 137 features "
        "clásicas con el rigor físico de las 4 variables SOTA, totalizando 141 características. Para controlar la dimensionalidad y evitar sesgos, "
        "aplicamos una estrategia de validación Leave-One-Subject-Out (LOSO) con selección dinámica de variables. En cada uno de los folds, "
        "calculamos SelectKBest (Mutual Information) seleccionando las mejores k=40 variables sobre el conjunto de entrenamiento de ese fold. "
        "Al recalcular la selección en cada pliegue de manera aislada, garantizamos que el modelo nunca vea información del paciente excluido.\""
    )
    pdf.script_paragraph(
        "\"[Diapositiva 8] Entrenamos los 9 modelos sobre este set híbrido. El modelo Random Forest y el Ensemble blando resultaron ser los más "
        "estables a nivel de segmento, con una exactitud de 68.2% y, sobre todo, una especificidad sobresaliente del 87.6%. Esta alta especificidad "
        "es clave en el entorno médico, ya que previene falsos positivos diagnósticos de sibilancia en pacientes sanos. Aunque a nivel de segmento "
        "las métricas parezcan modestas, esta clasificación robusta sirve como la base perfecta para el diagnóstico a nivel de paciente que "
        "veremos a continuación. [Paso a Integrante 5]\""
    )

    # -------------------------------------------------------------
    # INTEGRANTE 5 (Diapositivas 9 y 10)
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.speaker_title("5", "[Nombre Integrante 5]", "Biomarcador Delta CAS, Ley de Grandes Números y Conclusión", "Minuto 12:00 - 15:00", 410)
    pdf.figures_to_prepare([
        "Diapositiva 9: Boxplot híbrido de Delta CAS (p = 0.0035) y Scatter Plot Pre vs. Post CAS (hibrido_scatter_pre_post.png).",
        "Diapositiva 10: Tabla resumen de limitaciones técnicas y líneas de desarrollo futuro."
    ])
    pdf.ln(3)
    
    pdf.script_paragraph(
        "\"Muchas gracias. [Inicio - Diapositiva 9] Al realizar la inferencia y acumular las tasas de sibilancias por paciente, el biomarcador "
        "Delta CAS del pipeline híbrido logró por primera vez una **significación clínica e importancia estadística contundentes**, "
        "con un p-valor de **0.0035** en la prueba de Kruskal-Wallis. En el scatter plot Pre vs. Post se observa claramente cómo los sujetos respondedores "
        "(BDR+) caen por debajo de la diagonal indicando una clara reducción de sibilancias tras el fármaco, los BDR- caen por encima y los controles "
        "se quedan en tasas muy bajas y estables cerca del origen.\""
    )
    pdf.script_paragraph(
        "\"Quiero recalcar la reflexión matemática clave: aunque el clasificador segmentario individual sea modesto (AUC de segmento de ~0.60-0.68), "
        "el resultado del biomarcador Delta CAS a nivel de paciente es sumamente robusto. Esto se debe a que, al acumular las predicciones sobre "
        "cientos de ciclos respiratorios por sujeto (unos 280 de media), los errores aleatorios individuales de clasificación (falsos positivos "
        "y negativos) se cancelan mutuamente por la ley de los grandes números. Esto nos permite promediar y estimar la tendencia global del paciente "
        "con una precisión excelente, logrando predecir el diagnóstico clínico final con un **AUC de BDR de 0.825** y exactitud diagnóstica del **73.9%**.\""
    )
    pdf.script_paragraph(
        "\"[Diapositiva 10] Como investigadores es fundamental reconocer las limitaciones metodológicas de nuestro modelo: el tamaño de la muestra "
        "sigue siendo de 23 pacientes asmáticos útiles, la métrica porcentual puede ser inestable ante tasas basales pequeñas, y hay riesgo de sobreajuste "
        "de Deep Learning en muestras pequeñas. Como conclusión, validamos acústicamente un biomarcador digital robusto, pasivo y reproducible para "
        "asma, y proponemos para el trabajo futuro el uso de algoritmos semi-supervisados (como Label Propagation) para aprovechar los más de "
        "12 000 segmentos no anotados disponibles en el dataset. Quedamos a su disposición para preguntas. Muchas gracias.\""
    )
    
    # -------------------------------------------------------------
    # SAVE PDF
    # -------------------------------------------------------------
    output_path = os.path.join(_PROJECT_ROOT, "Guiones_Exposicion_5_Ponentes.pdf")
    pdf.output(output_path)
    print(f"PDF generado exitosamente en: {output_path}")

if __name__ == "__main__":
    from pathlib import Path
    _HERE = Path(__file__).resolve().parent
    _PROJECT_ROOT = next(
        (p for p in [_HERE.parent, _HERE] if (p / "proy_labels.mat").exists()),
        _HERE.parent,
    )
    
    generate_scripts_pdf()
