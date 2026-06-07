import sys
import os
from pathlib import Path
from fpdf import FPDF

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class MarkdownPDF(FPDF):
    def header(self):
        # Arial bold 8
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, 'Proyecto de Biomarcadores Digitales - Guía de Presentación', 0, 0, 'L')
        self.ln(10)

    def footer(self):
        # Posición a 1.5 cm del final
        self.set_y(-15)
        # Arial italic 8
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        # Número de página
        self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', 0, 0, 'C')

def clean_unsupported_chars(text: str) -> str:
    # Reemplazar caracteres unicode comunes no soportados por latin-1
    replacements = {
        "─": "-",
        "—": "-",
        "•": "*",
        "🎙️": "[exposición]",
        "🎙": "[exposición]",
        "🎓": "[académico]",
        "✓": "ok",
        "⚠": "aviso",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "Δ": "Delta",
        "∗": "*",
        "𝑝": "p",
        "𝑜": "o",
        "𝑠": "s",
        "𝑡": "t",
        "𝑒": "e",
        "𝜟": "Delta",
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    
    # Intentar codificar en latin-1 ignorando los caracteres no soportados
    try:
        text.encode("latin-1")
    except UnicodeEncodeError:
        # Si aún falla, codificar ignorando los caracteres no válidos
        text = text.encode("latin-1", errors="ignore").decode("latin-1")
        
    return text

def convert_md_to_pdf(md_path: Path, pdf_path: Path, doc_title: str):
    print(f"Convertiendo {md_path.name} -> {pdf_path.name}...")
    
    if not md_path.exists():
        print(f"Error: No existe el archivo {md_path}")
        return

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pdf = MarkdownPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Título del documento
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(0, 51, 102) # Azul oscuro
    # Usar nuevos métodos recomendados en FPDF2
    pdf.cell(0, 15, clean_unsupported_chars(doc_title), new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(5)

    in_blockquote = False

    for line in lines:
        line_str = line.strip()
        if not line_str:
            if not in_blockquote:
                pdf.ln(2)
            continue

        # Reset blockquote visual if line doesn't start with >
        if not line_str.startswith(">"):
            in_blockquote = False

        # Limpiar caracteres no soportados antes de procesar
        line_str = clean_unsupported_chars(line_str)

        # Header 1
        if line_str.startswith("# "):
            pdf.ln(6)
            pdf.set_font("helvetica", "B", 16)
            pdf.set_text_color(0, 51, 102)
            pdf.multi_cell(0, 10, line_str[2:])
            pdf.ln(2)
            
        # Header 2
        elif line_str.startswith("## "):
            pdf.ln(5)
            pdf.set_font("helvetica", "B", 13)
            pdf.set_text_color(51, 102, 153)
            pdf.multi_cell(0, 8, line_str[3:])
            pdf.ln(2)

        # Header 3
        elif line_str.startswith("### "):
            pdf.ln(4)
            pdf.set_font("helvetica", "B", 11)
            pdf.set_text_color(102, 102, 102)
            pdf.multi_cell(0, 6, line_str[4:])
            pdf.ln(1)

        # Blockquote (speech script)
        elif line_str.startswith(">"):
            if not in_blockquote:
                pdf.set_fill_color(240, 244, 248) # Gris azulado claro
                in_blockquote = True
            
            clean_text = line_str[1:].strip()
            # Remove MD bold stars
            clean_text = clean_text.replace("**", "")
            
            pdf.set_font("helvetica", "I", 9.5)
            pdf.set_text_color(50, 50, 50)
            
            # Print with light gray background fill
            pdf.multi_cell(0, 5.5, clean_text, border='L', fill=True)
            pdf.ln(1)

        # Bullet list
        elif line_str.startswith("* ") or line_str.startswith("- ") or line_str.startswith("• "):
            bullet_text = line_str[2:]
            bullet_text = bullet_text.replace("**", "")
            
            pdf.set_font("helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            
            # Draw bullet bullet point
            pdf.set_x(15)
            pdf.cell(5, 6, chr(149), border=0) # 149 is standard bullet point symbol in western latin font
            pdf.multi_cell(0, 6, bullet_text)
            pdf.ln(1)
            
        # Standard paragraph
        else:
            para_text = line_str.replace("**", "")
            pdf.set_font("helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 6, para_text)
            pdf.ln(2)

    pdf.output(str(pdf_path))
    print(f"PDF generado con éxito: {pdf_path.name}")

def main():
    convert_md_to_pdf(
        PROJECT_ROOT / "guia_presentacion.md",
        PROJECT_ROOT / "guia_presentacion.pdf",
        "Guia Visual de Presentacion"
    )
    convert_md_to_pdf(
        PROJECT_ROOT / "guion_exposicion.md",
        PROJECT_ROOT / "guion_exposicion.pdf",
        "Guion de Voz Detallado para Exposicion"
    )

if __name__ == "__main__":
    main()
