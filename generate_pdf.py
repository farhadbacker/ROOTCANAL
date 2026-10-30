import sys
import subprocess
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from fpdf import FPDF
except ImportError:
    print("Installing fpdf2...")
    install("fpdf2")
    from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 16)
        self.cell(0, 10, "Comprehensive Model Evaluation Metrics", align="C")
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

def create_pdf(input_md, output_pdf):
    pdf = PDF()
    pdf.add_page()
    
    try:
        with open(input_md, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                pdf.ln(5)
                continue
                
            if line.startswith("# "):
                # Skip title as it's in the header
                if "Comprehensive Model Evaluation Metrics" in line:
                    continue
                pdf.set_font("helvetica", "B", 16)
                pdf.multi_cell(0, 10, line.replace("# ", ""))
                pdf.ln(2)
            elif line.startswith("## "):
                pdf.set_font("helvetica", "B", 14)
                pdf.multi_cell(0, 10, line.replace("## ", ""))
                pdf.ln(2)
            elif line.startswith("### "):
                pdf.set_font("helvetica", "B", 12)
                pdf.multi_cell(0, 8, line.replace("### ", ""))
                pdf.ln(1)
            else:
                pdf.set_font("helvetica", "", 11)
                clean_line = line.replace("**", "").replace("* ", "- ")
                pdf.multi_cell(0, 6, clean_line)
                
        pdf.output(output_pdf)
        print(f"Successfully generated {output_pdf}")
    except Exception as e:
        print(f"Error generating PDF: {e}")

if __name__ == "__main__":
    create_pdf("Model_Evaluation_Metrics.md", "Model_Evaluation_Metrics.pdf")
