import sys
import pypdf

pdf_path = r"c:\Users\PRO\Desktop\Root Canal\Root-canal-AI-diagnostics\unused_assets\diagnostics-15-01009.pdf"

try:
    reader = pypdf.PdfReader(pdf_path)
    print(f"Total pages: {len(reader.pages)}")
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        print(f"--- Page {i+1} ---")
        print(text)
except Exception as e:
    print(f"Error: {e}")
