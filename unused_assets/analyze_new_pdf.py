import pypdf
import os

pdf_path = r"c:\Users\PRO\Desktop\Root Canal\Root-canal-AI-diagnostics\diagnostics-15-01009.pdf"

try:
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        exit(1)

    reader = pypdf.PdfReader(pdf_path)
    print(f"Total pages: {len(reader.pages)}")
    
    full_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text.strip():
            print(f"\n--- Page {i+1} [EMPTY/IMAGE] ---\n")
        else:
            print(f"\n--- Page {i+1} ---\n")
            print(text[:500] + "..." if len(text) > 500 else text)
            full_text += text

    print(f"\nExtracted {len(full_text)} characters.")

    print(f"Error: {e}")
