import sys
import pypdf

try:
    reader = pypdf.PdfReader(r"c:\Users\PRO\Desktop\Root Canal\Root-canal-AI-diagnostics\Fusion_of_Image_Filtering_and_Knowledge-Distilled_YOLO_Models_for_Root_Canal_Failure_Diagnosis.pdf")
    print(f"Total pages: {len(reader.pages)}")
    full_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        full_text += text
        print(f"--- Page {i+1} ---")
        print(text)
    
    print("\n\n--- ANALYSIS ---")
    lower_text = full_text.lower()
    if "best filter" in lower_text:
        print("Found 'best filter'")
    if "highest accuracy" in lower_text:
        print("Found 'highest accuracy'")
    
except Exception as e:
    print(f"Error: {e}")
