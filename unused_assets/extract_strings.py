import re
import sys
from pathlib import Path

pdf_path = r"c:\Users\PRO\Desktop\Root Canal\Root-canal-AI-diagnostics\diagnostics-15-01009.pdf"

try:
    with open(pdf_path, 'rb') as f:
        data = f.read()
        
    # Find sequence of 4 or more printable characters
    strings = re.findall(b"[a-zA-Z0-9\s\.\,\-\:\(\)]{10,}", data)
    
    print(f"Found {len(strings)} strings. Dumping significant ones...\n")
    
    for s in strings:
        try:
            decoded = s.decode('utf-8')
            # Filter out some noise
            if len(decoded) > 20 and " " in decoded: 
                print(decoded)
        except:
            pass
            
except Exception as e:
    print(f"Error: {e}")
