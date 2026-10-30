import os
import sys

print("Starting test...")
print("Python version:", sys.version)
print("Current directory:", os.getcwd())

try:
    import torch
    print("✓ PyTorch imported")
except Exception as e:
    print("✗ PyTorch import failed:", e)

try:
    import clip
    print("✓ CLIP imported")
except Exception as e:
    print("✗ CLIP import failed:", e)

try:
    from PIL import Image
    print("✓ PIL imported")
except Exception as e:
    print("✗ PIL import failed:", e)

try:
    from tqdm import tqdm
    print("✓ tqdm imported")
except Exception as e:
    print("✗ tqdm import failed:", e)

print("\nAll imports successful!")
print("\nNow testing autolabel_dataset...")

try:
    from autolabel_dataset import EndodonticClassifier
    print("✓ EndodonticClassifier imported successfully")
    
    classifier = EndodonticClassifier()
    print("✓ EndodonticClassifier initialized")
    
except Exception as e:
    print("✗ Error:", e)
    import traceback
    traceback.print_exc()
