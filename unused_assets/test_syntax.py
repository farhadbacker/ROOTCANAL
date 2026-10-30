import sys
import os

print("Checking Python syntax for all new files...")
print("=" * 60)

files_to_check = [
    "contourlet_filter.py",
    "preprocess_dataset.py", 
    "train_yolo.py",
    "inference_server.py"
]

for filename in files_to_check:
    try:
        with open(filename, 'r') as f:
            compile(f.read(), filename, 'exec')
        print(f"✓ {filename}: Syntax OK")
    except SyntaxError as e:
        print(f"✗ {filename}: Syntax Error at line {e.lineno}: {e.msg}")
    except Exception as e:
        print(f"✗ {filename}: Error - {e}")

print("=" * 60)
print("Syntax check complete!")
