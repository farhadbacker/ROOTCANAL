import os
import sys
import time

base_dir = os.path.dirname(os.path.abspath(__file__))

print(f"Script location: {__file__}")
print(f"Base directory: {base_dir}")
print(f"Time: {time.asctime()}")

os.chdir(base_dir)
sys.path.insert(0, base_dir)

from autolabel_dataset import EndodonticClassifier

print("=" * 70)
print("Endodontic X-ray Auto-Labeling with CLIP")
print("=" * 70)
print()

try:
    classifier = EndodonticClassifier()
    results = classifier.autolabel_dataset("dataset/images")
    
    if results:
        print("\n" + "=" * 70)
        print("Auto-labeling completed successfully!")
        print("=" * 70)
    else:
        print("No results generated")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Script finished at:", time.asctime())
