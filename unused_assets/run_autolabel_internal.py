#!/usr/bin/env python
import sys
import os

base_dir = r"c:\Users\PRO\Desktop\root canal\Root-canal-AI-diagnostics"
sys.path.insert(0, base_dir)
os.chdir(base_dir)

from autolabel_dataset import EndodonticClassifier

if __name__ == "__main__":
    print("=" * 70)
    print("Endodontic X-ray Auto-Labeling with CLIP")
    print("=" * 70)
    print()
    
    images_dir = os.path.join(base_dir, "dataset", "images")
    
    print(f"Base directory: {base_dir}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Images directory: {images_dir}")
    print(f"Images directory exists: {os.path.exists(images_dir)}")
    print()
    
    if not os.path.exists(images_dir):
        print(f"✗ Dataset directory not found: {images_dir}")
        sys.exit(1)
    
    try:
        classifier = EndodonticClassifier()
        results = classifier.autolabel_dataset(images_dir)
        
        if results:
            print("\n" + "=" * 70)
            print("Auto-labeling completed successfully!")
            print("=" * 70)
        else:
            print("No results generated")
    except Exception as e:
        print(f"✗ Error during auto-labeling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
