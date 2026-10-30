#!/usr/bin/env python3
import os
from pathlib import Path
import cv2
from tqdm import tqdm
from contourlet_filter import ContourletTransform

print("=" * 70)
print("Preprocessing Roboflow Dataset with Contourlet Filter")
print("=" * 70)

base_dir = Path(__file__).parent.absolute()
os.chdir(base_dir)

roboflow_dir = base_dir / "Root Canal.v1i.yolov8"
splits = ["train", "valid", "test"]

contourlet = ContourletTransform(num_levels=2, num_directions=8)
print("\n✓ Contourlet filter initialized")

for split in splits:
    images_dir = roboflow_dir / split / "images"
    
    if not images_dir.exists():
        print(f"⚠️  {split}/images not found, skipping")
        continue
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"\nProcessing {split} split ({len(image_files)} images)...")
    
    for img_path in tqdm(image_files, desc=f"Filtering {split}"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            filtered = contourlet.apply(img_rgb)
            
            filtered_bgr = cv2.cvtColor(filtered.astype('uint8'), cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(str(img_path), filtered_bgr)
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

print("\n" + "=" * 70)
print("✓ Roboflow dataset preprocessing completed!")
print("=" * 70)
