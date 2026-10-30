#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import torch
from contourlet_filter import ContourletTransform

os.chdir(Path(__file__).parent.absolute())

print("=" * 70)
print("FULL PIPELINE: Preprocess + Train with Contourlet Filter")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}\n")

roboflow_dir = Path("Root Canal.v1i.yolov8")

print("=" * 70)
print("STEP 1: Preprocessing Roboflow Dataset")
print("=" * 70)

contourlet = ContourletTransform(num_levels=2, num_directions=8)
print("\n✓ Contourlet filter initialized")

splits = ["train", "valid", "test"]
total_processed = 0

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
            total_processed += 1
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

print(f"\n✓ Processed {total_processed} images")

print("\n" + "=" * 70)
print("STEP 2: Creating YAML Config")
print("=" * 70)

yaml_content = """path: .
train: "Root Canal.v1i.yolov8/train/images"
val: "Root Canal.v1i.yolov8/valid/images"
test: "Root Canal.v1i.yolov8/test/images"

nc: 4
names: ['No Endodontic Treatment', 'Complete Endodontic Treatment', 'Incomplete Endodontic Treatment', 'Total Endodontic Failure']
"""

with open("data_roboflow_filtered.yaml", "w") as f:
    f.write(yaml_content)

print("✓ YAML config created\n")

print("=" * 70)
print("STEP 3: Training Model with Filtered Dataset")
print("=" * 70 + "\n")

try:
    model = YOLO("yolov8n.pt")
    print("✓ Loaded YOLOv8n model\n")
    
    print("Starting training...")
    print("Estimated time: 2-3 hours\n")
    
    results = model.train(
        data="data_roboflow_filtered.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        patience=20,
        save=True,
        verbose=True
    )
    
    print("\n✓ Training completed successfully")
    
    model.save("dental_yolo_roboflow_filtered.pt")
    print(f"✓ Model saved: dental_yolo_roboflow_filtered.pt")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Update inference_server.py to use: dental_yolo_roboflow_filtered.pt")
    print("2. Enable contourlet filter in inference_server.py")
    print("3. Run: python inference_server.py")
    print("=" * 70)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
