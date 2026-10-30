#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

os.chdir(Path(__file__).parent.absolute())

print("=" * 70)
print("Training YOLO with Filtered Roboflow Dataset")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

roboflow_dir = Path("Root Canal.v1i.yolov8")
train_dir = roboflow_dir / "train" / "images"
val_dir = roboflow_dir / "valid" / "images"

print(f"\nChecking dataset...")
print(f"  Train images: {len(list(train_dir.glob('*.jpg')))} files")
print(f"  Valid images: {len(list(val_dir.glob('*.jpg')))} files")

yaml_content = """path: .
train: "Root Canal.v1i.yolov8/train/images"
val: "Root Canal.v1i.yolov8/valid/images"
test: "Root Canal.v1i.yolov8/test/images"

nc: 4
names: ['No Endodontic Treatment', 'Complete Endodontic Treatment', 'Incomplete Endodontic Treatment', 'Total Endodontic Failure']
"""

with open("data_roboflow_filtered.yaml", "w") as f:
    f.write(yaml_content)

print("\n✓ YAML config created\n")

try:
    model = YOLO("yolov8n.pt")
    print("✓ Loaded YOLOv8n model\n")
    
    print("Starting training with filtered images...")
    print("This will take 30-60 minutes\n")
    
    results = model.train(
        data="data_roboflow_filtered.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        patience=20,
        save=True,
        verbose=True,
        project=None,
        name="train_filtered"
    )
    
    print("\n✓ Training completed successfully")
    
    model.save("dental_yolo_roboflow_filtered.pt")
    print(f"✓ Model saved: dental_yolo_roboflow_filtered.pt")
    
    print("\n" + "=" * 70)
    print("Training with filter completed!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
