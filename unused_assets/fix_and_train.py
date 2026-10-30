#!/usr/bin/env python3
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parent.absolute())

from ultralytics import YOLO
import torch

print("=" * 70)
print("Training YOLO with Roboflow Dataset")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

yaml_content = """path: .
train: "Root Canal.v1i.yolov8/train/images"
val: "Root Canal.v1i.yolov8/valid/images"
test: "Root Canal.v1i.yolov8/test/images"

nc: 4
names: ['No Endodontic Treatment', 'Complete Endodontic Treatment', 'Incomplete Endodontic Treatment', 'Total Endodontic Failure']
"""

with open("data_roboflow.yaml", "w") as f:
    f.write(yaml_content)

print("✓ Fixed data_roboflow.yaml\n")

try:
    model = YOLO("yolov8n.pt")
    print("✓ Loaded YOLOv8n model\n")
    
    print("Starting training...")
    print("This may take 30-60 minutes depending on your hardware\n")
    
    results = model.train(
        data="data_roboflow.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        patience=20,
        save=True,
        verbose=True
    )
    
    print("\n✓ Training completed successfully")
    
    model.save("dental_yolo_roboflow.pt")
    print(f"✓ Model saved: dental_yolo_roboflow.pt")
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
