from ultralytics import YOLO
import os
import torch
from pathlib import Path
import sys

print("=" * 70)
print("Training YOLO with Roboflow Dataset")
print("=" * 70)

base_dir = Path(__file__).parent.absolute()
print(f"Script location: {__file__}")
print(f"Base directory: {base_dir}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Working directory: {os.getcwd()}\n")

data_config = str(base_dir / "data_roboflow.yaml")
output_model = str(base_dir / "dental_yolo_roboflow.pt")

print(f"Data config path: {data_config}")
print(f"Data config exists: {Path(data_config).exists()}")
print(f"Output model path: {output_model}\n")

try:
    model = YOLO("yolov8n.pt")
    print("✓ Loaded YOLOv8n model\n")
    
    print("Starting training...")
    results = model.train(
        data=data_config,
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        patience=20,
        save=True,
        verbose=True
    )
    
    print("\n✓ Training completed successfully")
    
    model.save(output_model)
    print(f"✓ Model saved: {output_model}")
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
