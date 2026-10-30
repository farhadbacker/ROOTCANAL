#!/usr/bin/env python3
import os
from pathlib import Path

os.chdir(Path(__file__).parent.absolute())

print("=" * 70)
print("TRAINING VERIFICATION")
print("=" * 70)

runs_dir = Path("runs/detect")
if runs_dir.exists():
    train_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and "train" in d.name])
    print(f"\nFound {len(train_dirs)} training run(s):\n")
    for train_dir in train_dirs:
        print(f"  📁 {train_dir.name}")
        weights = train_dir / "weights" / "best.pt"
        last = train_dir / "weights" / "last.pt"
        results_csv = train_dir / "results.csv"
        
        if weights.exists():
            size_mb = weights.stat().st_size / (1024 * 1024)
            print(f"    ✓ Best model: {weights.name} ({size_mb:.1f} MB)")
        if last.exists():
            print(f"    ✓ Last checkpoint: {last.name}")
        if results_csv.exists():
            print(f"    ✓ Results CSV exists")

print("\n" + "=" * 70)
print("Checking saved model...")
print("=" * 70)

model_file = Path("dental_yolo_roboflow.pt")
if model_file.exists():
    size_mb = model_file.stat().st_size / (1024 * 1024)
    print(f"\n✓ Model saved: dental_yolo_roboflow.pt ({size_mb:.1f} MB)")
else:
    print(f"\n✗ Model not found: dental_yolo_roboflow.pt")

print("\n" + "=" * 70)
