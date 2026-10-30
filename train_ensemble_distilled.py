import os
import torch
from pathlib import Path
from ultralytics import YOLO

def main():
    print("=" * 70)
    print("Training Student (YOLO11n) on Ensemble Pseudo-Labels")
    print("=" * 70)

    base_dir = Path(__file__).parent.absolute()
    data_config = base_dir / "pseudo_data.yaml"
    output_model = base_dir / "dental_yolo11_ensemble_distilled.pt"

    print(f"Data config path: {data_config}")
    print(f"Output model path: {output_model}\n")

    if not data_config.exists():
        print(f"Error: Could not find configuration file at {data_config}")
        print("Please run 'python ensemble_pseudo_labeler.py' first!")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: Utilizing {device.upper()} for training")

    try:
        # We start with the base fast model again
        model = YOLO("yolo11n.pt")
        print("✓ Loaded YOLO11n student model\n")
        
        print("Starting distillation training on Ensemble's knowledge...")
        results = model.train(
            data=str(data_config),
            epochs=100,
            imgsz=640,
            batch=16,
            device=device,
            patience=20,
            save=True,
            verbose=True,
            project="yolo11_ensemble_distilled",
            name="run"          
        )
        
        print("\n✓ Training completed successfully")
        
        model.save(str(output_model))
        print(f"✓ Final explicitly saved to: {output_model}")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
