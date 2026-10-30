import os
import torch
from pathlib import Path
from ultralytics import YOLO

def main():
    print("=" * 70)
    print("Training YOLO11 on Dental Root Canal Dataset")
    print("=" * 70)

    # Setup Paths
    base_dir = Path(__file__).parent.absolute()
    
    # Using the exact dataset requested
    dataset_dir = Path(r"C:\Users\PRO\Desktop\Root Canal\Root-canal-AI-diagnostics\Root Canal.v1i.yolov8")
    
    # We need a data config YAML that points to this exact dataset.
    # The existing `data_roboflow_filtered.yaml` points to `Root Canal.v1i.yolov8/train/images` relative to the base directory.
    # Since both the script and dataset are in the same parent folder, we can seamlessly use the existing yaml.
    data_config = base_dir / "data_roboflow_filtered.yaml"
    
    output_model = base_dir / "dental_yolo11_custom.pt"

    print(f"Data config path: {data_config}")
    print(f"Dataset target: {dataset_dir}")
    print(f"Output model path: {output_model}\n")

    if not data_config.exists():
        print(f"Error: Could not find configuration file at {data_config}")
        print("Please ensure 'data_roboflow_filtered.yaml' exists in the main directory.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: Utilizing {device.upper()} for training")

    try:
        # Load the base YOLO11 weights we downloaded earlier
        model = YOLO("yolo11n.pt")
        print("✓ Loaded YOLO11n base model\n")
        
        print("Starting training (this may take a while depending on your GPU)...")
        # Start training
        results = model.train(
            data=str(data_config),
            epochs=100,            # 100 epochs is standard for fine-tuning
            imgsz=640,             # Keep resolution at 640x640 used by your system
            batch=16,              # Standard batch size
            device=device,
            patience=20,           # Early stopping if no improvement for 20 epochs
            save=True,
            verbose=True,
            project="yolo11_results",  # Save in a distinct folder
            name="dental_run"          
        )
        
        print("\n✓ Training completed successfully")
        
        # Optionally explicitly save the final weights to the main folder
        model.save(str(output_model))
        print(f"✓ Final trained model explicitly saved to: {output_model}")
        
        print("\n" + "=" * 70)
        print("Training absolute complete!")
        print("Make sure to update 'model_config.json' to use 'dental_yolo11_custom.pt' afterwards.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
