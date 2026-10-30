from ultralytics import YOLO
import argparse
import os
import torch


def train_yolo(
    data_config='data.yaml',
    epochs=100,
    imgsz=640,
    model_path='yolov8n.pt',
    output_model='dental_yolo.pt',
    use_filtered=False,
    batch_size=16,
    device=0,
    patience=20,
    resume=False
):
    """
    Train YOLO model for dental X-ray analysis
    
    Args:
        data_config: Path to data.yaml configuration
        epochs: Number of training epochs
        imgsz: Image size for training
        model_path: Path to pretrained model
        output_model: Path to save trained model
        use_filtered: Whether to use Contourlet-filtered images
        batch_size: Batch size for training
        device: GPU device index (0 for first GPU) or 'cpu'
        patience: Early stopping patience
        resume: Resume training from checkpoint
    """
    
    print("=" * 60)
    print("YOLO Dental Detection Model Training")
    print("=" * 60)
    
    if use_filtered:
        print("\n⚠️  Using Contourlet-filtered images for training")
        print("    Make sure you've run: python preprocess_dataset.py")
    else:
        print("\n✓ Training with raw images")
    
    # Auto-detect device if needed
    if device == "auto":
        if torch.cuda.is_available():
            device = 0
            print("\n✓ CUDA available, using GPU (device 0)")
        else:
            device = 'cpu'
            print("\n⚠️  CUDA not available, using CPU")
    elif isinstance(device, str) and device.isdigit():
        device = int(device)
        if not torch.cuda.is_available():
            print(f"\n⚠️  CUDA not available, switching from device {device} to CPU")
            device = 'cpu'
    elif isinstance(device, int):
        if not torch.cuda.is_available():
            print(f"\n⚠️  CUDA not available, switching from device {device} to CPU")
            device = 'cpu'
        else:
            print(f"\n✓ Using CUDA device: {device}")
    else:
        print(f"\n✓ Using device: {device}")
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}x{imgsz}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Data config: {data_config}")
    print(f"  Output model: {output_model}")
    print()
    
    try:
        model = YOLO(model_path)
        print(f"✓ Loaded pretrained model: {model_path}\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    try:
        print("Starting training...")
        results = model.train(
            data=data_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            patience=patience,
            resume=resume,
            save=True,
            verbose=True
        )
        
        print("\n✓ Training completed successfully")
        print(f"Results: {results}")
        
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        return False
    
    try:
        model.save(output_model)
        print(f"✓ Model saved: {output_model}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return False
    
    try:
        print(f"\nExporting to ONNX format...")
        model.export(format='onnx')
        print(f"✓ ONNX export completed")
    except Exception as e:
        print(f"⚠️  ONNX export warning: {e}")
    
    print("\n" + "=" * 60)
    print("Training pipeline completed!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLO model for dental X-ray analysis"
    )
    parser.add_argument(
        "--filtered",
        action="store_true",
        help="Use Contourlet-filtered images (requires preprocessing)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training (default: 640)"
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Pretrained model to use (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--output",
        default="dental_yolo.pt",
        help="Output model filename (default: dental_yolo.pt)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: 'cpu', 'cuda', or device index (default: auto-detect)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint"
    )
    
    args = parser.parse_args()
    
    success = train_yolo(
        epochs=args.epochs,
        imgsz=args.imgsz,
        model_path=args.model,
        output_model=args.output,
        use_filtered=args.filtered,
        batch_size=args.batch_size,
        device=args.device,
        patience=args.patience,
        resume=args.resume
    )
    
    exit(0 if success else 1)