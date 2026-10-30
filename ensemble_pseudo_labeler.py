import os
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm
from ensemble_detection import MedicalEnsemble

def main():
    print("=" * 70)
    print("Ensemble Pseudo-Label Generator (Super-Teacher Distillation)")
    print("=" * 70)

    # Configuration
    base_dir = Path(__file__).parent.absolute()
    dataset_dir = base_dir / "Root Canal.v1i.yolov8"
    
    # We will read from the existing 'train' and 'valid' image folders
    # And write new labels into 'pseudo_train' and 'pseudo_valid' folders
    splits = ['train', 'valid']
    
    output_dir = base_dir / "pseudo_dataset"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Creating pseudo-labels in: {output_dir}")

    # Initialize the Ensemble
    print("\nLoading the Consensus Ensemble models...")
    models = [
        str(base_dir / "hierarchical_results/1_teacher_v8m/weights/best.pt"),
        str(base_dir / "hierarchical_results/2_intermediate_v8n/weights/best.pt"),
        str(base_dir / "hierarchical_results/3_final_v5n/weights/best.pt")
    ]
    
    try:
        ensemble = MedicalEnsemble(models)
    except Exception as e:
        print(f"Failed to load ensemble: {e}")
        return

    # Process each split
    for split in splits:
        img_dir = dataset_dir / split / "images"
        lbl_out_dir = output_dir / split / "labels"
        img_out_dir = output_dir / split / "images"
        
        lbl_out_dir.mkdir(parents=True, exist_ok=True)
        img_out_dir.mkdir(parents=True, exist_ok=True)
        
        if not img_dir.exists():
            print(f"Skipping {split}, {img_dir} not found.")
            continue
            
        print(f"\nProcessing {split} split...")
        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        
        for img_path in tqdm(img_files, desc=f"Predicting {split}"):
            # Run Ensemble Inference
            # predict returns list of [x1, y1, x2, y2, conf, class_id]
            detections = ensemble.predict(str(img_path), conf=0.45, iou_thresh=0.5)
            
            # Read image to get dimensions for YOLO normalized format
            img = cv2.imread(str(img_path))
            if img is None: continue
            h, w = img.shape[:2]
            
            # Write labels in YOLO format (class x_center y_center width height)
            label_path = lbl_out_dir / f"{img_path.stem}.txt"
            with open(label_path, 'w') as f:
                for det in detections:
                    x1, y1, x2, y2, conf, cls_id = det
                    # Convert to normalized xywh
                    x_c = ((x1 + x2) / 2) / w
                    y_c = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    
                    # Ensure within bounds (0 to 1)
                    x_c, y_c = max(0, min(1, x_c)), max(0, min(1, y_c))
                    bw, bh = max(0, min(1, bw)), max(0, min(1, bh))
                    
                    f.write(f"{int(cls_id)} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
            
            # Create a symlink or copy the image so YOLO can find it seamlessly
            target_img = img_out_dir / img_path.name
            if not target_img.exists():
                # We copy on windows to be safe rather than mess with symlink privileges
                import shutil
                shutil.copy2(img_path, target_img)

    # 3. Create the data.yaml for the new pseudo dataset
    print("\nGenerating pseudo_data.yaml...")
    yaml_content = {
        'path': str(output_dir),
        'train': 'train/images',
        'val': 'valid/images',
        'nc': 4,
        'names': ['No Endodontic Treatment', 'Complete Endodontic Treatment', 
                 'Incomplete Endodontic Treatment', 'Total Endodontic Failure']
    }
    
    yaml_path = base_dir / "pseudo_data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"✓ Configuration saved to {yaml_path}")
    print("\n" + "=" * 70)
    print("Done! You can now train a student model on the Ensemble's knowledge")
    print("by running: python train_ensemble_distilled.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
