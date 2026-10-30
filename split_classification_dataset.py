import os
import shutil
import random
from pathlib import Path

def split_dataset(src_dir, dest_dir, split_ratio=0.8):
    src = Path(src_dir)
    dest = Path(dest_dir)
    
    # Create train and val directories
    train_dir = dest / 'train'
    val_dir = dest / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Iterate over class directories
    for class_dir in src.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            
            # Create class directories in train and val
            (train_dir / class_name).mkdir(exist_ok=True)
            (val_dir / class_name).mkdir(exist_ok=True)
            
            # Get all images for this class
            images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
            random.shuffle(images)
            
            # Calculate split point
            split_idx = int(len(images) * split_ratio)
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            # Copy images
            for img in train_images:
                shutil.copy2(img, train_dir / class_name / img.name)
            for img in val_images:
                shutil.copy2(img, val_dir / class_name / img.name)
                
            print(f"Class '{class_name}': {len(train_images)} train, {len(val_images)} val")

if __name__ == '__main__':
    src = r"fractured datset\Dental OPG XRAY Dataset\Dental OPG XRAY Dataset\Dental OPG (Classification)"
    dest = "dataset_classification"
    
    print("Splitting dataset...")
    # Set seed for reproducibility
    random.seed(42)
    split_dataset(src, dest)
    print("Done!")
