#!/usr/bin/env python3

import os
import csv
import torch
from pathlib import Path
from PIL import Image
import clip
from tqdm import tqdm
import sys

def main():
    base_dir = Path(__file__).parent.absolute()
    os.chdir(base_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    class_names = [
        'No Endodontic Treatment',
        'Incomplete Endodontic Treatment',
        'Complete Endodontic Treatment',
        'Total Endodontic Failure'
    ]
    
    class_descriptions = [
        "dental X-ray with no root canal treatment, intact tooth structure",
        "dental X-ray with incomplete root canal treatment, partial filling in canal",
        "dental X-ray with complete root canal treatment, full canal filling",
        "dental X-ray with total endodontic failure, failed root canal treatment"
    ]
    
    text_tokens = clip.tokenize([f"A {desc}" for desc in class_descriptions])
    text_tokens = text_tokens.to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    print(f"✓ Model loaded")
    print(f"✓ Text features encoded for 4 classes\n")
    
    images_dir = base_dir / "dataset" / "images"
    image_files = sorted(list(images_dir.glob('*.jpg')) + 
                        list(images_dir.glob('*.JPG')) + 
                        list(images_dir.glob('*.png')))
    
    if not image_files:
        print(f"✗ No images found in {images_dir}")
        return False
    
    print(f"Found {len(image_files)} images to label\n")
    
    results = []
    
    for image_path in tqdm(image_files, desc="Auto-labeling images"):
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            similarity = image_features @ text_features.T
            predicted_class = similarity.argmax(dim=-1).item()
            confidence = similarity[0, predicted_class].item()
            
            results.append({
                'image_path': image_path.name,
                'label': predicted_class,
                'class_name': class_names[predicted_class],
                'confidence': round(confidence, 4)
            })
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue
    
    csv_path = base_dir / 'image_labels.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'label', 'class_name', 'confidence'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Labels saved to {csv_path}")
    print(f"\nLabel Distribution:")
    for i, class_name in enumerate(class_names):
        count = sum(1 for r in results if r['label'] == i)
        percentage = (count / len(results)) * 100 if results else 0
        print(f"  {i}: {class_name:40} - {count:3d} images ({percentage:5.1f}%)")
    
    return True

if __name__ == "__main__":
    print("=" * 70)
    print("Endodontic X-ray Auto-Labeling with CLIP")
    print("=" * 70)
    print()
    
    try:
        success = main()
        if success:
            print("\n" + "=" * 70)
            print("Auto-labeling completed successfully!")
            print("=" * 70)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
