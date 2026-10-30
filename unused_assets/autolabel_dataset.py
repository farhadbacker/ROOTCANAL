import os
import csv
import torch
from pathlib import Path
from PIL import Image
import clip
from tqdm import tqdm

class EndodonticClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        self.class_names = [
            'No Endodontic Treatment',
            'Incomplete Endodontic Treatment',
            'Complete Endodontic Treatment',
            'Total Endodontic Failure'
        ]
        
        self.class_descriptions = [
            "dental X-ray with no root canal treatment, intact tooth structure",
            "dental X-ray with incomplete root canal treatment, partial filling in canal",
            "dental X-ray with complete root canal treatment, full canal filling",
            "dental X-ray with total endodontic failure, failed root canal treatment"
        ]
        
        self.text_features = self._encode_text_descriptions()
        print(f"✓ Model loaded on {self.device}")
        print(f"✓ Text features encoded for 4 classes\n")
    
    def _encode_text_descriptions(self):
        text_tokens = clip.tokenize([f"A {desc}" for desc in self.class_descriptions])
        text_tokens = text_tokens.to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def classify_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            similarity = image_features @ self.text_features.T
            predicted_class = similarity.argmax(dim=-1).item()
            confidence = similarity[0, predicted_class].item()
            
            return predicted_class, confidence
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, 0.0
    
    def autolabel_dataset(self, images_dir, output_csv='image_labels.csv'):
        images_dir = Path(images_dir)
        image_files = sorted([f for f in images_dir.glob('*.jpg')] + 
                            [f for f in images_dir.glob('*.JPG')] + 
                            [f for f in images_dir.glob('*.png')])
        
        if not image_files:
            print(f"✗ No images found in {images_dir}")
            return
        
        print(f"Found {len(image_files)} images to label\n")
        
        results = []
        
        for image_path in tqdm(image_files, desc="Auto-labeling images"):
            predicted_class, confidence = self.classify_image(str(image_path))
            
            if predicted_class is not None:
                results.append({
                    'image_path': image_path.name,
                    'label': predicted_class,
                    'class_name': self.class_names[predicted_class],
                    'confidence': round(confidence, 4)
                })
        
        csv_path = Path(images_dir.parent) / output_csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image_path', 'label', 'class_name', 'confidence'])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✓ Labels saved to {csv_path}")
        print(f"\nLabel Distribution:")
        for i, class_name in enumerate(self.class_names):
            count = sum(1 for r in results if r['label'] == i)
            percentage = (count / len(results)) * 100
            print(f"  {i}: {class_name:40} - {count:3d} images ({percentage:5.1f}%)")
        
        return results

if __name__ == "__main__":
    print("=" * 70)
    print("Endodontic X-ray Auto-Labeling with CLIP")
    print("=" * 70)
    print()
    
    images_dir = "dataset/images"
    
    classifier = EndodonticClassifier()
    results = classifier.autolabel_dataset(images_dir)
    
    if results:
        print("\n" + "=" * 70)
        print("Auto-labeling completed successfully!")
        print("=" * 70)
