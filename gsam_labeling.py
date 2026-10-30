import torch
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# Note: GSAM usually requires:
# pip install git+https://github.com/IDEA-Research/Grounded-Segment-Anything.git
# and specific weights for Grounding DINO and SAM.

import torch
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class GroundedSAMLabeler:
    """
    High-Precision Labeler using HuggingFace Transformers implementation of Grounding DINO.
    This acts as the 'Super-Teacher' for the distillation hierarchy.
    Crucially, this uses standard pip-installable libraries to avoid CUDA build errors.
    """
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        
        self.prompts = {
            "root_canal_failure": "tooth with failed root canal treatment.",
            "incomplete_treatment": "incomplete root canal filling.",
            "complete_treatment": "well-filled root canal.",
            "cavity": "dental cavity."
        }
        print(f"GSAM (Transformers) initialized on {self.device}")

    def label_image(self, image_path, output_yolo_path):
        """
        Runs Grounding DINO zero-shot detection.
        """
        image = Image.open(image_path).convert("RGB")
        combined_prompt = " . ".join(self.prompts.values())
        
        inputs = self.processor(images=image, text=combined_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process detections
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.3,
            target_sizes=[image.size[::-1]]
        )[0]

        # Convert to YOLO format [class_id, x_center, y_center, width, height]
        w, h = image.size
        yolo_lines = []
        
        # Mapping detected labels to our 4 classes
        label_map = {
            "root_canal_failure": 0,
            "incomplete_treatment": 2,
            "complete_treatment": 1,
            "cavity": 3
        }

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # If label is a tensor/int, look it up. If it's a string, use it directly.
            if hasattr(label, 'item'):
                label_name = self.model.config.id2label[label.item()].lower()
            else:
                label_name = str(label).lower()
            
            # Simple matching logic (e.g. "root_canal_failure" -> 0)
            class_id = 0 # Default
            for key, val in label_map.items():
                if key in label_name:
                    class_id = val
                    break
            xmin, ymin, xmax, ymax = box.tolist()
            xc = (xmin + xmax) / 2 / w
            yc = (ymin + ymax) / 2 / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            
            yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        with open(output_yolo_path, 'w') as f:
            f.writelines(yolo_lines)

    def process_dataset(self, images_dir, labels_output_dir):
        images_dir = Path(images_dir)
        labels_output_dir = Path(labels_output_dir)
        labels_output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        print(f"Starting Super-Teacher labeling for {len(image_files)} images...")
        
        for img_path in tqdm(image_files, desc="GSAM Labeling"):
            yolo_file = labels_output_dir / f"{img_path.stem}.txt"
            try:
                self.label_image(str(img_path), str(yolo_file))
            except Exception as e:
                print(f"Error labeling {img_path.name}: {e}")
                
if __name__ == "__main__":
    # Example usage for the User
    labeler = GroundedSAMLabeler()
    # labeler.process_dataset('dataset/images', 'dataset/gsam_labels')
    print("Grounded SAM Labeling script ready for deployment.")
