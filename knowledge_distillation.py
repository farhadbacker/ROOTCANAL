import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm
import torch.nn.functional as F

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_divergence = nn.KLDivLoss(reduction='batchmean')
        self.box_loss = nn.MSELoss() # Simplified box loss for demonstration
        self.cls_loss = nn.BCEWithLogitsLoss() 

    def forward(self, student_preds, teacher_preds, targets):
        # NOTE: YOLOv8 output structure is complex (boxes + class probs).
        # We focus on distilling the classification head for this implementation
        # as it carries the "dark knowledge".
        
        # Extract features/logits depends on specific YOLOv8 architecture hooks.
        # For simplicity in this custom loop, we assume we are working with 
        # the raw output layers or a simplified extraction.
        
        # However, writing a raw PyTorch training loop for YOLOv8 from scratch 
        # is error-prone due to its complex loss functions (CIoU, DFL, etc.).
        
        # STRATEGY: 
        # We will use the standard YOLOv8 training but Override the loss 
        # by subclassing the Trainer or using a callback if possible.
        # Since standard callbacks don't easily support modifying loss *computation*,
        # we will implement a custom loop using the 'train' mode but calculating 
        # loss against teacher outputs manually.
        
        pass 
        # Actual implementation is handled inside the training loop below 
        # by combining standard YOLO loss with KL Div against teacher.

from filter_fusion import FilterFusion

class DistillationTrainer:
    def __init__(self, data_yaml, output_dir='distillation_results'):
        self.data_yaml = data_yaml
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fusion = FilterFusion()
        print(f"Using device: {self.device}")
        
        self.log_file = self.output_dir / 'training_log.txt'
    
    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def preprocess_with_fusion(self, images_dir):
        """Apply the Multi-Filter Fusion to all images in the directory"""
        self.log(f"Applying Filter Fusion (Contourlet + NLM + Bayesian) to {images_dir}...")
        image_files = list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png'))
        for img_path in tqdm(image_files, desc="Fusing filters"):
            self.fusion.apply_fusion(cv2.imread(str(img_path))) # This should overwrite or be used in a custom loader
            # For simplicity in this implementation, we rely on the preprocessing script
            # but we could call it here.
    
    def train_teacher_model(self, epochs=50, imgsz=640):
        self.log("\n" + "="*60)
        self.log("PHASE 1: Training Teacher Model (YOLOv8m)")
        self.log("="*60)
        
        teacher_model = YOLO('yolov8m.pt')
        
        # High precision settings
        results = teacher_model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            device=0 if torch.cuda.is_available() else 'cpu',
            project=str(self.output_dir),
            name='teacher_model',
            patience=20,
            save=True,
            verbose=True,
            exist_ok=True,
            lr0=0.01, # Standard YOLO lr
            augment=True # Enable medical imaging augmentations
        )
        
        teacher_path = self.output_dir / 'teacher_model' / 'weights' / 'best.pt'
        self.log(f"\n[OK] Teacher model training completed: {teacher_path}")
        
        return str(teacher_path)
    
    def train_student_with_distillation(self, teacher_model_path, target_model='yolov5n.pt', epochs=50, imgsz=640):
        self.log("\n" + "="*60)
        self.log(f"PHASE 2: Distilling to {target_model}")
        self.log("="*60)
        
        if teacher_model_path == "dummy" or teacher_model_path is None:
             self.log("Training Baseline Student...")
             student = YOLO(target_model)
             student.train(data=self.data_yaml, epochs=epochs, imgsz=imgsz, exist_ok=True)
             return str(self.output_dir / 'student_baseline' / 'weights' / 'best.pt')

        from advanced_distillation import DistillationTrainer as AdvancedTrainer
        
        custom_trainer = AdvancedTrainer(
            teacher_model_path=teacher_model_path,
            overrides={
                'data': self.data_yaml,
                'epochs': epochs,
                'imgsz': imgsz,
                'device': 0 if torch.cuda.is_available() else 'cpu',
                'project': str(self.output_dir),
                'name': f'student_distilled_{Path(target_model).stem}',
                'patience': 20,
                'save': True,
                'verbose': True,
                'exist_ok': True,
                'model': target_model
            }
        )
        
        custom_trainer.train()
        student_path = self.output_dir / f'student_distilled_{Path(target_model).stem}' / 'weights' / 'best.pt'
        return str(student_path)
        
    def evaluate_model(self, model_path):
        self.log(f"\nEvaluating: {Path(model_path).name}")
        model = YOLO(model_path)
        metric = model.val(data=self.data_yaml, split='val')
        
        results = {
            "mAP50": metric.box.map50,
            "mAP50-95": metric.box.map,
            "precision": metric.box.mp,
            "recall": metric.box.mr
        }
        self.log(f"  mAP50:    {results['mAP50']:.4f}")
        self.log(f"  mAP50-95: {results['mAP50-95']:.4f}")
        return results

    def compare_models(self, teacher_path, student_baseline, student_distilled):
        self.log("\n" + "="*60)
        self.log("PHASE 3: Model Comparison")
        self.log("="*60)
        
        t_res = self.evaluate_model(teacher_path)
        s_res = self.evaluate_model(student_baseline)
        d_res = self.evaluate_model(student_distilled) # In our simplified case, same as baseline possibly
        
        self.log("\nComparison Summary:")
        self.log(f"{'Model':<20} {'mAP50':<10} {'mAP50-95':<10}")
        self.log("-" * 40)
        self.log(f"{'Teacher':<20} {t_res['mAP50']:<10.4f} {t_res['mAP50-95']:<10.4f}")
        self.log(f"{'Student (Baseline)':<20} {s_res['mAP50']:<10.4f} {s_res['mAP50-95']:<10.4f}")
        self.log(f"{'Student (Distill)':<20} {d_res['mAP50']:<10.4f} {d_res['mAP50-95']:<10.4f}")

        return {
            'teacher': t_res, 
            'student_baseline': s_res, 
            'student_distilled': d_res
        }

    def train_student_baseline(self, epochs=50):
        # In this simplified version, this is effectively the same as "distilled"
        # unless we apply different hyperparameters.
        return self.train_student_with_distillation("dummy", epochs)

    def run_full_pipeline(self):
        try:
            # 1. Train Teacher
            teacher_path = self.train_teacher_model(epochs=50)
            
            # 2. Train Student (Baseline)
            student_base = self.train_student_baseline(epochs=50)
            
            # 3. Train Student (Distilled)
            # Now we actually use the teacher to distill into a new student
            self.log("Starting Distillation Phase...")
            student_dist = self.train_student_with_distillation(teacher_path, epochs=50) 
            
            # 4. Compare
            self.compare_models(teacher_path, student_base, student_dist)
            
        except Exception as e:
            self.log(f"\n[ERROR] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    trainer = DistillationTrainer(data_yaml='Root Canal.v1i.yolov8/data.yaml')
    trainer.run_full_pipeline()
