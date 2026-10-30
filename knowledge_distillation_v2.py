import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import xywh2xyxy
import os
from pathlib import Path

# ==============================================================================
# Custom Distillation Loss Class
# ==============================================================================
class DistillationLoss(v8DetectionLoss):
    def __init__(self, model, teacher_model, temperature=4.0, alpha=0.5):
        super().__init__(model)
        self.teacher = teacher_model
        self.T = temperature
        self.alpha = alpha
        
        # Cross-Entropy Loss (or KL Div) for softening class probabilities
        self.kd_loss = nn.KLDivLoss(reduction="batchmean")

    def __call__(self, preds, batch):
        # 1. Calculate Standard Student Loss (Task Loss)
        loss, loss_items = super().__call__(preds, batch)
        
        # 2. Get Teacher Predictions (Knowledge)
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])
        
        # 3. Calculate Distillation Loss
        # We focus on Class Probabilities (Logits) from the head
        student_logits = preds[1]  # Normally tuple: (features, (box_preds, cls_preds, ...))
        # Note: Structure depends on YOLO version, we need to extract cls score safely
        
        # Safer extraction of class logits for v8
        # preds is a list of [batch, anchors, classes+boxes] or similar depending on head
        # We'll use a simplified Approach B: Output-based distillation
        
        # To make this robust in Colab without digging into C++ source or complex head indexing:
        # We will DISTILL ONLY IF shapes match (which they might not for n vs m models)
        # So we resize or map teacher outputs.
        
        # SIMPLIFIED ROBUST IMPLEMENTATION:
        # If direct feature distillation is too brittle, we use Response-Based on Final Output
        # But for 'True' distillation we need logits.
        
        return loss, loss_items
        # (Placeholder: The actual implementation hooking into the Trainer requires 
        # subclassing the Trainer and overriding criterion.)

# ==============================================================================
# Robust Distillation Trainer
# ==============================================================================
class KDTrainer(DetectionTrainer):
    def __init__(self, teacher_model, temperature=4.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.T = temperature
        self.alpha = alpha
        
        # Move teacher to same device
        if torch.cuda.is_available():
            self.teacher.to('cuda')
        self.teacher.eval() # Ensure teacher is in eval mode

    def criterion(self, preds, batch):
        # Create standard loss instance if not exists
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = v8DetectionLoss(self.model)
            
        # 1. Standard Loss (Student vs Ground Truth)
        loss, loss_items = self.loss_fn(preds, batch)
        
        # 2. Distillation Loss (Student vs Teacher)
        # Get Teacher Outputs
        with torch.no_grad():
            teacher_preds = self.teacher(batch['img'])
            
        # Extract Logits (Class Probabilities)
        # Note: preds[1] contains list of 3 tensors (one for each scale P3, P4, P5)
        # shape: [Batch, Channels, GridH, GridW]
        # We need to match features or distill soft targets.
        
        # Iterating over the 3 scales (P3, P4, P5)
        for i, (student_pred, teacher_pred) in enumerate(zip(preds[1], teacher_preds[1])):
            # preds[1] is list of tensors: [Batch, 4+num_classes, GridH, GridW]
            
            # Check if shapes align (Grid sizes should match for same input size)
            if student_pred.shape[-2:] == teacher_pred.shape[-2:]:
               
                # Extract Class Logits: Channels 4 onwards (0-3 are box coords)
                s_cls = student_pred[:, 4:, :, :]
                t_cls = teacher_pred[:, 4:, :, :]
                
                # Check channel depth (should match num_classes)
                if s_cls.shape[1] == t_cls.shape[1]:
                    # Soften targets
                    t_soft = F.softmax(t_cls / self.T, dim=1)
                    s_log_soft = F.log_softmax(s_cls / self.T, dim=1)
                    
                    # KL Divergence
                    kd_loss_scale = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(s_log_soft, t_soft)
                    distill_loss += kd_loss_scale
            else:
                pass # Skip if grid sizes don't match (shouldn't happen with fixed imgsz)

        # Combine losses
        # total_loss = alpha * task_loss + (1-alpha) * distill_loss
        loss = (self.alpha * loss) + ((1 - self.alpha) * distill_loss)
        
        return loss, loss_items

# ==============================================================================
# Main Execution Script
# ==============================================================================
def run_true_distillation():
    print("Starting True Knowledge Distillation (Model-to-Model)...")
    
    # 1. Load Teacher
    teacher = YOLO("yolov8m.pt") # Or your trained teacher path
    
    # 2. Load Student
    model = YOLO("yolov8n.pt") 
    
    # 3. Define Training Args
    args = dict(
        model="yolov8n.pt",
        data="Root Canal.v1i.yolov8/data.yaml",
        epochs=50,
        imgsz=640,
        device=0 if torch.cuda.is_available() else 'cpu',
        project="distillation_results_v2",
        name="student_kd_true",
    )
    
    # 4. Initialize Custom Trainer
    trainer = KDTrainer(
        teacher_model=teacher.model,
        temperature=4.0,
        alpha=0.5,
        overrides=args
    )
    
    # 5. Train
    trainer.train()

if __name__ == "__main__":
    run_true_distillation()
