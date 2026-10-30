import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss

class CustomDistillationLoss(v8DetectionLoss):
    """
    Custom Loss function that combines standard YOLOv8 loss with Knowledge Distillation (KL Divergence).
    """
    def __init__(self, model, teacher_model, temperature=4.0, alpha=0.7):
        super().__init__(model)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha # alpha is the weight for standard YOLO loss
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def __call__(self, preds, batch):
        """
        Compute loss for valid batches.
        preds: Student model predictions (list of 3 tensors for each scale)
        batch: Dictionary containing ground truth and input images
        """
        # 1. Compute Standard YOLO Loss (Box, Cls, DFL)
        loss, loss_items = super().__call__(preds, batch)
        
        # 2. Compute Distillation Loss
        dist_loss = torch.tensor(0.0, device=loss.device)
        
        try:
            with torch.no_grad():
                # Run teacher model
                teacher_preds = self.teacher_model.model(batch['img'])
            
            total_kl_loss = 0.0
            
            # Iterate over each scale head
            for i in range(len(preds)):
                student_out = preds[i]
                teacher_out = teacher_preds[i]
                
                if student_out.shape != teacher_out.shape:
                    continue 

                # Reshape to [Batch, Channels, -1] -> [Batch, -1, Channels]
                b, c, h, w = student_out.shape
                s_logits = student_out.view(b, c, -1).permute(0, 2, 1) 
                t_logits = teacher_out.view(b, c, -1).permute(0, 2, 1) 
                
                # Apply Temperature
                s_log_probs = F.log_softmax(s_logits / self.temperature, dim=-1)
                t_probs = F.softmax(t_logits / self.temperature, dim=-1)
                
                # KL Divergence (T^2 scaling is standard for KD)
                curr_kl = self.kl_loss(s_log_probs, t_probs) * (self.temperature ** 2)
                total_kl_loss += curr_kl

            # Average over heads
            dist_loss = total_kl_loss / len(preds)
            
            # Combine Losses: Total Loss = α × Task Loss + (1 - α) × Distillation Loss
            # Based on paper: α = 0.7 for task loss
            final_loss = (self.alpha * loss) + ((1.0 - self.alpha) * dist_loss)
            
            return final_loss, loss_items
            
        except Exception as e:
            return loss, loss_items

class DistillationTrainer(DetectionTrainer):
    """
    Custom Trainer that injects the Distillation Loss.
    """
    def __init__(self, teacher_model_path, *args, **kwargs):
        self.teacher_model_path = teacher_model_path
        self.teacher_model = None
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        # Load standard model (Student)
        model = super().get_model(cfg, weights, verbose)
        
        # Load Teacher Model securely
        from ultralytics import YOLO
        teacher = YOLO(self.teacher_model_path)
        self.teacher_model = teacher.model
        
        # Ensure teacher is in eval mode and frozen
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        return model

    def get_loss(self, loss_name=None):
        """
        Override to return CustomDistillationLoss.
        """
        criterion = CustomDistillationLoss(
            model=self.model,
            teacher_model=self.teacher_model,
            temperature=4.0, 
            alpha=0.7 # 0.7 weight for task loss, 0.3 for distillation
        )
        return criterion
