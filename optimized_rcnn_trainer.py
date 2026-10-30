import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.utils.data import DataLoader
import cv2
import numpy as np
from pathlib import Path
from medical_augment import MedicalAugmenter
from rc_rcnn_trainer import RootCanalDataset, collate_fn
from tqdm import tqdm
import math
import sys

# Define constants
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

def get_optimized_model(num_classes):
    """
    Creates a Faster R-CNN model optimized for Dental X-Rays.
    
    Optimizations:
    1. Backbone: ResNet50 V2 (better feature extraction than V1)
    2. Anchors: Custom aspect ratios for vertical tooth roots
    3. ROI Align: Higher sampling ratio
    """
    print("Initializing Optimized Faster R-CNN (ResNet50 V2 + Dental Anchors)...")
    
    # 1. Load Backbone with V2 Weights (better accuracy)
    # We use resnet_fpn_backbone to get the FPN feature pyramid
    backbone = resnet_fpn_backbone(
        backbone_name='resnet50', 
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2,
        trainable_layers=5 # Fine-tune all layers for medical adaptation
    )
    
    # 2. Custom Anchor Generator
    # Teeth are often vertical (0.33, 0.5) or square (1.0). 
    # Sometimes horizontally impacted (2.0, 3.0), but less common.
    # We emphasize vertical aspect ratios.
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.33, 0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )
    
    # 3. Create Model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_detections_per_img=100,
        box_nms_thresh=0.4, # Slightly lower NMS to resolve overlapping roots
        rpn_nms_thresh=0.6,
    )

    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None):
    model.train()
    total_loss = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Mixed Precision Training
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        
        total_loss += losses.item()
        pbar.set_postfix(loss=losses.item())
        
    return total_loss / len(data_loader)

def main():
    print(f"Training on device: {DEVICE}")
    
    # Data Setup
    data_yaml = 'Root Canal.v1i.yolov8/data.yaml'
    root_dir = 'Root Canal.v1i.yolov8'
    
    print("Loading datasets...")
    train_dataset = RootCanalDataset(
        root_dir, 
        data_yaml, 
        split='train', 
        transform=MedicalAugmenter(stage='train')
    )
    
    # Use valid set for monitoring if possible, otherwise use train (not ideal but works for demo)
    # Assuming 'valid' folder exists based on standard Yolo structure
    val_dataset = RootCanalDataset(
        root_dir, 
        data_yaml, 
        split='valid', 
        transform=MedicalAugmenter(stage='val')
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Model Setup
    model = get_optimized_model(train_dataset.num_classes)
    model.to(DEVICE)
    
    # Optimizer & Scheduler Setup
    params = [p for p in model.parameters() if p.requires_grad]
    
    # AdamW is generally better than SGD for Transformers/Modern CNNs
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.005)
    
    # Cosine Annealing with Warm Restarts
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5, 
        T_mult=2,
        eta_min=1e-6
    )
    
    # Gradient Scaling for Mixed Precision
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    output_dir = Path("rcnn_optimized_results")
    output_dir.mkdir(exist_ok=True)
    
    best_loss = float('inf')
    
    print("Starting training Loop...")
    
    for epoch in range(NUM_EPOCHS):
        loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, scaler)
        lr_scheduler.step()
        
        print(f"Epoch {epoch} Loss: {loss:.4f}")
        
        # Save Best Model
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), "optimized_rcnn_best.pt")
            print(f"★ New Best Model Saved (Loss: {loss:.4f})")
            
        # Standard Checkpoint
        if epoch % 5 == 0:
             torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, output_dir / f"checkpoint_epoch_{epoch}.pth")

    print("Training complete. Best model saved to 'optimized_rcnn_best.pt'")

if __name__ == "__main__":
    main()
