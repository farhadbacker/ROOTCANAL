import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
from pathlib import Path
from medical_augment import MedicalAugmenter
import yaml
from tqdm import tqdm

class RootCanalDataset(Dataset):
    """
    Custom Dataset for Root Canal Failure Diagnosis.
    Converts YOLO format to PyTorch Faster R-CNN format.
    """
    def __init__(self, root_dir, data_yaml_path, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        self.classes = data_config['names']
        # Torchvision classes start from 1 (0 is background)
        self.num_classes = len(self.classes) + 1 
        
        self.img_dir = self.root_dir / split / 'images'
        self.label_dir = self.root_dir / split / 'labels'
        self.img_files = list(self.img_dir.glob('*.jpg')) + list(self.img_dir.glob('*.png'))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_dir / (img_path.stem + '.txt')
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    # YOLO: class x_center y_center width height (normalized)
                    cls_id, x_c, y_c, w, h = map(float, line.split())
                    
                    # Convert to Absolute COCO [x_min, y_min, width, height]
                    # for our MedicalAugmenter (albumentations)
                    x_min = (x_c - w/2) * width
                    y_min = (y_c - h/2) * height
                    bbox_w = w * width
                    bbox_h = h * height
                    
                    boxes.append([x_min, y_min, bbox_w, bbox_h])
                    labels.append(int(cls_id) + 1) # +1 for background offset

        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        else:
            # Baseline normalization if no transform
            image = image.astype(np.float32) / 255.0
            image = torch.as_tensor(image).permute(2, 0, 1)
        
        # Convert COCO boxes [x_min, y_min, w, h] to RCNN boxes [x_min, y_min, x_max, y_max]
        if len(boxes) > 0:
            boxes = np.array(boxes)
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2] # x_max
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3] # y_max
            target_boxes = torch.as_tensor(boxes, dtype=torch.float32)
            target_labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            target_boxes = torch.zeros((0, 4), dtype=torch.float32)
            target_labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": target_boxes,
            "labels": target_labels,
            "image_id": torch.tensor([idx])
        }
        
        return image, target

def get_model(num_classes):
    # Load Faster R-CNN with ResNet50-FPN backbone pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Replace the classifier with a new one for our dental classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        pbar.set_postfix(loss=losses.item())
    
    return total_loss / len(data_loader)

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on device: {device}")
    
    data_yaml = 'Root Canal.v1i.yolov8/data.yaml'
    train_dataset = RootCanalDataset(
        'Root Canal.v1i.yolov8', 
        data_yaml, 
        split='train', 
        transform=MedicalAugmenter(stage='train')
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=collate_fn
    )
    
    model = get_model(train_dataset.num_classes)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    num_epochs = 10
    output_dir = Path("rcnn_results")
    output_dir.mkdir(exist_ok=True)
    
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, output_dir / f"checkpoint_epoch_{epoch}.pth")
    
    # Save best model
    torch.save(model.state_dict(), "faster_rcnn_root_canal.pt")
    print("Training complete. Model saved to faster_rcnn_root_canal.pt")

if __name__ == "__main__":
    main()
