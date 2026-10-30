import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_normal_rcnn_model(num_classes):
    """
    Creates a standard Faster R-CNN model as used in the baseline.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_optimized_rcnn_model(num_classes):
    """
    Creates a Faster R-CNN model optimized for Dental X-Rays.
    Matches the architecture defined in optimized_rcnn_trainer.py
    """
    # 1. Load Backbone with V2 Weights
    # We use resnet_fpn_backbone to get the FPN feature pyramid
    # Note: weights=None because we will load our own trained weights
    backbone = resnet_fpn_backbone(
        backbone_name='resnet50', 
        weights=None, 
        trainable_layers=5
    )
    
    # 2. Custom Anchor Generator
    # Teeth are often vertical (0.33, 0.5) or square (1.0). 
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
        box_nms_thresh=0.4,
        rpn_nms_thresh=0.6,
    )

    return model
