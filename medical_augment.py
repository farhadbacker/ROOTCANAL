import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

class MedicalAugmenter:
    """
    Specialized augmentation pipeline for dental radiographs and medical scans.
    Focuses on preserving anatomical structures while simulating variations in 
    exposure, contrast, and subtle geometric deformations common in dental X-rays.
    """
    def __init__(self, stage='train'):
        self.stage = stage
        self.transform = self._get_train_transforms() if stage == 'train' else self._get_val_transforms()

    def _get_train_transforms(self):
        return A.Compose([
            # 1. Geometric - Simulates slight angle variations in X-ray positioning
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            
            # 2. Medical Specific - Simulation of camera/sensor noise and artifacts
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
            ], p=0.3),
            
            # 3. Clinical Precision - Simulates different exposure levels and equipment contrast
            A.OneOf([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            ], p=0.7),
            
            # 4. Anatomical Deformation - Elastic transforms simulate patient movement or detector curve
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            
            # 5. Flip - Dental radiographs can be oriented differently but usually fixed
            A.HorizontalFlip(p=0.5),
            
            # Normalization and Tensor conversion
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    def _get_val_transforms(self):
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    def __call__(self, image, bboxes=None, class_labels=None):
        if bboxes is not None and class_labels is not None:
            return self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        return self.transform(image=image)

if __name__ == "__main__":
    # Test script
    augmenter = MedicalAugmenter(stage='train')
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_bboxes = [[10, 10, 100, 100]] # COCO format: [x_min, y_min, width, height]
    dummy_labels = [0]
    
    augmented = augmenter(dummy_img, dummy_bboxes, dummy_labels)
    print("Augmentation successful. Output tensor shape:", augmented['image'].shape)
