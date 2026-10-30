import torch
import torchvision
from ultralytics import YOLO
import numpy as np
import cv2
from pathlib import Path

class ClinicalEnsemble:
    """
    Advanced ensemble for dental diagnostics.
    Fuses YOLO (Phase 1 / High Speed) with Faster R-CNN (Phase 2 / High Precision).
    Includes FDI-style tooth indexing heuristics.
    """
    def __init__(self, yolo_path, rcnn_path, num_classes=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load YOLO
        self.yolo = YOLO(yolo_path)
        
        # Load Faster R-CNN
        self.rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.rcnn.roi_heads.box_predictor.cls_score.in_features
        self.rcnn.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        
        if Path(rcnn_path).exists():
            checkpoint = torch.load(rcnn_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.rcnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.rcnn.load_state_dict(checkpoint)
        
        self.rcnn.to(self.device)
        self.rcnn.eval()
        
        self.class_names = ['No Endodontic Treatment', 'Incomplete Endodontic Treatment', 
                            'Complete Endodontic Treatment', 'Total Endodontic Failure']

    def predict(self, image_path, conf_thresh=0.5):
        image = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 1. YOLO Inference
        yolo_res = self.yolo.predict(image, conf=conf_thresh, verbose=False)[0]
        yolo_boxes = []
        for box in yolo_res.boxes:
            yolo_boxes.append({
                'box': box.xyxy[0].tolist(),
                'conf': box.conf.item(),
                'cls': int(box.cls.item()),
                'source': 'yolo'
            })
            
        # 2. RCNN Inference
        img_tensor = torch.as_tensor(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            rcnn_res = self.rcnn(img_tensor)[0]
        
        rcnn_boxes = []
        for b, l, s in zip(rcnn_res['boxes'], rcnn_res['labels'], rcnn_res['scores']):
            if s > conf_thresh:
                rcnn_boxes.append({
                    'box': b.tolist(),
                    'conf': s.item(),
                    'cls': l.item() - 1, # Convert back to 0-indexed
                    'source': 'rcnn'
                })
        
        # 3. Fusion Logic (Simple NMS-based fusion)
        all_detections = yolo_boxes + rcnn_boxes
        fused = self._fuse_detections(all_detections, iou_thresh=0.45)
        
        # 4. Clinical Indexing (FDI Heuristic)
        indexed_results = self._apply_fdi_indexing(fused, w)
        
        return indexed_results

    def _fuse_detections(self, detections, iou_thresh):
        if not detections: return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
        keep = []
        used = [False] * len(detections)
        
        for i in range(len(detections)):
            if used[i]: continue
            
            keep.append(detections[i])
            used[i] = True
            
            for j in range(i + 1, len(detections)):
                if used[j]: continue
                if self._get_iou(detections[i]['box'], detections[j]['box']) > iou_thresh:
                    used[j] = True
                    # Optional: Could average boxes or boost confidence if both models found it
                    if detections[i]['source'] != detections[j]['source']:
                        keep[-1]['conf'] = min(1.0, keep[-1]['conf'] + 0.05) # "Agreement Boost"
                        
        return keep

    def _get_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)

    def _apply_fdi_indexing(self, detections, img_width):
        """
        Heuristic for relative tooth indexing.
        In a typical periapical X-ray, we label teeth from left-to-right.
        """
        if not detections: return []
        
        # Sort by x-coordinate (left to right)
        detections = sorted(detections, key=lambda x: x['box'][0])
        
        for idx, det in enumerate(detections):
            # Assign a clinical label based on position
            # This is a placeholder for real anatomical quadrant detection
            pos_label = f"Tooth_{idx + 1}" 
            det['clinical_index'] = pos_label
            det['class_name'] = self.class_names[det['cls']] if det['cls'] < len(self.class_names) else "Unknown"
            
        return detections

if __name__ == "__main__":
    print("Clinical Ensemble initialized for fusion and FDI indexing.")
    # Usage:
    # ensemble = ClinicalEnsemble('dental_yolo.pt', 'faster_rcnn_root_canal.pt')
    # results = ensemble.predict('test_xray.jpg')
