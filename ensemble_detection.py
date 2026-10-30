from ultralytics import YOLO
import numpy as np

def box_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

class MedicalEnsemble:
    def __init__(self, model_paths):
        self.models = [YOLO(p) for p in model_paths]
        print(f"Medical Ensemble initialized with {len(self.models)} models.")
        
    def predict(self, image_path, conf=0.45, iou_thresh=0.5):
        """
        Consensus Prediction: Aggregates boxes and only keeps those found by >1 model.
        This focuses exclusively on Precision (reducing False Positives).
        """
        all_boxes = [] # List of [x1, y1, x2, y2, conf, class_id]
        
        for model in self.models:
            res = model.predict(image_path, conf=conf, verbose=False)[0]
            for box in res.boxes:
                coords = box.xyxy[0].tolist()
                c = box.conf.item()
                cls = int(box.cls.item())
                all_boxes.append(coords + [c, cls])

        if not all_boxes:
            return []

        # Consensus logic
        final_consensus_boxes = []
        used = [False] * len(all_boxes)
        
        for i in range(len(all_boxes)):
            if used[i]: continue
            
            matches = [all_boxes[i]]
            used[i] = True
            
            for j in range(i + 1, len(all_boxes)):
                if used[j]: continue
                if all_boxes[i][5] != all_boxes[j][5]: continue # Must be same class
                
                if box_iou(all_boxes[i][:4], all_boxes[j][:4]) > iou_thresh:
                    matches.append(all_boxes[j])
                    used[j] = True
            
            # CONSENSUS RULE: Only keep if >= 2 models agreed on this pathology
            if len(matches) >= 2:
                # Average the coordinates and confidence
                avg_box = np.mean([m[:4] for m in matches], axis=0)
                avg_conf = np.mean([m[4] for m in matches])
                final_consensus_boxes.append(list(avg_box) + [avg_conf, matches[0][5]])

        return final_consensus_boxes

if __name__ == "__main__":
    # Example usage:
    # models = ['hierarchical_results/1_teacher_v8m/weights/best.pt', 
    #           'hierarchical_results/3_final_v5n/weights/best.pt']
    # ensemble = MedicalEnsemble(models)
    print("Consensus Ensemble module ready for ultra-high precision validation.")
