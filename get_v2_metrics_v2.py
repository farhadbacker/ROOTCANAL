from ultralytics import YOLO
import numpy as np

def main():
    model_path = r'c:\Users\PRO\Desktop\Root Canal\Root-canal-AI-diagnostics\distillation_results_v2\student_kd_true\weights\best.pt'
    data_yaml = r'c:\Users\PRO\Desktop\Root Canal\Root-canal-AI-diagnostics\Root Canal.v1i.yolov8\data.yaml'
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print("Running validation...")
    results = model.val(data=data_yaml, imgsz=640, conf=0.25, verbose=False)
    
    # Access Confusion Matrix
    # Ultralytics confusion matrix is typically (N+1)x(N+1) where last row/col is background/no-detection
    # But usually results.confusion_matrix.matrix gives us what we need
    
    try:
        cm = results.confusion_matrix.matrix
        print("\nConfusion Matrix Shape:", cm.shape)
        print(cm)
        
        class_names = results.names
        print("\nClass Mapping:", class_names)
        
        print("\nMedical Metrics:")
        print("-" * 50)
        
        for i, class_id in enumerate(class_names):
            class_name = class_names[class_id]
            
            # True Positives
            tp = cm[i, i]
            
            # False Negatives (Sum of row i - TP) => Missed detections for this class
            fn = np.sum(cm[i, :]) - tp
            
            # False Positives (Sum of col i - TP) => Wrongly predicted as this class
            fp = np.sum(cm[:, i]) - tp
            
            # True Negatives (Total - (TP + FN + FP))
            tn = np.sum(cm) - (tp + fn + fp)
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print(f"Class: {class_name}")
            print(f"  Sensitivity (Recall): {sensitivity:.4f}")
            print(f"  Specificity:          {specificity:.4f}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error computing metrics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
