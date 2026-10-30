from ultralytics import YOLO
import os

def train_detection_model():
    print("Starting YOLO11 object detection training...")
    
    # Load YOLO11 object detection model
    model = YOLO("yolo11n.pt")
    
    # Use absolute path for the yaml config to prevent path resolution issues
    yaml_path = os.path.abspath("fractured_data.yaml")
    
    results = model.train(
        data=yaml_path,
        epochs=30,       # 30 epochs for a quick robust detection test
        imgsz=640,       # Standard detection size
        batch=8,         
        project="yolo11_det_results",
        name="fractured_dental_det",
        device="cpu",    
        patience=10
    )
    
    print("Training complete!")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    train_detection_model()
