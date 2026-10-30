from ultralytics import YOLO
import os

def train_classification_model():
    print("Starting YOLO11 classification training...")
    
    # Load YOLO11 classification model
    # We use yolo11n-cls.pt (nano) for speed and efficiency, since the dataset is small (~500 images)
    model = YOLO("yolo11n-cls.pt")
    
    # Train the model
    # Note: data path must be absolute or relative to the current working directory
    # yolov8/11 classification expects the root directory containing 'train' and 'val'
    data_path = os.path.abspath("dataset_classification")
    
    results = model.train(
        data=data_path,
        epochs=50,       # 50 epochs should be enough for a small classification set to prevent extreme overfitting
        imgsz=224,       # Standard classification size
        batch=16,
        project="yolo11_cls_results",
        name="fractured_dental_cls",
        device="cpu",    # User is on a standard setup, we can use CPU or fallback to ultralytics auto
        patience=10
    )
    
    print("Training complete!")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    train_classification_model()
