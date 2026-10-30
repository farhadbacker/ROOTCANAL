from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Export to TensorFlow.js format
model.export(format='tfjs')

print("Model exported to TensorFlow.js format")