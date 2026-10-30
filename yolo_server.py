from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
print("Loading YOLO model...")
model = YOLO('runs/detect/train/weights/best.pt')
print("Model loaded successfully")

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        image_data = data['image']

        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)

        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to numpy array
        image_np = np.array(image)

        # Run inference
        results = model(image_np)

        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates, class, and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    # Convert to normalized coordinates
                    height, width = image_np.shape[:2]
                    x = x1 / width
                    y = y1 / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height

                    class_names = ['No Endodontic Treatment', 'Incomplete Endodontic Treatment',
                                 'Complete Endodontic Treatment', 'Total Endodontic Failure']

                    detections.append({
                        'box': {
                            'x': float(x),
                            'y': float(y),
                            'width': float(w),
                            'height': float(h)
                        },
                        'label': class_names[cls] if cls < len(class_names) else f'Class {cls}',
                        'confidence': float(conf)
                    })

        return jsonify({'detections': detections})

    except Exception as e:
        print(f"Error in detection: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)