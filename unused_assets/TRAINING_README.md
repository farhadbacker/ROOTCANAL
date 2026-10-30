# Dental AI Diagnostics System

## Architecture
This system uses a **client-server architecture**:
- **Frontend**: React app with TypeScript (runs on http://localhost:3002)
- **Backend**: Python Flask server with YOLO inference (runs on http://localhost:5000)

## Training Dental YOLO Model

### Setup
1. Python environment is set up with required dependencies (Ultralytics YOLOv8, Flask, etc.)

### Prepare Dataset
1. Place your dental X-ray images in `dataset/images/`
2. Create corresponding label files in `dataset/labels/` with YOLO format:
   - Each label file should have the same name as the image (e.g., `image1.jpg` -> `image1.txt`)
   - Format: `<class> <x_center> <y_center> <width> <height>`
   - Classes:
     - 0: No Endodontic Treatment
     - 1: Incomplete Endodontic Treatment
     - 2: Complete Endodontic Treatment
     - 3: Total Endodontic Failure
   - Coordinates are normalized (0-1)

### Train Model
Run: `python train_yolo.py`

This will:
- Train YOLOv8 on your dataset for 100 epochs
- Save the model as `dental_yolo.pt`
- Export to ONNX format as `dental_yolo.onnx`

## Running the System

### Option 1: Automated Startup
Run: `start_servers.bat`

This starts both servers automatically.

### Option 2: Manual Startup

1. **Start Inference Server**:
   ```bash
   python inference_server.py
   ```
   Server runs on http://localhost:5000

2. **Start React App**:
   ```bash
   npm run dev
   ```
   App runs on http://localhost:3002

## How It Works

1. User uploads dental X-ray image in React app
2. Image is sent to Python Flask server via HTTP POST
3. Server runs YOLO inference on the trained dental model
4. Detection results are sent back to React app
5. Results are displayed with bounding boxes and classifications

## Fallback System

If the YOLO model fails to load or run:
1. Falls back to COCO-SSD (general object detection)
2. Finally falls back to mock dental detections for testing

## API Endpoints

- `GET /health` - Health check
- `POST /detect` - Run YOLO detection on uploaded image

## Notes
- Ensure you have sufficient labeled data (recommended: 100+ images per class)
- Training may take time depending on your hardware
- The system provides real-time dental diagnostics with your trained model





to run
npm run dev

another terminal
python inference_server.py