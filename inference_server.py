
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
import base64
from io import BytesIO
from PIL import Image
import ultralytics
from ultralytics import YOLO
from contourlet_filter import ContourletTransform
from grad_cam import YOLOGradCAM, overlay_heatmap
from ensemble_detection import MedicalEnsemble
import json
from pathlib import Path
from rcnn_utils import get_optimized_rcnn_model, get_normal_rcnn_model

app = Flask(__name__)
CORS(app)

config = {}
try:
    with open('model_config.json', 'r') as f:
        config = json.load(f)
except:
    print("⚠️  model_config.json not found, using defaults")

default_model_name = config.get('default_model', 'student_distilled')
model_specs = config.get('available_models', {})
model_path = model_specs.get(default_model_name, {}).get('path', 'dental_yolo_roboflow_filtered.pt')
use_filter = config.get('preprocessing', {}).get('contourlet_enabled', True)

# Global variables to hold active models
active_model = None         # Can be YOLO object or FasterRCNN object
active_model_type = "yolo"  # "yolo", "rcnn", "ensemble"
ensemble = None
contourlet_filter = None

# Dental Conditions Model
dental_condition_model = None
dental_condition_model_path = "yolo11_det_results/fractured_dental_det/weights/best.pt"

dental_class_names = [
  "Impacted Teeth",
  "Infection",
  "Caries",
  "Fractured Teeth",
  "BDC-BDR",
  "Healthy Teeth"
]

class_names = [
    'No Endodontic Treatment',
    'Complete Endodontic Treatment',
    'Incomplete Endodontic Treatment',
    'Total Endodontic Failure'
]

def load_model_by_name(name):
    """Loads a model by its name from config and returns (model_obj, type_str)"""
    global ensemble

    if name not in model_specs:
        raise ValueError(f"Unknown model: {name}")

    spec = model_specs[name]
    path = spec.get('path')
    
    # Handle Ensemble
    if name == 'ensemble':
        paths = spec.get('paths', [])
        try:
            ens = MedicalEnsemble(paths)
            print(f"✓ Ensemble loaded successfully")
            return ens, "ensemble"
        except Exception as e:
            print(f"✗ Error loading ensemble: {e}")
            return None, "error"

    # Handle R-CNN
    if 'rcnn' in name or 'rcnn' in spec.get('name', '').lower():
        try:
            # Initialize architecture
            # +1 for background class in R-CNN (usually handled by num_classes=len(class_names)+1)
            # However, our trained model might have been trained with strictly len(class_names) + 1 (0 is background)
            # Let's assume standard torchvision FasterRCNN behavior: 0 is background.
            # We need to know how many classes the model was trained on.
            # Assuming 4 classes + background = 5
            if name == 'normal_rcnn':
                rcnn = get_normal_rcnn_model(num_classes=len(class_names) + 1)
            else:
                rcnn = get_optimized_rcnn_model(num_classes=len(class_names) + 1)
            
            # Load weights
            checkpoint = torch.load(path, map_location=torch.device('cpu')) # Safe load to CPU first
            # Check if it's a full checkpoint or just state_dict
            if 'model_state_dict' in checkpoint:
                rcnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                rcnn.load_state_dict(checkpoint)
            
            rcnn.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rcnn.to(device)
            print(f"✓ R-CNN loaded successfully: {path}")
            return rcnn, "rcnn"
        except Exception as e:
            print(f"✗ Error loading R-CNN: {e}")
            import traceback
            traceback.print_exc()
            return None, "error"

    # Handle YOLO (default)
    try:
        yolo = YOLO(path)
        print(f"✓ YOLO loaded successfully: {path}")
        return yolo, "yolo"
    except Exception as e:
        print(f"✗ Error loading YOLO: {e}")
        return None, "error"

def load_active_model():
    global active_model, active_model_type, ensemble
    
    # If already loaded and correct type, skip (simple check)
    # We rely on 'default_model_name' tracking the current intent
    
    print(f"Loading active model: {default_model_name}...")
    model_obj, m_type = load_model_by_name(default_model_name)
    
    if model_obj:
        active_model = model_obj
        active_model_type = m_type
        if m_type == 'ensemble':
            ensemble = model_obj # Keep reference for compatibility
        return True
    return False

def load_filter():
    global contourlet_filter
    if contourlet_filter is None:
        try:
            contourlet_filter = ContourletTransform(num_levels=2, num_directions=8)
            print("✓ Contourlet filter initialized")
        except Exception as e:
            print(f"⚠️  Error initializing filter: {e}")
            return False
    return True

def load_dental_condition_model():
    global dental_condition_model
    if dental_condition_model is None and Path(dental_condition_model_path).exists():
        try:
            print(f"Loading dental condition model from {dental_condition_model_path}...")
            dental_condition_model = YOLO(dental_condition_model_path)
            print("✓ Dental condition model loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Error loading dental condition model: {e}")
            return False
    elif not Path(dental_condition_model_path).exists():
        print("⚠️  Dental condition model not found yet (might still be training)")
        return False
    return True

def apply_preprocessing(img_array):
    """Apply Contourlet transform preprocessing"""
    if not use_filter:
        return img_array
    
    if contourlet_filter is None:
        return img_array
    
    try:
        filtered = contourlet_filter.apply(img_array)
        
        if filtered is None:
            print("⚠️  Contourlet filter returned None, using original image")
            return img_array
            
        return filtered
    except Exception as e:
        print(f"⚠️  Error applying filter: {e}")
        return img_array

def run_inference_on_image(model_obj, model_type, img_array, conf_threshold=0.25):
    """
    Unified inference function for single model.
    Returns list of dicts: {'bbox': [x1,y1,x2,y2], 'auth': float, 'class': str, 'class_id': int}
    """
    detections = []
    
    if model_type == 'ensemble':
        # Ensemble input: img_path or numpy array. MedicalEnsemble expects path or array?
        # Checking ensemble_detection.py: predict(self, image_path, ...) -> YOLO(p).predict(image_path)
        # Ultralytics YOLO supports numpy arrays.
        results_raw = model_obj.predict(img_array, conf=0.45) # Higher conf for ensemble usually
        for box in results_raw:
             detections.append({
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "confidence": float(box[4]),
                "class": class_names[int(box[5])] if int(box[5]) < len(class_names) else f"class_{int(box[5])}",
                "class_id": int(box[5])
            })
            
    elif model_type == 'rcnn':
        # R-CNN inference
        device = next(model_obj.parameters()).device
        
        # Preprocess for R-CNN (Tensor [C, H, W], 0-1 range)
        # img_array is likely [H, W, 3] BGR (from cv2) or RGB.
        # Ensure RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model_obj(img_tensor)
            
        # Parse outputs via NMS standard in R-CNN is usually done, but we filter by score
        output = outputs[0] # Batch size 1
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        
        for i in range(len(boxes)):
            if scores[i] >= conf_threshold:
                # Labels in R-CNN are 1-indexed (0 is background)
                # Map back to 0-indexed class_names
                cls_id = int(labels[i]) - 1
                if 0 <= cls_id < len(class_names):
                    cls_name = class_names[cls_id]
                else:
                    cls_name = f"Unknown_{cls_id}"
                
                detections.append({
                    "bbox": [float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])],
                    "confidence": float(scores[i]),
                    "class": cls_name,
                    "class_id": cls_id
                })

    elif model_type == 'yolo':
        # Standard YOLO
        results = model_obj.predict(img_array, conf=conf_threshold, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class": class_name,
                    "class_id": cls_id
                })
                
    return detections

@app.route('/health', methods=['GET'])
def health():
    model_status = "loaded" if active_model is not None else "not_loaded"
    filter_status = "active" if (use_filter and contourlet_filter is not None) else "disabled"
    
    health_data = {
        "status": "healthy",
        "model": {
            "status": model_status,
            "name": default_model_name,
            "type": active_model_type,
            "path": model_path
        },
        "filter": {
            "status": filter_status,
            "enabled": use_filter
        },
        "available_models": list(model_specs.keys())
    }
    
    if default_model_name in model_specs:
        health_data["model"]["info"] = model_specs[default_model_name]
    
    return jsonify(health_data)

@app.route('/models', methods=['GET'])
def list_models():
    return jsonify({
        "available_models": model_specs,
        "current_model": default_model_name
    })

@app.route('/models/<model_name>', methods=['POST'])
def switch_model(model_name):
    global active_model, active_model_type, model_path, use_filter, default_model_name
    
    if model_name not in model_specs:
        return jsonify({"error": f"Model '{model_name}' not found"}), 404
    
    new_model_path = model_specs[model_name].get('path')
    # Use generic check, path might be list for ensemble
    if model_name != 'ensemble' and new_model_path and not Path(new_model_path).exists():
        return jsonify({"error": f"Model file not found: {new_model_path}"}), 404
    
    try:
        # Update config tracking
        default_model_name = model_name
        model_path = new_model_path
        
        # Reload
        active_model = None
        if not load_active_model():
            return jsonify({"error": "Failed to load new model"}), 500
        
        if use_filter and not load_filter():
            print("⚠️  Filter initialization failed")
        
        return jsonify({
            "status": "success",
            "message": f"Switched to model: {model_name}",
            "model_info": model_specs[model_name]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if active_model is None:
            if not load_active_model():
                return jsonify({"error": "Model failed to load"}), 500
        
        if use_filter and not load_filter():
            pass # Already logged

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode Image (Shared Logic)
        try:
            base64_string = data['image']
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
        except Exception as e:
            return jsonify({"error": f"Image decode error: {str(e)}"}), 400

        # Convert to numpy
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

        # Preprocessing
        try:
            img_array = apply_preprocessing(img_array)
        except Exception as e:
            print(f"Pre-process error: {e}")

        # Run Inference (Object Detection)
        detections = run_inference_on_image(active_model, active_model_type, img_array)
        
        # Run Dental Condition Detection
        if dental_condition_model is None:
            load_dental_condition_model()
            
        if dental_condition_model is not None:
            try:
                # Run YOLO detection for dental conditions
                results = dental_condition_model.predict(img_array, conf=0.25, verbose=False)
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        
                        # Apply frequency-inferred mapping
                        class_name = dental_class_names[cls_id] if cls_id < len(dental_class_names) else f"cd_class_{cls_id}"
                        
                        detections.append({
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": conf,
                            "class": class_name,
                            "class_id": -1 # Using -1 to distinct from root canal class_ids if needed
                        })
            except Exception as e:
                print(f"Dental condition detection failed: {e}")
        
        # Grad-CAM (Only for YOLO currently)
        heatmap_base64 = None
        if data.get('enable_gradcam') and active_model_type == 'yolo':
            try:
                # Simplified GradCAM call
                # Note: This relies on global 'model' variable in original code if using YOLOGradCAM class directly
                # We need to adapt YOLOGradCAM to accept our 'active_model' instance if needed
                # For now, let's try to reuse the existing logic if compatible
                target_layer = active_model.model.model[-2]
                grad_cam = YOLOGradCAM(active_model, target_layer)
                
                img_resized = cv2.resize(img_array, (640, 640))
                input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                input_tensor = input_tensor.to(active_model.device)
                
                grayscale_cam = grad_cam(input_tensor, None)
                if grayscale_cam is not None:
                    # Resize heatmap to original image size
                    heatmap_resized = cv2.resize(grayscale_cam, (img_array.shape[1], img_array.shape[0]))
                    # Convert to colored heatmap (JET colormap) — send raw heatmap without X-ray overlay
                    # The frontend handles blending via CSS mix-blend-screen
                    heatmap_uint8 = (255 * heatmap_resized).astype(np.uint8)
                    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    _, buffer = cv2.imencode('.jpg', heatmap_colored)
                    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                print(f"GradCAM failed: {e}")

        return jsonify({
            "detections": detections,
            "image_size": {"width": img_array.shape[1], "height": img_array.shape[0]},
            "heatmap": heatmap_base64,
            "model_used": default_model_name
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/compare_all', methods=['POST'])
def compare_all():
    """
    Run the uploaded image against ALL available models and return combined results.
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode Image once
        base64_string = data['image']
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        
        # Prepare image array
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

        # Preprocessing (Common)
        load_filter()
        img_processed = apply_preprocessing(img_array)

        results_by_model = {}
        
        print("\n--- Starting Comparison Run ---")
        
        # Iterate through all models in config
        for name, spec in model_specs.items():
            print(f"Running comparison for: {name}")
            
            # Temporary load
            temp_model, temp_type = load_model_by_name(name)
            
            if temp_model:
                try:
                    dets = run_inference_on_image(temp_model, temp_type, img_processed)
                    results_by_model[name] = {
                        "detections": dets,
                        "details": spec
                    }
                except Exception as e:
                    print(f"Inference failed for {name}: {e}")
                    results_by_model[name] = {"error": str(e)}
                
                # Cleanup to save memory?
                # For PyTorch/YOLO models, 'del' might help, but garbage collector is lazy.
                # If we hit OOM, we might need explicit torch.cuda.empty_cache()
                del temp_model
                torch.cuda.empty_cache()
            else:
                 results_by_model[name] = {"error": "Failed to load model"}
        
        # Restore the originally active model to ensure state consistency
        load_active_model()
        
        return jsonify({
            "comparison_results": results_by_model,
            "image_size": {"width": img_array.shape[1], "height": img_array.shape[0]}
        })

    except Exception as e:
        print(f"Compare error: {e}")
        import traceback
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("Dental X-Ray Detection Server - Multi-Model Support")
    print("=" * 70)
    
    if load_active_model():
        print(f"✓ Active Model: {default_model_name} ({active_model_type})")
    else:
        print("⚠️  Initial model load failed")
    
    # Try to load dental condition model
    load_dental_condition_model()
    
    if use_filter and load_filter():
        print("✓ Preprocessing filter enabled")
        
    app.run(host='0.0.0.0', port=5000, debug=True)