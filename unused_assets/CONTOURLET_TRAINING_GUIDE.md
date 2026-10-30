# YOLO Training with Contourlet Transform Filter

This guide explains how to train YOLO with **Contourlet Transform** filter applied to images, as mentioned in the paper: "Fusion of Image Filtering and Knowledge-Distilled YOLO Models for Root Canal Failure Diagnosis".

## Overview

Contourlet Transform is a directional image decomposition method that captures curvilinear structures in images better than wavelet transforms. This makes it excellent for detecting subtle anatomical structures in dental X-rays.

**Key Benefits:**
- ✓ Enhanced edge detection for better object localization
- ✓ Improved contrast for diagnostic features
- ✓ Better handling of curved anatomical structures
- ✓ Superior object detection performance

## Setup

### 1. **Verify Dependencies**

Make sure you have the required packages installed:
```bash
pip install ultralytics opencv-python scipy pillow flask flask-cors numpy
```

Or use the requirements.txt:
```bash
pip install -r requirements.txt
```

### 2. **Prepare Your Dataset**

Ensure your dataset structure is:
```
dataset/
├── images/          # Original X-ray images
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── labels/          # YOLO format annotations
    ├── 1.txt
    ├── 2.txt
    └── ...
```

## Training with Contourlet Filter

### Step 1: Preprocess Dataset (Apply Contourlet Filter)

The first step is to apply the Contourlet Transform filter to all training images:

```bash
python preprocess_dataset.py --backup
```

**Options:**
- `--backup` - Create a backup of original images in `dataset/images_original/` (recommended)
- `--levels` - Number of pyramid levels (default: 2)
- `--directions` - Number of directional filters (default: 8)
- `--no-replace` - Save filtered images to separate directory instead of replacing originals

**Examples:**

**With Backup (Recommended):**
```bash
python preprocess_dataset.py --backup
```
This will:
- ✓ Backup original images to `dataset/images_original/`
- ✓ Apply Contourlet filter to images in `dataset/images/`
- ✓ Replace original images with filtered versions

**Without Backup:**
```bash
python preprocess_dataset.py
```

**To Separate Directory:**
```bash
python preprocess_dataset.py --no-replace
```

**With Custom Parameters:**
```bash
python preprocess_dataset.py --backup --levels 3 --directions 16
```

### Step 2: Train YOLO Model

After preprocessing, train your YOLO model:

```bash
python train_yolo.py --filtered
```

**Options:**
- `--filtered` - Use Contourlet-filtered images (required if you preprocessed)
- `--epochs` - Number of training epochs (default: 100)
- `--batch-size` - Batch size (default: 16)
- `--imgsz` - Image size (default: 640)
- `--model` - Pretrained model to use (default: yolov8n.pt)
- `--output` - Output model filename (default: dental_yolo.pt)
- `--device` - GPU device index (default: 0)
- `--patience` - Early stopping patience (default: 20)
- `--resume` - Resume training from checkpoint

**Examples:**

**Basic Training with Filtered Images:**
```bash
python train_yolo.py --filtered
```

**With Custom Epochs and Batch Size:**
```bash
python train_yolo.py --filtered --epochs 150 --batch-size 32
```

**Using Larger YOLO Model:**
```bash
python train_yolo.py --filtered --model yolov8m.pt --epochs 150
```

**With Custom Output Name:**
```bash
python train_yolo.py --filtered --output dental_yolo_filtered.pt
```

**Resume Training:**
```bash
python train_yolo.py --filtered --resume
```

## Inference with Contourlet Filter

The inference server automatically applies the Contourlet filter to incoming images during detection:

### Start the Inference Server

```bash
python inference_server.py
```

The server will:
- ✓ Load the trained YOLO model
- ✓ Initialize Contourlet filter
- ✓ Apply filter to each incoming image
- ✓ Run YOLO inference on filtered image

### API Endpoints

**Health Check:**
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "model": "loaded",
  "filter": "active"
}
```

**Run Detection:**
```bash
curl -X POST http://localhost:5000/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,..."}'
```

## Complete Workflow Example

### 1. **Backup and Preprocess Dataset**
```bash
python preprocess_dataset.py --backup --levels 2 --directions 8
```

### 2. **Train Model**
```bash
python train_yolo.py --filtered --epochs 100 --batch-size 16
```

### 3. **Start Frontend**
```bash
npm run dev
```

### 4. **Start Inference Server** (in another terminal)
```bash
python inference_server.py
```

### 5. **Access Application**
Open http://localhost:3002 in your browser

## Switching Between Raw and Filtered Training

### Use Raw Images:
```bash
python train_yolo.py
```

### Use Filtered Images:
```bash
python train_yolo.py --filtered
```

### Restore Original Images (if backed up):
```bash
cp dataset/images_original/* dataset/images/
```

## Comparing Performance

To compare performance between raw and filtered training:

1. **Train with raw images and save as `dental_yolo_raw.pt`:**
   ```bash
   python train_yolo.py --output dental_yolo_raw.pt --epochs 100
   ```

2. **Apply filter and train as `dental_yolo_filtered.pt`:**
   ```bash
   python preprocess_dataset.py --backup
   python train_yolo.py --filtered --output dental_yolo_filtered.pt --epochs 100
   ```

3. **Compare results in the training output**

## Inference Configuration

### Enable/Disable Filter During Inference

Edit `inference_server.py` line 20:

**To enable filter:**
```python
use_filter = True
```

**To disable filter:**
```python
use_filter = False
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'contourlet_filter'"
**Solution:** Make sure `contourlet_filter.py` is in the same directory as your training scripts

### Issue: Memory error during preprocessing
**Solution:** Process images in batches or use fewer directional filters:
```bash
python preprocess_dataset.py --directions 4
```

### Issue: Training is slow with filtered images
**Solution:** This is normal! Filtered images contain more information. You can:
- Reduce batch size
- Use fewer pyramid levels
- Use yolov8n.pt (nano model) instead of larger models

### Issue: Filter not applied during inference
**Solution:** Check that `use_filter = True` in `inference_server.py` and the filter initialized successfully (check server logs)

## File Structure

```
Root-canal-AI-diagnostics/
├── contourlet_filter.py          # Contourlet transform implementation
├── preprocess_dataset.py         # Preprocessing script
├── train_yolo.py                 # Training script
├── inference_server.py           # Inference server with filter
├── CONTOURLET_TRAINING_GUIDE.md  # This file
├── dataset/
│   ├── images/                   # Filtered images (after preprocessing)
│   ├── images_original/          # Backup of original images
│   └── labels/                   # YOLO annotations
├── runs/                         # Training output (metrics, weights)
└── dental_yolo.pt               # Trained model
```

## Key Parameters Explained

### Contourlet Transform Parameters:
- **num_levels** (default: 2): Pyramid decomposition levels
  - Higher = more detailed analysis but slower
  - Recommended: 2-3 for dental X-rays

- **num_directions** (default: 8): Directional filters
  - Higher = better directional sensitivity but slower
  - Recommended: 8 for dental X-rays

### YOLO Training Parameters:
- **epochs**: Higher = better accuracy but longer training
- **batch_size**: Higher = faster but needs more GPU memory
- **patience**: Early stopping if validation loss doesn't improve
- **imgsz**: Larger = more detail but slower training

## Performance Tips

1. **Start with filtered preprocessing:**
   ```bash
   python preprocess_dataset.py --backup
   ```

2. **Use appropriate batch size for your GPU:**
   - GPU with 6GB VRAM: batch_size=8-16
   - GPU with 12GB VRAM: batch_size=16-32
   - GPU with 24GB VRAM: batch_size=32-64

3. **Monitor training loss** - If loss increases, reduce learning rate

4. **Use validation set** - Split dataset for better evaluation

## References

- Paper: "Fusion of Image Filtering and Knowledge-Distilled YOLO Models for Root Canal Failure Diagnosis"
- Contourlet Transform provides superior edge and directional feature extraction
- Combined with YOLO's object detection capabilities for accurate diagnostic detection

## Next Steps

After training:
1. ✓ Deploy model to production
2. ✓ Monitor detection accuracy
3. ✓ Collect feedback for model improvement
4. ✓ Consider fine-tuning with additional data

