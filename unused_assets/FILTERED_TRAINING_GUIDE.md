# Training YOLO with Contourlet-Filtered Images

## Quick Start

Run this batch file to train with filtered images:
```bash
train_with_filtered.bat
```

This will:
1. Backup your original images
2. Apply Contourlet filtering to dataset/images
3. Train YOLO model with filtered images on CPU

## Manual Steps

### Step 1: Preprocess Dataset with Contourlet Filtering

```bash
python preprocess_dataset.py --backup --levels 2 --directions 8
```

**What this does:**
- Creates backup of original images in `dataset/images_original/`
- Applies Contourlet Transform to all images in `dataset/images/`
- Replaces original images with filtered versions
- Preserves YOLO labels automatically

**Options:**
- `--backup`: Create backup before filtering (recommended)
- `--levels 2`: Number of pyramid levels (default: 2)
- `--directions 8`: Number of directional filters (default: 8)

### Step 2: Train YOLO Model

```bash
python train_yolo.py --filtered --epochs 100 --batch-size 16
```

**What this does:**
- Loads YOLOv8 nano model (yolov8n.pt)
- Detects CUDA availability (auto-switches to CPU if not available)
- Trains for 100 epochs on filtered images
- Saves model as `dental_yolo.pt`
- Exports to ONNX format

**Command-line Options:**
```
--filtered          Use filtered images (shows in output)
--epochs N          Number of training epochs (default: 100)
--batch-size N      Batch size (default: 16, may need to reduce for CPU)
--device cpu        Force CPU (auto-detected if not CUDA available)
--model PATH        Path to pretrained model (default: yolov8n.pt)
--output PATH       Output model filename (default: dental_yolo.pt)
--patience N        Early stopping patience (default: 20)
--resume            Resume from checkpoint
```

## Dataset Structure

Before training, ensure:
```
dataset/
├── images/          # Your dental X-ray images (will be filtered)
├── images_original/ # Backup of originals (created by preprocess)
└── labels/          # YOLO format label files
    ├── image1.txt
    ├── image2.txt
    └── ...
```

Label format (YOLO):
```
<class_id> <x_center> <y_center> <width> <height>
```

Classes:
- 0: No Endodontic Treatment
- 1: Incomplete Endodontic Treatment
- 2: Complete Endodontic Treatment
- 3: Total Endodontic Failure

## Hardware Considerations

### Training on CPU
If you don't have CUDA:
- Training will be slower
- Reduce batch size if running out of memory:
  ```bash
  python train_yolo.py --filtered --batch-size 8
  ```
- Consider reducing epochs or image size:
  ```bash
  python train_yolo.py --filtered --epochs 50 --imgsz 416
  ```

### Monitoring Training
The script will show:
- Loss metrics
- mAP (mean Average Precision)
- Training progress
- Time per epoch

## Troubleshooting

### Issue: Low GPU/CPU memory
**Solution:** Reduce batch size
```bash
python train_yolo.py --filtered --batch-size 4
```

### Issue: Preprocessing takes too long
**Solution:** Reduce contourlet levels
```bash
python preprocess_dataset.py --levels 1 --directions 4
```

### Issue: Filtered images look wrong
**Solution:** Adjust contourlet parameters:
```bash
python preprocess_dataset.py --levels 3 --directions 16
```

### Issue: Want to use original images again
**Solution:** Restore from backup
```bash
python -c "
import shutil
from pathlib import Path
shutil.rmtree('dataset/images')
shutil.copytree('dataset/images_original', 'dataset/images')
print('Restored original images')
"
```

## Training Output

Model outputs:
- `dental_yolo.pt` - PyTorch model
- `dental_yolo.onnx` - ONNX format (for web/inference)
- `runs/detect/train/` - Training results and metrics

## Using Trained Model

After training:
```bash
npm run dev
python inference_server.py
```

The inference server will automatically use your trained `dental_yolo.pt` model.

## Paper Reference

This training follows the paper:
"Fusion of Image Filtering and Knowledge-Distilled YOLO Models for Root Canal Failure Diagnosis"

The Contourlet Transform captures directional information useful for:
- Better edge detection
- Improved feature extraction
- Enhanced sensitivity to root canal structures
