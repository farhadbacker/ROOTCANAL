# Quick Start: Training YOLO with Contourlet Filter

## Prerequisites
```bash
pip install ultralytics opencv-python scipy pillow flask flask-cors
```

## Quick Steps

### 1️⃣ Preprocess Images (Apply Contourlet Filter)
```bash
python preprocess_dataset.py --backup
```
- `--backup`: Saves originals in `dataset/images_original/` (recommended)
- Applies Contourlet transform to all images

### 2️⃣ Train YOLO Model
```bash
python train_yolo.py --filtered --epochs 100
```
- `--filtered`: Uses filtered images (must preprocess first)
- `--epochs 100`: Number of training epochs
- Output: `dental_yolo.pt` (trained model)

### 3️⃣ Run Inference Server
```bash
python inference_server.py
```
- Server: http://localhost:5000
- Filter automatically applied to incoming images

### 4️⃣ Start Frontend (new terminal)
```bash
npm run dev
```
- Frontend: http://localhost:3002

## Training Options

**Use raw images (no filter):**
```bash
python train_yolo.py --epochs 100
```

**Use filtered images:**
```bash
python train_yolo.py --filtered --epochs 100
```

**Advanced options:**
```bash
python train_yolo.py --filtered \
  --epochs 150 \
  --batch-size 32 \
  --model yolov8m.pt \
  --output my_model.pt
```

## Key Files

| File | Purpose |
|------|---------|
| `contourlet_filter.py` | Contourlet Transform implementation |
| `preprocess_dataset.py` | Batch apply filter to dataset |
| `train_yolo.py` | Train YOLO with/without filter |
| `inference_server.py` | API server with filter support |
| `CONTOURLET_TRAINING_GUIDE.md` | Detailed documentation |

## What is Contourlet Filter?

Contourlet Transform captures **curvilinear structures** in images through:
- Laplacian pyramid decomposition
- Directional filter banks (Gabor-like filters)
- Edge enhancement

**Benefits for dental X-rays:**
- ✓ Better edge detection
- ✓ Enhanced anatomical structures  
- ✓ Improved detection accuracy
- ✓ Handles curved anatomy better than wavelets

## Results

Training with Contourlet filter typically shows:
- Better object localization
- Higher precision in small object detection
- Improved robustness to image variations

## Troubleshooting

**Question: Will preprocessing overwrite my original images?**
- Answer: Use `--backup` flag to save originals in `dataset/images_original/`

**Question: How long does preprocessing take?**
- Answer: ~1-2 minutes for ~200 images on modern CPU. 
- Can reduce time by using `--directions 4` (fewer filters)

**Question: Can I switch between filtered and raw training?**
- Answer: Yes! Restore originals with:
  ```bash
  cp dataset/images_original/* dataset/images/
  ```

**Question: Does inference need the same filter?**
- Answer: Yes! `inference_server.py` applies filter automatically.
- Filter settings must match training (levels=2, directions=8)

## Next Steps

1. See `CONTOURLET_TRAINING_GUIDE.md` for detailed documentation
2. Monitor training metrics in `runs/detect/train/`
3. Compare performance: filtered vs raw training
4. Deploy best performing model to production

---

**Performance Tips:**
- Start with GPU to speed up training
- Monitor GPU memory: use smaller batch size if needed
- Check `runs/detect/train/results.csv` for metrics
