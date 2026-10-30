# Google Colab Quick Start Guide

Run your Knowledge Distillation Pipeline on **FREE GPU** in minutes!

⚠️ **IMPORTANT:** The GitHub repo structure may differ from your local files. **Upload files manually** to ensure everything works correctly.

## Step 1: Open Google Colab
Go to: https://colab.research.google.com

## Step 2: Create New Notebook
- Click **File** → **New notebook**
- Name it: "Root Canal Training"

## Step 3: Copy-Paste Cells

Paste each code block below as a **separate cell** in Colab (press `Ctrl+M` to add new cell).

---

### **Cell 1: Setup Environment & GPU**
```python
# Check GPU availability
!nvidia-smi

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create working directory
!mkdir -p /content/root-canal
%cd /content/root-canal

print("✓ Google Colab environment ready")
print("✓ GPU available for training")
```

**Expected Output:** Shows your GPU details (NVIDIA Tesla T4, V100, or A100)

---

### **Cell 2: Install Dependencies**
```python
!pip install -q ultralytics torch torchvision opencv-python numpy pillow matplotlib scikit-learn seaborn tqdm flask pandas

print("✓ Dependencies installed")
```

---

### **Cell 3: Upload Your Project**

**Option A: Upload All Files Manually** (RECOMMENDED)
```python
# 1. Click Files icon (📁) on left sidebar
# 2. Click "Upload to session"
# 3. Select all files from: c:\Users\PRO\Desktop\Root Canal\Root-canal-AI-diagnostics\
# 4. Wait for upload to complete
# 5. Verify:
!ls -la
```

**Option B: Clone from GitHub** (if you pushed to GitHub)
```python
!git clone https://github.com/YOUR-USERNAME/Root-Canal-AI.git .
# After cloning, navigate to the correct folder if needed
!ls -la
```

**Note:** Make sure these files are present:
- `run_distillation_pipeline.py` (or `FULL_PIPELINE_FILTERED.py`)
- `knowledge_distillation.py`
- `evaluate_models.py`
- `comparative_analysis.py`

---

### **Cell 4: Verify Dataset**
```python
import os

# Check if dataset is in Google Drive
dataset_path = '/content/drive/MyDrive/Root Canal.v1i.yolov8'

if os.path.exists(dataset_path):
    print(f"✓ Dataset found: {dataset_path}")
    # Copy to local for faster training
    !cp -r '/content/drive/MyDrive/Root Canal.v1i.yolov8' /content/root-canal/
else:
    print("⚠ UPLOAD DATASET TO GOOGLE DRIVE FIRST!")
    print("Steps:")
    print("1. Click Files icon (📁) on left")
    print("2. Click 'Mounts' → 'Drive'")
    print("3. Upload 'Root Canal.v1i.yolov8' folder")
    print("4. Come back and run this cell")
```

---

### **Cell 5: Run Training Pipeline**

**OPTION 1: Full Knowledge Distillation Pipeline**

Quick Training (1 hour)
```python
!python run_distillation_pipeline.py --stage all --teacher-epochs 30 --student-epochs 30
```

Full Training (3-6 hours)
```python
!python run_distillation_pipeline.py --stage all --teacher-epochs 100 --student-epochs 100
```

Run Individual Stages
```python
# Teacher training only
!python run_distillation_pipeline.py --stage teacher --teacher-epochs 30

# Student training only  
!python run_distillation_pipeline.py --stage student --student-epochs 30

# Evaluation only
!python run_distillation_pipeline.py --stage evaluate

# Comparative analysis
!python run_distillation_pipeline.py --stage analyze
```

---

**OPTION 2: Simpler Filtered Pipeline** (if distillation script has issues)

Preprocessing + Training with Contourlet Filter (2-3 hours)
```python
!python FULL_PIPELINE_FILTERED.py
```

This is simpler and includes:
- ✓ Contourlet image filtering
- ✓ Automatic YAML generation
- ✓ Model training (100 epochs)
- ✓ Saves trained model

---

### **Cell 6: Monitor Training** (run during training)
```python
# Monitor GPU memory
!nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv --loop=1 --loop-ms=5000
```

Press `Interrupt Execution` to stop

---

### **Cell 7: Save Results to Google Drive**
```python
import shutil
from pathlib import Path

drive_results = '/content/drive/MyDrive/Root-Canal-Results'
os.makedirs(drive_results, exist_ok=True)

# Copy all results
for folder in ['distillation_results', 'evaluation_results', 'comparative_results']:
    if os.path.exists(folder):
        shutil.copytree(folder, f'{drive_results}/{folder}', dirs_exist_ok=True)
        print(f"✓ Saved {folder}")

# Also copy logs
shutil.copy('distillation_pipeline.log', f'{drive_results}/')

print(f"\n✓ Results saved to Google Drive: {drive_results}")
print("Download or view in Drive online")
```

---

### **Cell 8: Download Results Locally** (Optional)
```python
# This creates a zip file to download
!cd /content/root-canal && zip -r results.zip distillation_results evaluation_results comparative_results

# In Colab sidebar, click Files (📁) and download results.zip
```

---

## Timing Guide

| Stage | Epochs | Time (GPU) |
|-------|--------|-----------|
| Teacher Training | 30 | 15-20 min |
| Student Training | 30 | 15-20 min |
| Evaluation | - | 2-3 min |
| Analysis | - | 2-3 min |
| **Total** | **30/30** | **~40-50 min** |

---

## Troubleshooting

### **"Script not found" or "ModuleNotFoundError"**
```python
# Check what files are in current directory
!ls -la

# If files are missing, you didn't upload them correctly
# Go back to Cell 3 and upload all .py files again
```

**Solution:**
1. Click Files (📁) on left
2. Click Upload
3. Select ALL `.py` files from your local folder
4. Wait for complete upload (check file list)
5. Run: `!ls -la` to verify

### **"Dataset not found"**
```python
# Check if dataset exists
!ls -la '/content/drive/MyDrive/'
```

- Upload `Root Canal.v1i.yolov8` to Google Drive
- Path: `/content/drive/MyDrive/Root Canal.v1i.yolov8`

### **"CUDA out of memory"**
```python
# Use fewer epochs
!python run_distillation_pipeline.py --stage all --teacher-epochs 20 --student-epochs 20
```

### **"GPU not available"**
- Runtime → Change runtime type → **GPU**
- Wait 30 seconds and re-run `!nvidia-smi`

### **Training stopped/disconnected**
- Normal after 12 hours (free tier limit)
- Results saved to Drive
- Can resume in new session

---

## Cost & Limits

✅ **Completely FREE**
- No credit card needed
- NVIDIA GPU (12+ hours/session)
- 100GB Google Drive storage

---

## Next Steps

After training:
1. Review metrics in `evaluation_results/`
2. Check visualizations in `comparative_results/`
3. Optionally run inference: `python inference_server.py`
4. Download results for your presentation

---

**Questions?** Run the full pipeline with: 
```python
!python run_distillation_pipeline.py --stage all --teacher-epochs 50 --student-epochs 50
```
