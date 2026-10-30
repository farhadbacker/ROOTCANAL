# Google Colab: Phase 4 - Grounded SAM Labeling

This guide describes how to run **Phase 4** to generate high-fidelity "Super-Teacher" labels using Grounding DINO and SAM.

---

## 1. Project Setup
Run this cell first to connect to your Drive and navigate to your project folder. This ensures the script `gsam_labeling.py` is found:

```python
from google.colab import drive
import os

# 1. Mount Drive
drive.mount('/content/drive')

# 2. Set Project Path (Update this to your actual folder name in Drive)
PROJECT_PATH = '/content/drive/MyDrive/Root-Canal-AI/Root-canal-AI-diagnostics'
os.chdir(PROJECT_PATH)

print(f"✓ gsam_labeling.py found: {os.path.exists('gsam_labeling.py')}")
print(f"✓ gsam_box_refinement.py found: {os.path.exists('gsam_box_refinement.py')}")

# Verification: Check if your latest code changes reflected in Colab
if os.path.exists('gsam_labeling.py'):
    with open('gsam_labeling.py', 'r') as f:
        content = f.read()
        if "AutoModelForZeroShotObjectDetection" in content:
            print("⭐ SUCCESS: Your code is synced and ready!")
        else:
            print("⚠️ WARNING: Colab is still seeing the OLD version. Please wait 1 minute and re-run this cell.")
```

## 2. Setup Environment (Foolproof)
Configure both the labeling (GSAM) and training (YOLO) dependencies in one step:

```python
# Install Labeling + Training libraries
!pip install -q transformers accelerate supervision ultralytics tqdm opencv-python numpy scipy
print("✓ Environment set up successfully.")
```

## 2. Generate High-Fidelity Labels
Run the labeling script using the text prompts we've optimized for dental radiographs:

```python
from gsam_labeling import GroundedSAMLabeler

# Initialize with dental prompts
labeler = GroundedSAMLabeler()

# Path to your fused dataset images
images_dir = '/content/drive/MyDrive/Root-Canal-AI/Root Canal.v1i.yolov8/train/images'
output_dir = '/content/drive/MyDrive/Root-Canal-AI/super_labels'

labeler.process_dataset(images_dir, output_dir)
```

## 3. Refine and Merge Labels
Once the super-labels are generated, run the refinement tool to merge them with your existing dataset:

```python
from gsam_box_refinement import refine_yolo_labels

# Paths to merge
original_labels = '/content/drive/MyDrive/Root-Canal-AI/Root Canal.v1i.yolov8/train/labels'
gsam_labels = '/content/drive/MyDrive/Root-Canal-AI/super_labels'
refined_output = '/content/drive/MyDrive/Root-Canal-AI/refined_labels'

refine_yolo_labels(original_labels, gsam_labels, refined_output)
```

## 4. Final Training
After refinement, update your `data.yaml` to point to the `refined_labels` directory and restart the Hierarchical Distillation:

```python
!python hierarchical_distillation.py
```
*Note: This will now train the Teacher (YOLOv8m) on near-perfect labels, which will trickle down to the 99% precision goal in the student models.*
