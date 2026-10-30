# Google Colab: Hierarchical Distillation Guide

This guide describes how to run the Maximum Precision (Hierarchical) pipeline on Google Colab to utilize its GPU power.

---

## 1. Setup Your Google Drive
1.  **Mount Drive**: First, upload your project folder (`Root-canal-AI-diagnostics`) and your dataset folder (`Root Canal.v1i.yolov8`) to your Google Drive.
2.  **Organization**: I recommend putting both in a folder named `Root-Canal-AI`.

---

## 2. Open Colab & Connect to GPU
1.  Go to [Google Colab](https://colab.research.google.com).
2.  Click **Runtime > Change runtime type** and select **T4 GPU** (or better).

---

## 3. Environment Setup (Cell 1)
Copy and run this in your first cell:

```python
from google.colab import drive
import os

# 1. Mount Drive
drive.mount('/content/drive')

# 2. Set Project Path (Update this to your actual folder name in Drive)
PROJECT_PATH = '/content/drive/MyDrive/Root-Canal-AI/Root-canal-AI-diagnostics'
os.chdir(PROJECT_PATH)

# 3. Install Dependencies
!pip install -q ultralytics tqdm opencv-python numpy scipy
print(f"✓ Ready in {os.getcwd()}")
```

---

## 4. Fix Dataset Paths for Colab (Cell 2)
The dataset path inside the code must be absolute for Colab to find it reliably. Run this cell to update the `hierarchical_distillation.py` script with the Colab-specific path:

```python
import sys

# Define your absolute dataset path in Colab
COLAB_DATASET_PATH = '/content/drive/MyDrive/Root-Canal-AI/Root Canal.v1i.yolov8'
DATA_YAML = os.path.join(COLAB_DATASET_PATH, 'data.yaml')

# Update the hierarchical_distillation.py file to use the absolute path
with open('hierarchical_distillation.py', 'r') as f:
    lines = f.readlines()

with open('hierarchical_distillation.py', 'w') as f:
    for line in lines:
        if "pipeline = HierarchicalDistillation(data_yaml=" in line:
            f.write(f"    pipeline = HierarchicalDistillation(data_yaml='{DATA_YAML}')\n")
        else:
            f.write(line)

print(f"✓ hierarchical_distillation.py updated to use: {DATA_YAML}")
```

---

## 5. Pre-process Images (Cell 3)
Since you've already processed them locally, you skip this, but if your Drive images are raw, run:

```python
!python filter_fusion.py "{COLAB_DATASET_PATH}"
```

---

## 6. Run the Full Model Ladder (Cell 4)
This will run the 3-stage training: Teacher (v8m) -> Intermediate (v8n) -> Final (v5n).

```python
# This will take 3-5 hours on Colab's T4 GPU
!python hierarchical_distillation.py
```

---

## 7. Retrieve Results
Your results will be saved in your Google Drive under:
`Root-canal-AI-diagnostics/hierarchical_results/`

Inside, you will find:
1. `1_teacher_v8m/`
2. `2_intermediate_v8n/`
3. `3_final_v5n/` (This is your highest precision model)
