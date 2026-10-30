# Kaggle Training Quick Start Guide

Run your Knowledge Distillation Pipeline on **Kaggle Kernels (Free GPU)**.

## Step 1: Create New Notebook
1. Go to [Kaggle Kernels](https://www.kaggle.com/code)
2. Click **New Notebook**
3. Select **Python** as language

## Step 2: Setup Accelerator (GPU)
1. In the notebook sidebar (right side), look for **Session Options**
2. Set **Accelerator** to **GPU P100** (or T4 x2)
3. Set **Internet** to **On** (needed to install dependencies)

## Step 3: Add Your Dataset
1. In the sidebar, click **Add Input**
2. Click **Upload** -> **New Dataset**
3. Create a title (e.g., "Root Canal Dataset")
4. Upload your `Root Canal.v1i.yolov8` folder (zip it first if needed, Kaggle handles zips well)
5. Click **Create**
6. Your dataset will appear at `/kaggle/input/root-canal-dataset/` (actual path may vary based on your title)

## Step 4: Python Code Setup

Paste the following blocks into separate cells in your Kaggle notebook.

### **Cell 1: Install Dependencies**
```python
# Install libraries not pre-installed in Kaggle
!pip install -q ultralytics supervision
# Standard libs like numpy, pandas, torch, cv2 are usually pre-installed on Kaggle
```

### **Cell 2: Setup Workspace & Copy Files**
*Since you cannot "mount" a local drive like Colab, the easiest way is to clone your code from GitHub or copy-paste the script contents. Here we assume you copy-pasted the script contents or cloned a repo.*

**Option A: Clone from GitHub (Recommended)**
```python
import os
os.chdir('/kaggle/working')

# Clone your repo
!git clone https://github.com/YOUR-USERNAME/Root-Canal-AI.git .

# Verify files
!ls -la
```

**Option B: Manually Create Scripts**
*If you don't use GitHub, you can create the python files directly in the notebook using the `%%writefile` magic command.*

```python
%%writefile run_distillation_pipeline.py
# Paste the ENTIRE content of run_distillation_pipeline.py here
```
*(Repeat for `knowledge_distillation.py`, `evaluate_models.py`, etc.)*

### **Cell 3: Link Dataset**
Kaggle datasets are read-only input. We should point the script to the input directory.

```python
import os
from ultralytics import YOLO

# Define path to the dataset in input directory
# NOTE: Update 'root-canal-dataset' to matches your dataset name in the right sidebar
dataset_input_dir = '/kaggle/input/root-canal-dataset/Root Canal.v1i.yolov8'

# Verify it exists
if os.path.exists(dataset_input_dir):
    print(f"Dataset found at: {dataset_input_dir}")
else:
    print("Dataset not found! Check the path in the sidebar.")
```

### **Cell 4: Run Training Pipeline**

```python
# Run the pipeline
# We use --stage all to run everything
# Adjust epochs as needed for time constraints

!python run_distillation_pipeline.py --stage all --teacher-epochs 50 --student-epochs 50
```

### **Cell 5: Save & Download Results**
Files in `/kaggle/working` are temporary but can be downloaded after the session if you commit the notebook, or you can zip them up to download immediately.

```python
import shutil

# Zip the results
!zip -r training_results.zip distillation_results evaluation_results comparative_results

print("Results zipped! Download 'training_results.zip' from the Output section in the sidebar.")
```

## Tips for Kaggle
*   **Persistent Storage**: Kaggle sessions are ephemeral. Always download your weights (`best.pt`) or commit the notebook to save outputs.
*   **Time Limit**: Standard sessions are ~9 hours.
*   **GPU Quota**: 30 hours/week.
