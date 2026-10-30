# Step-by-Step Guide to Run Knowledge Distillation in Google Colab

This guide provides a detailed process to run your custom knowledge distillation pipeline on Google Colab using the free GPU resources.

---

## **Prerequisites**

1.  A Google Account.
2.  Your dataset folder `Root Canal.v1i.yolov8` (containing `data.yaml`, `train`, `val`, `test`).
3.  The project code files (specifically `run_distillation_pipeline.py`, `knowledge_distillation.py`, etc.).

---

## **Step 1: Prepare Your Google Drive**

1.  Go to [Google Drive](https://drive.google.com).
2.  Create a new folder named `Root-Canal-AI`.
3.  **Upload your Dataset**: Upload the entire `Root Canal.v1i.yolov8` folder into this new `Root-Canal-AI` folder.
    *   *Note: If the folder is large, zip it first, upload, and we can unzip in Colab.*
4.  **Upload Code**: Upload your project python files (or the whole project folder) to `Root-Canal-AI` as well.
    *   Essential files to upload: `run_distillation_pipeline.py`, `knowledge_distillation.py`, `requirements.txt`.

---

## **Step 2: Start Google Colab**

1.  Go to [Google Colab](https://colab.research.google.com).
2.  Click **"New Notebook"**.
3.  **Enable GPU**:
    *   Go to menu: **Runtime** > **Change runtime type**.
    *   Select **T4 GPU** (or any available GPU) under "Hardware accelerator".
    *   Click **Save**.

---

## **Step 3: Setup the Environment (Copy-Paste these Cells)**

Copy the following code blocks into separate cells in your Colab notebook and run them in order by pressing the **Run** button (or Shift+Enter).

### **Cell 1: Connect to Drive & Setup**

```python
import os
import sys
from google.colab import drive

# 1. Mount Google Drive
drive.mount('/content/drive')

# 2. Setup paths
# TIP: To find your path:
# 1. Click the 'Folder' icon on the left sidebar in Colab
# 2. Navigate to drive -> MyDrive -> Root-Canal-AI -> Root-canal-AI-diagnostics
# 3. Right-click 'Root-canal-AI-diagnostics' and select 'Copy path'
# 4. Paste it as PROJECT_PATH
PROJECT_PATH = '/content/drive/MyDrive/Root-Canal-AI/Root-canal-AI-diagnostics' 

# Dataset is likely one folder up
DATASET_PATH = '/content/drive/MyDrive/Root-Canal-AI/Root Canal.v1i.yolov8'

# 3. Verify files exist
if os.path.exists(PROJECT_PATH):
    print(f"✅ Project found at: {PROJECT_PATH}")
    # CRITICAL: This is what 'Navigating' means in code
    os.chdir(PROJECT_PATH) # Change working directory to your project
    print(f"📂 Current Directory: {os.getcwd()}")
else:
    print(f"❌ Could not find path: {PROJECT_PATH}. Please check your Drive folder name.")

if os.path.exists(DATASET_PATH):
    print(f"✅ Dataset found at: {DATASET_PATH}")
else:
    print(f"❌ Dataset not found at: {DATASET_PATH}")
```

### **Cell 2: Install Dependencies**

```python
!pip install -q ultralytics torch torchvision opencv-python numpy pillow matplotlib scikit-learn seaborn tqdm flask flask-cors pandas
print("✅ Dependencies Installed!")
```

### **Cell 3: Verify GPU is Working**

```python
!nvidia-smi
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

---

## **Step 4: Run the Distillation Pipeline**

Now for the main event. You can run the entire pipeline (Teacher -> Student -> Distillation -> Comparison) with one command.

### **Cell 4: Run Training**

```python
# Run the full pipeline
# We use f-string formatting to insert the path correctly
# NOTE: Ensure there is a SPACE before --data

!python run_distillation_pipeline.py \
    --stage all \
    --teacher-epochs 50 \
    --student-epochs 50 \
    --batch-size 16 \
    --data "{DATASET_PATH}/data.yaml"
```

> **Note:** If you face path issues with `data.yaml`, you might need to edit `data.yaml` inside your text editor before uploading to ensure the paths inside it are relative or correct for Colab. Often, YOLO requires absolute paths in Colab.
>
> **Quick Fix for `data.yaml` in Colab:**
> You can create a temporary `data.yaml` in Colab with the correct paths using this cell:
```python
import yaml

# Create a colab specific data.yaml
data_config = {
    'path': DATASET_PATH,
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'names': {0: 'Root Canal Failure'} # Verify your class names match!
}

with open('colab_data.yaml', 'w') as f:
    yaml.dump(data_config, f)

print("Created colab_data.yaml")

# Run using this new config
!python run_distillation_pipeline.py --stage all --teacher-epochs 50 --student-epochs 50 --data colab_data.yaml
```

---

## **Step 5: View and Save Results**

The script saves results in `distillation_results/`. Since we are running this directly in your Google Drive folder (`/content/drive/MyDrive/Root-Canal-AI`), **your results are automatically saved to your Drive!** you don't need to manually download them.

### **Cell 5: Check Results**

```python
# List the results
!ls -R distillation_results/

# Print the comparison summary log
!cat distillation_results/training_log.txt
```

---

## **Troubleshooting**

*   **"File not found"**: Double-check your Google Drive folder names. They are case-sensitive.
*   **"CUDA out of memory"**: Reduce the `--batch-size` in the command (e.g., `--batch-size 8`).
*   **"Dataset not found"**: Ensure `data.yaml` points to the correct images. The "Quick Fix" in Step 4 usually solves this.
