#!/usr/bin/env python3
"""
Google Colab-Compatible Knowledge Distillation Pipeline
========================================================

Run this in Google Colab for faster training with free GPU access.

Setup Instructions:
1. Open Google Colab: https://colab.research.google.com
2. Create a new notebook
3. Upload this file to Colab
4. Run each cell in order

Or paste each section as a separate cell in Colab.
"""

import os
import sys
from pathlib import Path

def setup_colab():
    """Setup Google Colab environment"""
    try:
        import google.colab
        from google.colab import drive
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    
    return IN_COLAB

def mount_google_drive():
    """Mount Google Drive to access/save files"""
    from google.colab import drive
    drive.mount('/content/drive')
    print("✓ Google Drive mounted at /content/drive")
    print("\nNext steps:")
    print("1. Upload your 'Root Canal.v1i.yolov8' dataset to Google Drive")
    print("2. Or update the dataset path in the script")

def install_dependencies():
    """Install required packages"""
    print("Installing dependencies...")
    os.system('pip install -q ultralytics torch torchvision opencv-python numpy pillow matplotlib scikit-learn seaborn tqdm flask flask-cors pandas')
    print("✓ Dependencies installed")

def setup_project_paths():
    """Setup paths for Colab environment"""
    # Option 1: Using Google Drive
    drive_path = Path('/content/drive/MyDrive/Root-Canal-AI')
    
    # Option 2: Using workspace
    workspace_path = Path('/content/Root-Canal-AI')
    
    # Create workspace if doesn't exist
    workspace_path.mkdir(exist_ok=True, parents=True)
    
    return workspace_path, drive_path

def download_from_github():
    """Download dataset from GitHub or provide instructions"""
    print("\n" + "="*70)
    print("DATASET SETUP")
    print("="*70)
    print("\nYou have two options to get your dataset:")
    print("\n1. MANUAL UPLOAD (Recommended if dataset is small):")
    print("   - Upload 'Root Canal.v1i.yolov8' folder to /content/drive/MyDrive/")
    print("   - Update DATASET_PATH in the script")
    print("\n2. DOWNLOAD FROM CLOUD STORAGE:")
    print("   - If you have a cloud link, use:")
    print("   os.system('wget -q https://your-link-here/dataset.zip && unzip dataset.zip')")
    print("\nPress Enter after uploading to continue...")
    input()

def print_colab_header():
    """Print Colab setup information"""
    print("\n" + "="*70)
    print("GOOGLE COLAB SETUP - KNOWLEDGE DISTILLATION PIPELINE")
    print("="*70)
    print("\n✓ Running in Google Colab with GPU support")
    print("✓ Free training with NVIDIA GPU (12+ hours per session)")
    print("✓ 100GB+ storage via Google Drive")
    print("\n" + "="*70 + "\n")

# ============================================================================
# MAIN COLAB NOTEBOOK CELLS - PASTE THESE INTO SEPARATE CELLS IN COLAB
# ============================================================================

CELL_1_SETUP = """
# Cell 1: Setup Google Colab Environment
import os
import sys
from pathlib import Path

# Check GPU availability
!nvidia-smi

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create working directory
!mkdir -p /content/root-canal-ai
%cd /content/root-canal-ai

print("✓ Google Colab environment setup complete")
"""

CELL_2_INSTALL = """
# Cell 2: Install Dependencies
!pip install -q ultralytics torch torchvision opencv-python numpy pillow matplotlib scikit-learn seaborn tqdm flask flask-cors pandas

print("✓ All dependencies installed")
"""

CELL_3_DOWNLOAD = """
# Cell 3: Download Project Files
# Option A: If you cloned a GitHub repo
!git clone https://github.com/your-username/Root-Canal-AI.git /content/root-canal-ai-project
%cd /content/root-canal-ai-project

# OR Option B: Upload files manually
# Upload your files to Google Drive, then:
# !cp -r /content/drive/MyDrive/Root-Canal-AI/* /content/root-canal-ai-project/

# Verify files
!ls -la

print("✓ Project files ready")
"""

CELL_4_DATASET = """
# Cell 4: Prepare Dataset
# IMPORTANT: Make sure your dataset is uploaded to Google Drive

# Check if dataset exists
import os
dataset_path = '/content/drive/MyDrive/Root Canal.v1i.yolov8'
if os.path.exists(dataset_path):
    print(f"✓ Dataset found at {dataset_path}")
else:
    print("⚠ Dataset not found. Please upload 'Root Canal.v1i.yolov8' to Google Drive")
    print("Expected path: /content/drive/MyDrive/Root Canal.v1i.yolov8")

# Copy dataset to workspace for faster access (optional)
!cp -r '/content/drive/MyDrive/Root Canal.v1i.yolov8' .

print("✓ Dataset prepared")
"""

CELL_5_TRAIN = """
# Cell 5: Run Knowledge Distillation Pipeline

# Option A: Run full pipeline (takes 4-6 hours with GPU)
!python run_distillation_pipeline.py --stage all --teacher-epochs 100 --student-epochs 100

# Option B: Run with fewer epochs (takes 1-2 hours)
# !python run_distillation_pipeline.py --stage all --teacher-epochs 30 --student-epochs 30

# Option C: Run individual stages
# !python run_distillation_pipeline.py --stage teacher --teacher-epochs 100
# !python run_distillation_pipeline.py --stage student --student-epochs 100
# !python run_distillation_pipeline.py --stage evaluate
# !python run_distillation_pipeline.py --stage analyze

print("✓ Training complete. Check logs and results directories")
"""

CELL_6_RESULTS = """
# Cell 6: Save Results to Google Drive
import shutil
from pathlib import Path

# Save training results
results_dirs = [
    'distillation_results',
    'evaluation_results', 
    'comparative_results',
    'distillation_pipeline.log'
]

drive_results_path = '/content/drive/MyDrive/Root-Canal-Results'
os.makedirs(drive_results_path, exist_ok=True)

for result_dir in results_dirs:
    if Path(result_dir).exists():
        dst = Path(drive_results_path) / result_dir
        if Path(result_dir).is_dir():
            shutil.copytree(result_dir, dst, dirs_exist_ok=True)
        else:
            shutil.copy(result_dir, dst)
        print(f"✓ Saved {result_dir}")

print(f"\\n✓ All results saved to: {drive_results_path}")
print("Download from Google Drive or access directly in Colab")
"""

CELL_7_INFERENCE = """
# Cell 7: Run Inference Server (Optional)
# Note: Flask server may have limited functionality in Colab

# Start inference server
!python inference_server.py --host 0.0.0.0 --port 5000

# In a production setup, consider using:
# - Gradio for web interface
# - Streamlit for dashboard
"""


def print_colab_instructions():
    """Print detailed Colab setup instructions"""
    print("\n" + "="*70)
    print("GOOGLE COLAB SETUP INSTRUCTIONS")
    print("="*70)
    
    print("\n1. OPEN GOOGLE COLAB")
    print("   Go to: https://colab.research.google.com")
    
    print("\n2. CREATE NEW NOTEBOOK")
    print("   File → New notebook")
    
    print("\n3. COPY AND PASTE CELLS BELOW (One per cell)")
    print("\n" + "-"*70)
    print("CELL 1: SETUP ENVIRONMENT")
    print("-"*70)
    print(CELL_1_SETUP)
    
    print("\n" + "-"*70)
    print("CELL 2: INSTALL DEPENDENCIES")
    print("-"*70)
    print(CELL_2_INSTALL)
    
    print("\n" + "-"*70)
    print("CELL 3: DOWNLOAD PROJECT FILES")
    print("-"*70)
    print(CELL_3_DOWNLOAD)
    print("   Note: Choose either GitHub clone OR manual upload")
    
    print("\n" + "-"*70)
    print("CELL 4: PREPARE DATASET")
    print("-"*70)
    print(CELL_4_DATASET)
    print("   Note: Upload 'Root Canal.v1i.yolov8' to Google Drive first")
    
    print("\n" + "-"*70)
    print("CELL 5: RUN PIPELINE")
    print("-"*70)
    print(CELL_5_TRAIN)
    print("   Uncomment the option you want to run")
    print("   With GPU: ~2-6 hours depending on epochs")
    
    print("\n" + "-"*70)
    print("CELL 6: SAVE RESULTS")
    print("-"*70)
    print(CELL_6_RESULTS)
    
    print("\n" + "-"*70)
    print("CELL 7: INFERENCE (Optional)")
    print("-"*70)
    print(CELL_7_INFERENCE)
    
    print("\n" + "="*70)
    print("ADDITIONAL TIPS")
    print("="*70)
    print("""
✓ GPU ACCELERATION:
  - Colab provides free NVIDIA GPUs
  - Enable GPU: Runtime → Change runtime type → GPU
  - Training will be 10-20x faster than CPU

✓ FILE MANAGEMENT:
  - Use Google Drive to persist files between sessions
  - Copy large datasets to /content for faster access
  - Save results back to Drive for download

✓ COST:
  - Completely FREE with Google account
  - Up to 12 hours per session
  - No credit card required

✓ REDUCING TRAINING TIME:
  - Use --teacher-epochs 30 --student-epochs 30 (1-2 hours)
  - Or run individual stages separately
  - Cache intermediate results

✓ DEBUGGING:
  - Check GPU: !nvidia-smi
  - Monitor memory: !nvidia-smi --query-gpu=memory.used,memory.free --format=csv
  - Check CUDA: !python -c "import torch; print(torch.cuda.is_available())"
""")

    print("\n" + "="*70)
    print("QUICK START COMMAND")
    print("="*70)
    print("""
Copy this entire command into a single Colab cell for quick setup:

# Quick setup
!pip install -q ultralytics torch torchvision opencv-python numpy pillow matplotlib scikit-learn seaborn tqdm
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/root-canal && cd /content/root-canal
!git clone <YOUR_REPO_URL> .
!cp -r '/content/drive/MyDrive/Root Canal.v1i.yolov8' .
!python run_distillation_pipeline.py --stage all --teacher-epochs 30 --student-epochs 30
""")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    print_colab_instructions()
    
    # Check if running in Colab
    try:
        import google.colab
        print("\n✓ Running in Google Colab!")
        print_colab_header()
    except ImportError:
        print("\n⚠ Not running in Google Colab")
        print("To use this script in Colab:")
        print("1. Copy all instructions above")
        print("2. Go to colab.research.google.com")
        print("3. Create new notebook and paste cells")
