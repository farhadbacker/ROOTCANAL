import shutil
import os

src = r'c:\Users\PRO\Desktop\Root Canal\Root-canal-AI-diagnostics\runs\detect\train\weights\best.onnx'
dst = r'c:\Users\PRO\Desktop\root canal\Root-canal-AI-diagnostics\public\models\dental_yolo.onnx'

# Ensure destination directory exists
os.makedirs(os.path.dirname(dst), exist_ok=True)

shutil.copy(src, dst)
print(f"Copied {src} to {dst}")