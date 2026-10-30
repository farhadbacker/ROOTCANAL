import shutil
import os

src = r'c:\Users\PRO\Desktop\root canal\Root-canal-AI-diagnostics\public\models\dental_yolo.onnx'
dst = r'c:\Users\PRO\Desktop\root canal\Root-canal-AI-diagnostics\public\dental_yolo.onnx'

shutil.move(src, dst)
print(f"Moved {src} to {dst}")