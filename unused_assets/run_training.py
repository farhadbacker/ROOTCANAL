import os
import sys
from pathlib import Path

os.chdir(str(Path(__file__).parent))

from check_dataset import *
print("\n" + "="*60)
print("Now running preprocessing...")
print("="*60 + "\n")

from preprocess_dataset import main as preprocess_main
import argparse

sys.argv = ['preprocess_dataset.py', '--backup']
preprocess_main()

print("\n" + "="*60)
print("Now training YOLO model with filtered images...")
print("="*60 + "\n")

from train_yolo import train_yolo
train_yolo(use_filtered=True, device='cpu', epochs=10)
