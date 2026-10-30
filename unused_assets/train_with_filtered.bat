@echo off
REM Step 1: Apply Contourlet filtering to dataset
echo.
echo ============================================================
echo Step 1: Preprocessing images with Contourlet Transform
echo ============================================================
echo.
python preprocess_dataset.py --backup --levels 2 --directions 8

echo.
echo ============================================================
echo Step 2: Training YOLO model with filtered images
echo ============================================================
echo.
python train_yolo.py --filtered --device cpu --epochs 100 --batch-size 16

pause
