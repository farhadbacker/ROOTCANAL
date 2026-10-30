@echo off
cd /d "%~dp0"
echo.
echo ========================================
echo Full Pipeline: Filter + Train
echo ========================================
echo.
echo This will:
echo 1. Apply contourlet filter to all images
echo 2. Train YOLO model (2-3 hours)
echo 3. Save dental_yolo_roboflow_filtered.pt
echo.
echo ========================================
echo.
python FULL_PIPELINE_FILTERED.py
pause
