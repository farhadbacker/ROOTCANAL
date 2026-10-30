@echo off
cd /d "%~dp0"
echo Starting Dental X-Ray Inference Server...
echo Using model: dental_yolo_roboflow.pt
echo.
python inference_server.py
pause
