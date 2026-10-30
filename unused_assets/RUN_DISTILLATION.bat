@echo off
REM Knowledge Distillation Pipeline Runner for Windows
REM This script runs the complete knowledge distillation pipeline

echo.
echo ================================================================================
echo Knowledge Distillation Pipeline - Root Canal Diagnosis
echo ================================================================================
echo.
echo Select stage to run:
echo   1. Full Pipeline (Teacher + Student + Evaluate + Analyze)
echo   2. Teacher Training Only
echo   3. Student Training with Distillation Only
echo   4. Model Evaluation Only
echo   5. Comparative Analysis Only
echo   6. Start Inference Server
echo.

set /p choice="Enter choice (1-6): "

if "%choice%"=="1" (
    echo Running full pipeline...
    python run_distillation_pipeline.py --stage all
) else if "%choice%"=="2" (
    echo Training teacher model...
    python run_distillation_pipeline.py --stage teacher
) else if "%choice%"=="3" (
    echo Training student model with distillation...
    python run_distillation_pipeline.py --stage student
) else if "%choice%"=="4" (
    echo Evaluating models...
    python run_distillation_pipeline.py --stage evaluate
) else if "%choice%"=="5" (
    echo Running comparative analysis...
    python run_distillation_pipeline.py --stage analyze
) else if "%choice%"=="6" (
    echo Starting inference server...
    python run_distillation_pipeline.py --stage inference
) else (
    echo Invalid choice!
    exit /b 1
)

pause
