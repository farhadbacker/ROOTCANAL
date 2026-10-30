@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
python run_autolabel_internal.py
pause
