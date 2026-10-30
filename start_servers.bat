@echo off
echo Starting Dental AI Diagnostics System...
echo.

echo Starting Python inference server...
start cmd /k "cd /d %~dp0 && python inference_server.py"

timeout /t 3 /nobreak > nul

echo Starting React development server...
start cmd /k "cd /d %~dp0 && npm run dev"

echo.
echo Both servers are starting up...
echo - Inference server: http://localhost:5000
echo - React app: http://localhost:3002
echo.
echo Press any key to close this window...
pause > nul