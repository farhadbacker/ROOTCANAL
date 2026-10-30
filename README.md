<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Root Canal AI Diagnostics

A dental radiograph analysis system for detecting root canal treatment conditions in X-ray images. The project combines a Python Flask inference server with a React + Vite frontend for image upload, model selection, and AI-assisted reporting.

## Overview

This application is designed to:

- Upload dental X-ray images
- Run object detection using trained YOLO / R-CNN models
- Apply preprocessing filters such as Contourlet enhancement
- Generate heatmap overlays for attention visualization
- Compare model outputs and switch active models
- Produce a clinician-friendly analysis summary using Gemini AI

## Project structure

- `inference_server.py` – Python Flask API that handles model loading and inference
- `model_config.json` – model registry and default model configuration
- `App.tsx` – React UI for image upload and analysis results
- `services/` – frontend services that call the Flask API and Gemini AI
- `*.pt` – trained model weights
- `START_INFERENCE_SERVER.bat` / `run_server.bat` – Windows quick-start scripts

## Prerequisites

Before running the app, make sure you have:

- Python 3.10+ recommended
- Node.js 18+ and npm
- A working internet connection for downloading Python packages and model dependencies
- A Gemini API key if you want AI report generation

## 1) Python backend setup

Open a terminal in the project root and create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

Install the required Python packages:

```bash
python -m pip install --upgrade pip
pip install ultralytics torch torchvision opencv-python pillow numpy flask flask-cors scikit-image
```

If you prefer, you can also install from a project requirements file if one is added later.

## 2) Set the Gemini API key

Create a file named `.env.local` in the project root and add:

```env
API_KEY=your_gemini_api_key_here
```

This project currently reads `process.env.API_KEY` in the frontend service code.

Note: if the AI description feature is not working, double-check the key and restart the frontend server.

## 3) Start the Python inference server

From the project root:

```bash
python inference_server.py
```

Or on Windows, run the helper script:

```cmd
START_INFERENCE_SERVER.bat
```

This starts the Flask API on:

```text
http://localhost:5000
```

The server loads the default YOLO model from `model_config.json` and exposes endpoints such as:

- `GET /models`
- `POST /models/<model_name>`
- `POST /detect`

## 4) Install frontend dependencies

In a second terminal, from the project root:

```bash
npm install
```

## 5) Run the web app

Start the Vite frontend:

```bash
npm run dev
```

Then open the local URL shown in the terminal, typically:

```text
http://localhost:5173
```

## Full startup sequence

Use this in order:

```bash
# Terminal 1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install ultralytics torch torchvision opencv-python pillow numpy flask flask-cors scikit-image
python inference_server.py
```

```bash
# Terminal 2
npm install
npm run dev
```

## Using the app

1. Open the frontend in the browser
2. Upload a dental X-ray image
3. The app sends the image to the Flask backend for detection
4. The active model runs inference
5. Results are displayed with bounding boxes and AI analysis
6. You can switch models from the model selector in the UI

## Model notes

The app is configured with multiple trained models in `model_config.json`, including:

- student baseline models
- distilled student models
- teacher models
- ensemble models
- Faster R-CNN variants
- YOLO11 fine-tuned variants

If a specific `.pt` model file is missing, the app may fall back to a different model or show a warning.

## Troubleshooting

### The frontend cannot connect to the backend

Make sure the Flask server is running and reachable at:

```text
http://localhost:5000
```

### Python imports fail

Recreate the virtual environment and reinstall dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install ultralytics torch torchvision opencv-python pillow numpy flask flask-cors scikit-image
```

### Model files not found

Check that the model weight files exist in the project root or in the expected folders listed in `model_config.json`.

### Gemini analysis fails

Verify that:

- `.env.local` exists in the project root
- `API_KEY` is correctly set
- the frontend server was restarted after editing the environment file

## Development commands

```bash
npm run dev
npm run build
npm run preview
```

## Notes

This project includes both machine learning inference and a user interface for clinical image review. For best results, use a machine with GPU support if available, but CPU mode is also supported for lighter testing and local development.

## License

This project is for research and internal clinical-assist tooling. Check with the project owner before redistribution or commercial use.