import requests
import json
import base64
import os

# Create a dummy image
from PIL import Image
import numpy as np
img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
img.save("dummy.jpg")

with open("dummy.jpg", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode('utf-8')

payload = {
    "image": base64_image,
    "enable_filter": True,
    "enable_gradcam": True
}

try:
    response = requests.post("http://localhost:5000/detect", json=payload)
    data = response.json()
    print("Keys in response:", data.keys())
    if 'heatmap' in data:
        print("Heatmap present:", data['heatmap'] is not None)
        if data['heatmap']:
            print("Heatmap string length:", len(data['heatmap']))
    if 'error' in data:
        print("Error:", data['error'])
except Exception as e:
    print("Request failed:", e)

if os.path.exists("dummy.jpg"):
    os.remove("dummy.jpg")
