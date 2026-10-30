"""Quick test to verify GradCAM heatmap generation from the inference server."""
import requests
import base64
import json

# Read a test image
with open("dataset/images/1.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode("utf-8")

# Call the detect endpoint with GradCAM enabled
resp = requests.post("http://localhost:5000/detect", json={
    "image": img_base64,
    "enable_gradcam": True
})

data = resp.json()

print(f"Status: {resp.status_code}")
print(f"Detections: {len(data.get('detections', []))}")
print(f"Image size: {data.get('image_size')}")
print(f"Model used: {data.get('model_used')}")

heatmap = data.get("heatmap")
if heatmap:
    heatmap_bytes = base64.b64decode(heatmap)
    print(f"[OK] Heatmap received: {len(heatmap_bytes)} bytes")
    # Save heatmap to disk to visually inspect
    with open("test_heatmap_output.jpg", "wb") as f:
        f.write(heatmap_bytes)
    print("[OK] Saved heatmap to test_heatmap_output.jpg")
else:
    print("✗ No heatmap received!")
