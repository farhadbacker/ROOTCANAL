import numpy as np
import cv2
from contourlet_filter import ContourletTransform

print("Testing fixed preprocessing...")

contourlet_filter = ContourletTransform(num_levels=2, num_directions=8)

def apply_preprocessing(img_array):
    try:
        filtered = contourlet_filter.apply(img_array)
        return filtered
    except Exception as e:
        print(f"Exception out: {e}")
        return img_array

# Create dummy uint8 image
test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

print("Original shape:", test_img.shape, "dtype:", test_img.dtype)
processed = apply_preprocessing(test_img)
print("Processed shape:", processed.shape, "dtype:", processed.dtype)
print("Max value:", np.max(processed))
