with open('public/dental_yolo.onnx', 'rb') as f:
    data = f.read(10)
    print('First 10 bytes:', [hex(b) for b in data])

# Also check file size
import os
size = os.path.getsize('public/dental_yolo.onnx')
print(f'File size: {size} bytes')