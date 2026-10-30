import os
from pathlib import Path

project_dir = Path(r'c:\Users\PRO\Desktop\root canal\Root-canal-AI-diagnostics')
images_dir = project_dir / 'dataset' / 'images'
labels_dir = project_dir / 'dataset' / 'labels'

print(f'Project dir exists: {project_dir.exists()}')
print(f'Images dir exists: {images_dir.exists()}')
print(f'Labels dir exists: {labels_dir.exists()}')

if images_dir.exists():
    images = list(images_dir.glob('*.*'))
    print(f'Found {len(images)} images')
    if images:
        for img in images[:5]:
            print(f'  - {img.name}')

if labels_dir.exists():
    labels = list(labels_dir.glob('*.txt'))
    print(f'Found {len(labels)} label files')
