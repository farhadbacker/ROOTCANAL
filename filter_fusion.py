import cv2
import numpy as np
from contourlet_filter import ContourletTransform

class FilterFusion:
    """
    Implements the 'Filter Fusion' strategy for dental radiograph enhancement.
    Combines:
    1. Contourlet Transform (Edge & Directional enhancement)
    2. Non-local Means Denoising (Preserves textures while removing noise)
    3. Bayesian-like Contrast Sharpening (Enhances anatomical pathology)
    """
    def __init__(self, use_contourlet=True, use_nlm=True, use_sharpen=True):
        self.use_contourlet = use_contourlet
        self.use_nlm = use_nlm
        self.use_sharpen = use_sharpen
        self.ct = ContourletTransform(num_levels=2, num_directions=8)

    def apply_nlm(self, image):
        """Apply Non-Local Means Denoising"""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    def apply_bayesian_sharpen(self, image):
        """
        Approximate Bayesian-like sharpening using Laplacian of Gaussian 
        and selective contrast enhancement.
        """
        # Convert to gray for sharpening mask if color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        # Sharpening mask
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) - Standard in Medical Imaging
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened)

        if len(image.shape) == 3:
            # Re-merge with original color if needed, or return enhanced gray
            res = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            return cv2.addWeighted(image, 0.5, res, 0.5, 0)
        
        return enhanced

    def apply_fusion(self, image):
        """Execute the fusion pipeline"""
        work_img = image.copy()

        # 1. Denoising (Non-local Means)
        if self.use_nlm:
            work_img = self.apply_nlm(work_img)

        # 2. Structural Enhancement (Contourlet)
        if self.use_contourlet:
            work_img = self.ct.apply(work_img)

        # 3. Clinical Precision Enhancement (Sharpening)
        if self.use_sharpen:
            work_img = self.apply_bayesian_sharpen(work_img)

        return work_img

def fuse_filters(image_path, output_path=None):
    """Utility function to apply fusion to a file"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    fusion = FilterFusion()
    result = fusion.apply_fusion(img)
    
    if output_path:
        cv2.imwrite(output_path, result)
        
    return result

if __name__ == "__main__":
    import sys
    from tqdm import tqdm
    from pathlib import Path

    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if target.is_file():
            fuse_filters(sys.argv[1], "fusion_test_output.jpg")
            print(f"Fusion applied to {sys.argv[1]}")
        elif target.is_dir():
            print(f"Processing directory: {target}")
            image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
            image_files = [f for f in target.glob('**/*') if f.suffix in image_extensions]
            
            for img_path in tqdm(image_files, desc="Applying Filter Fusion"):
                fuse_filters(str(img_path), str(img_path)) # Overwrite with fused version
            print(f"✓ Processed {len(image_files)} images in {target}")
