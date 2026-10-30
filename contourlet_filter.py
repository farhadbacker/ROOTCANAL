import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter


class ContourletTransform:
    def __init__(self, num_levels=2, num_directions=8):
        self.num_levels = num_levels
        self.num_directions = num_directions
    
    def apply_laplacian_pyramid(self, image):
        """Apply Laplacian pyramid decomposition"""
        gaussian_pyramid = [image.copy().astype(np.float32)]
        current = image.copy().astype(np.float32)
        
        for i in range(self.num_levels - 1):
            if current.size == 0 or current.shape[0] < 2 or current.shape[1] < 2:
                break
            smoothed = cv2.pyrDown(current)
            gaussian_pyramid.append(smoothed)
            
            if smoothed.shape[0] > 0 and smoothed.shape[1] > 0:
                expanded = cv2.pyrUp(smoothed, dstsize=(current.shape[1], current.shape[0]))
                current = current - expanded
            else:
                break
        
        return gaussian_pyramid
    
    def apply_directional_filter(self, image, direction, scale=1.0):
        """Apply directional Gabor-like filter"""
        angle = (direction / self.num_directions) * np.pi
        
        kernel_size = 15
        x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
        y = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
        X, Y = np.meshgrid(x, y)
        
        sigma_x = 3.0
        sigma_y = 1.0
        
        X_theta = X * np.cos(angle) + Y * np.sin(angle)
        Y_theta = -X * np.sin(angle) + Y * np.cos(angle)
        
        gabor_kernel = np.exp(
            -(X_theta**2 / (2 * sigma_x**2) + Y_theta**2 / (2 * sigma_y**2))
        ) * np.cos(2 * np.pi * X_theta / 5)
        
        gabor_kernel = gabor_kernel / np.sum(np.abs(gabor_kernel))
        
        if len(image.shape) == 3:
            filtered = cv2.filter2D(image, -1, gabor_kernel.astype(np.float32))
        else:
            filtered = cv2.filter2D(image, -1, gabor_kernel.astype(np.float32))
        
        return filtered
    
    def apply_dft_directional_filter_bank(self, image):
        """Apply DFB (Directional Filter Bank) using directional filters"""
        filtered_images = []
        
        for direction in range(self.num_directions):
            filtered = self.apply_directional_filter(image, direction)
            filtered_images.append(np.abs(filtered))
        
        return filtered_images
    
    def combine_directional_responses(self, directional_responses):
        """Combine directional filter responses using various methods"""
        if not directional_responses:
            return np.zeros((1, 1), dtype=np.float32)
        
        try:
            max_response = directional_responses[0].copy()
            mean_response = directional_responses[0].copy()
            
            for response in directional_responses[1:]:
                if response.shape == max_response.shape:
                    max_response = np.maximum(max_response, response)
                    mean_response = mean_response + response
            
            mean_response = mean_response / len(directional_responses)
            combined = 0.6 * max_response + 0.4 * mean_response
            
            return combined
        except Exception as e:
            if directional_responses:
                return directional_responses[0]
            return np.zeros((1, 1), dtype=np.float32)
    
    def enhance_edges(self, image):
        """Enhance edges using Sobel operators"""
        grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return magnitude
    
    def apply(self, image):
        """Apply Contourlet-like transform to image"""
        try:
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                elif image.shape[2] == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image[:,:,0]
            else:
                gray = image.copy()
            
            if gray is None or gray.size == 0:
                return image
            
            gray = gray.astype(np.float32) / 255.0
            
            laplacian_pyramid = self.apply_laplacian_pyramid(gray)
            
            if not laplacian_pyramid:
                return image
            
            contourlet_coefficients = None
            valid_responses = []
            
            for level_img in laplacian_pyramid:
                if level_img is None or level_img.size == 0:
                    continue
                
                try:
                    min_val = np.min(level_img)
                    max_val = np.max(level_img)
                    
                    if max_val - min_val < 1e-6:
                        level_img_normalized = np.ones_like(level_img) * 0.5
                    else:
                        level_img_normalized = (level_img - min_val) / (max_val - min_val + 1e-6)
                    
                    directional_responses = self.apply_dft_directional_filter_bank(level_img_normalized)
                    
                    if directional_responses:
                        combined_response = self.combine_directional_responses(directional_responses)
                        if combined_response.size > 0:
                            valid_responses.append(combined_response)
                except Exception as e:
                    continue
            
            if valid_responses:
                if len(valid_responses) == 1:
                    contourlet_coefficients = valid_responses[0]
                else:
                    try:
                        contourlet_coefficients = np.mean(np.array(valid_responses), axis=0)
                    except:
                        contourlet_coefficients = valid_responses[0]
            else:
                contourlet_coefficients = gray
            
            edge_map = self.enhance_edges(gray)
            
            min_coeff = np.min(contourlet_coefficients)
            max_coeff = np.max(contourlet_coefficients)
            if max_coeff - min_coeff > 1e-6:
                contourlet_coefficients = (contourlet_coefficients - min_coeff) / (max_coeff - min_coeff + 1e-6)
            else:
                contourlet_coefficients = np.ones_like(contourlet_coefficients) * 0.5
            
            min_edge = np.min(edge_map)
            max_edge = np.max(edge_map)
            if max_edge - min_edge > 1e-6:
                edge_map = (edge_map - min_edge) / (max_edge - min_edge + 1e-6)
            else:
                edge_map = np.ones_like(edge_map) * 0.5
            
            enhanced = 0.7 * contourlet_coefficients + 0.3 * edge_map
            enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
            
            if len(image.shape) == 3:
                enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                enhanced_color = cv2.addWeighted(image, 0.4, enhanced_color, 0.6, 0)
                return enhanced_color
            else:
                return enhanced
        
        except Exception as e:
            return image


def apply_contourlet_filter(image_path, output_path=None, num_levels=2, num_directions=8):
    """
    Apply Contourlet transform to an image
    
    Args:
        image_path: Path to input image
        output_path: Path to save filtered image (optional)
        num_levels: Number of pyramid levels
        num_directions: Number of directional filters
    
    Returns:
        Filtered image array
    """
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    ct = ContourletTransform(num_levels=num_levels, num_directions=num_directions)
    filtered = ct.apply(image)
    
    if output_path:
        cv2.imwrite(output_path, filtered)
    
    return filtered
