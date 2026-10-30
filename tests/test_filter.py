import numpy as np
import pytest
from contourlet_filter import ContourletTransform, apply_contourlet_filter

def test_laplacian_pyramid():
    """Test the laplacian pyramid decomposition produces correct level counts"""
    img = np.random.rand(100, 100).astype(np.float32)
    filter_obj = ContourletTransform(num_levels=3, num_directions=8)
    pyramid = filter_obj.apply_laplacian_pyramid(img)
    # 3 levels down = length 3
    assert len(pyramid) == 3

def test_contourlet_apply_shape():
    """Test that applying contourlet filter returns an array of the exact same dimensions"""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    filter_obj = ContourletTransform(num_levels=2, num_directions=8)
    filtered = filter_obj.apply(img)
    
    assert filtered.shape == img.shape
    assert filtered.dtype == np.uint8

def test_contourlet_grayscale():
    """Test grayscale images are handled correctly without crashing"""
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    filter_obj = ContourletTransform(num_levels=2, num_directions=8)
    filtered = filter_obj.apply(img)
    
    assert filtered.shape == img.shape
    assert filtered.dtype == np.uint8

def test_contourlet_empty_image():
    """Test edge case handling of an empty or degenerated zero-image"""
    img = np.array([], dtype=np.uint8)
    filter_obj = ContourletTransform(num_levels=2, num_directions=8)
    filtered = filter_obj.apply(img)
    
    # Empty inputs should safely return the input directly
    assert filtered.size == 0
