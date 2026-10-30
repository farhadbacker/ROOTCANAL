import pytest
import json
import base64
import numpy as np

def test_health_endpoint(client):
    """Test the /health endpoint returns successfully and lists model metadata"""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert 'available_models' in data
    assert 'model' in data

def test_models_list_endpoint(client):
    """Test the /models endpoint returns the available model specifications"""
    response = client.get('/models')
    assert response.status_code == 200
    data = response.get_json()
    assert 'available_models' in data
    assert 'current_model' in data

def test_detect_endpoint_missing_image(client):
    """Test /detect endpoint handles missing image payload correctly"""
    response = client.post('/detect', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert data['error'] == 'No image provided'

def test_detect_endpoint_success(client):
    """Test a successful payload through the /detect endpoint with mocked inference"""
    # Create a simple 10x10 dummy RGB image in base64
    import cv2
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', dummy_img)
    encoded = base64.b64encode(buffer).decode('utf-8')
    
    response = client.post('/detect', json={
        'image': encoded,
        'enable_filter': False,
        'enable_gradcam': False
    })
    
    assert response.status_code == 200
    data = response.get_json()
    
    # Since we mocked YOLO, we expect exactly 1 detection back
    assert 'detections' in data
    assert 'image_size' in data
    assert len(data['detections']) == 1
    
    detection = data['detections'][0]
    assert 'bbox' in detection
    assert 'confidence' in detection
    assert 'class' in detection

def test_compare_all_endpoint(client):
    """Test the /compare_all endpoint"""
    import cv2
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', dummy_img)
    encoded = base64.b64encode(buffer).decode('utf-8')
    
    response = client.post('/compare_all', json={'image': encoded})
    
    assert response.status_code == 200
    data = response.get_json()
    
    assert 'comparison_results' in data
    assert 'image_size' in data
