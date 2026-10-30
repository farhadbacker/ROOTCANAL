import pytest
import os
import sys
import numpy as np
from unittest.mock import MagicMock, patch

# Add parent dir to sys.path so we can import server modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def mock_yolo():
    with patch('inference_server.YOLO') as mock:
        instance = MagicMock()
        # Mock predict method to return dummy detections
        # YOLO model.predict returns a list of Results objects
        mock_result = MagicMock()
        mock_box = MagicMock()
        # Simulate CPU tensor logic inside predict
        mock_box.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: [100.0, 100.0, 200.0, 200.0]))]
        mock_box.conf = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: 0.95))]
        mock_box.cls = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: 0))]
        mock_result.boxes = [mock_box]
        
        instance.predict.return_value = [mock_result]
        mock.return_value = instance
        yield mock

@pytest.fixture
def mock_rcnn():
    with patch('inference_server.get_optimized_rcnn_model') as mock_get, \
         patch('inference_server.torch') as mock_torch:
         
        # Mock RCNN instance
        instance = MagicMock()
        # Mock torch dict output
        instance.return_value = [{
            'boxes': MagicMock(cpu=lambda: MagicMock(numpy=lambda: [[100, 100, 200, 200]])),
            'scores': MagicMock(cpu=lambda: MagicMock(numpy=lambda: [0.95])),
            'labels': MagicMock(cpu=lambda: MagicMock(numpy=lambda: [1])) # RCNN labels 1-indexed
        }]
        
        mock_get.return_value = instance
        yield mock_get

@pytest.fixture
def mock_filter():
    with patch('inference_server.ContourletTransform') as mock:
        instance = MagicMock()
        # The filter just returns the passed-in array in mock
        instance.apply.side_effect = lambda x: x
        mock.return_value = instance
        yield mock

@pytest.fixture
def client(mock_yolo, mock_rcnn, mock_filter):
    import inference_server
    
    # Disable actual contourlet transform loading overhead for quick tests
    inference_server.use_filter = False
    
    # Initialize the flask app test client
    inference_server.app.config['TESTING'] = True
    with inference_server.app.test_client() as client:
        yield client
