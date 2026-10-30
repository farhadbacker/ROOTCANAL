import pytest
import numpy as np
from ensemble_detection import box_iou, MedicalEnsemble
from unittest.mock import patch, MagicMock

def test_box_iou_perfect_match():
    """Test Intersection over Union for identical bounding boxes"""
    box1 = [10, 10, 50, 50]
    box2 = [10, 10, 50, 50]
    iou = box_iou(box1, box2)
    assert iou == 1.0

def test_box_iou_no_overlap():
    """Test IoU for completely separated boxes"""
    box1 = [10, 10, 50, 50]
    box2 = [100, 100, 150, 150]
    iou = box_iou(box1, box2)
    assert iou == 0.0

def test_box_iou_partial_overlap():
    """Test IoU for partially overlapping boxes"""
    # 40x40 area = 1600
    box1 = [10, 10, 50, 50] 
    # Intersects top left quarter
    box2 = [30, 30, 70, 70]
    
    iou = box_iou(box1, box2)
    assert iou > 0.0 and iou < 1.0

@patch('ensemble_detection.YOLO')
def test_ensemble_consensus_filtering(mock_yolo_constructor):
    """
    Test that the ensemble logic safely prunes detections that
    don't have mult-model consensus agreement (IoU > thresh).
    """
    mock_model1 = MagicMock()
    mock_model2 = MagicMock()
    
    # Model 1 detects something at config class '0'
    res1_box = MagicMock()
    res1_box.xyxy = [np.array([10, 10, 50, 50])]
    res1_box.conf = MagicMock(item=lambda: 0.9)
    res1_box.cls = MagicMock(item=lambda: 0)
    
    # Model 1 also detects a false positive isolated noise box
    res1_noise = MagicMock()
    res1_noise.xyxy = [np.array([200, 200, 220, 220])]
    res1_noise.conf = MagicMock(item=lambda: 0.6)
    res1_noise.cls = MagicMock(item=lambda: 0)

    # Model 2 detects the first real object (consensus!)
    res2_box = MagicMock()
    res2_box.xyxy = [np.array([12, 12, 48, 48])]
    res2_box.conf = MagicMock(item=lambda: 0.95)
    res2_box.cls = MagicMock(item=lambda: 0)
    
    mock_res1 = MagicMock(boxes=[res1_box, res1_noise])
    mock_res2 = MagicMock(boxes=[res2_box])
    
    mock_model1.predict.return_value = [mock_res1]
    mock_model2.predict.return_value = [mock_res2]
    
    # Side effect binds our mock instances
    mock_yolo_constructor.side_effect = [mock_model1, mock_model2]
    
    ensemble = MedicalEnsemble(['path1.pt', 'path2.pt'])
    predictions = ensemble.predict('dummy.jpg')
    
    # The noise box should be filtered out by consensus threshold logic.
    # Therefore we expect exactly 1 merged detection.
    assert len(predictions) == 1
    
    merged_box = predictions[0]
    # merged_box format is [x1, y1, x2, y2, avg_conf, class_id]
    assert merged_box[4] == 0.925 # (0.9 + 0.95) / 2
    assert merged_box[5] == 0
