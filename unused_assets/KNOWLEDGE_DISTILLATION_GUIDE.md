# Knowledge Distillation Implementation Guide

## Overview

This document describes the knowledge distillation pipeline implementation for the root canal failure diagnosis system, based on the paper:

**"Fusion of Image Filtering and Knowledge-Distilled YOLO Models for Root Canal Failure Diagnosis"**

## What is Knowledge Distillation?

Knowledge distillation is a technique where a smaller student model learns from a larger teacher model. The student model learns to mimic the teacher's behavior, resulting in:

- **Smaller models** with fewer parameters
- **Faster inference** with minimal accuracy loss
- **Better generalization** through teacher guidance
- **Improved performance** over training the student alone

## Architecture

### Teacher Model (YOLOv8m)
- **Size**: ~25.9 million parameters
- **Purpose**: Full-capacity model serving as knowledge source
- **Training**: Standard YOLO training on filtered dental X-rays
- **Role**: Provides soft targets for student distillation

### Student Model (YOLOv8n)
- **Size**: ~6.9 million parameters (73% reduction)
- **Purpose**: Lightweight model for deployment
- **Training**: Learning from teacher via distillation loss
- **Benefit**: Fast inference with comparable accuracy to teacher

### Distillation Loss Function

The training combines two losses:

```
Total Loss = α × Task Loss + (1 - α) × Distillation Loss

Where:
- α = 0.7 (task loss weight, typically 0.5-0.9)
- Temperature = 4.0 (controls softness of soft targets)
```

**Task Loss**: Standard YOLO detection loss
**Distillation Loss**: KL-divergence between student and teacher predictions

## Pipeline Stages

### Stage 1: Teacher Model Training
```bash
python run_distillation_pipeline.py --stage teacher
```

**Steps**:
1. Load YOLOv8m architecture
2. Train on filtered dental X-ray images
3. Save best.pt with highest mAP50
4. Generate training metrics

**Output**:
- `distillation_results/teacher_model/weights/best.pt`
- Training logs and metrics

**Expected Training Time**: 4-8 hours (GPU recommended)

### Stage 2: Student Training with Distillation
```bash
python run_distillation_pipeline.py --stage student
```

**Steps**:
1. Load teacher model weights
2. Initialize student model (YOLOv8n)
3. Compute distillation loss during training
4. Apply temperature-scaled softmax
5. Save best performing student

**Output**:
- `distillation_results/student_with_distillation/weights/best.pt`
- Training logs with distillation metrics

**Expected Training Time**: 2-4 hours

### Stage 3: Model Evaluation
```bash
python run_distillation_pipeline.py --stage evaluate
```

**Metrics Computed**:
- **Overall**: Precision, Recall, mAP50, mAP50-95
- **Per-Class**: AP50 for each endodontic treatment category
- **Medical**: Sensitivity, Specificity per class
- **Confusion Matrix**: Visual classification breakdown

**Output**:
- `evaluation_results/evaluation_report.txt`
- `evaluation_results/results.json`
- Confusion matrices (PNG images)
- Model comparison visualizations

### Stage 4: Comparative Analysis
```bash
python run_distillation_pipeline.py --stage analyze
```

**Analysis Includes**:

1. **Filtering Impact**:
   - Raw images vs. Filtered images
   - Performance improvement from Contourlet transform

2. **Distillation Impact**:
   - Student baseline vs. Student with distillation
   - Accuracy gains from knowledge transfer

3. **Model Efficiency**:
   - Parameter count comparison
   - File size comparison
   - Inference speed estimates

**Output**:
- `comparative_results/analysis_report.txt`
- `comparative_results/comprehensive_analysis.png`
- `comparative_results/analysis_results.json`

## Complete Pipeline Execution

Run the entire pipeline with one command:

```bash
python run_distillation_pipeline.py --stage all
```

Or using the batch script:
```batch
RUN_DISTILLATION.bat
```

## Model Configuration

The `model_config.json` file manages available models:

```json
{
  "default_model": "student_distilled",
  "available_models": {
    "student_distilled": {
      "name": "Student Distilled (YOLOv8n + KD)",
      "path": "distillation_results/student_with_distillation/weights/best.pt",
      "use_filter": true,
      "improvement_metrics": {...}
    }
  }
}
```

## Inference Server

The inference server supports dynamic model switching:

### Start Server
```bash
python inference_server.py
```

Or via pipeline:
```bash
python run_distillation_pipeline.py --stage inference
```

### API Endpoints

**Health Check**:
```bash
curl http://localhost:5000/health
```
Returns current model and available models.

**List Available Models**:
```bash
curl http://localhost:5000/models
```

**Switch Model**:
```bash
curl -X POST http://localhost:5000/models/student_baseline
curl -X POST http://localhost:5000/models/student_distilled
curl -X POST http://localhost:5000/models/teacher_model
```

**Run Detection**:
```bash
curl -X POST http://localhost:5000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,..."
  }'
```

## Results Interpretation

### Performance Metrics

**Precision**: % of predicted positives that are correct
**Recall**: % of actual positives that were found
**mAP50**: Average precision at 50% IoU threshold
**F1-Score**: Harmonic mean of precision and recall

### Medical Metrics

**Sensitivity**: Ability to detect positive cases (true positive rate)
**Specificity**: Ability to reject negative cases (true negative rate)

### Distillation Gains

Look for improvements in student model when distilled:
- **Good**: >2% accuracy improvement
- **Excellent**: >5% accuracy improvement
- **Typical**: 2-4% improvement in mAP

### Model Size Reduction

- **Student Parameters**: 6.9M vs 25.9M (Teacher) = **73% reduction**
- **Inference Speed**: Expected 40-50% faster with student
- **Accuracy Drop**: Minimal with knowledge distillation

## File Structure

```
root-canal-ai-diagnostics/
├── knowledge_distillation.py           # Main distillation trainer
├── evaluate_models.py                  # Evaluation and metrics
├── comparative_analysis.py             # Analysis script
├── run_distillation_pipeline.py        # Pipeline orchestrator
├── inference_server.py                 # Updated with model config
├── model_config.json                   # Model specifications
├── RUN_DISTILLATION.bat               # Windows batch runner
│
├── distillation_results/               # Training outputs
│   ├── teacher_model/weights/
│   ├── student_with_distillation/weights/
│   └── [training logs]
│
├── evaluation_results/                 # Evaluation metrics
│   ├── evaluation_report.txt
│   ├── results.json
│   └── confusion_matrix_*.png
│
└── comparative_results/                # Analysis outputs
    ├── analysis_report.txt
    ├── analysis_results.json
    └── comprehensive_analysis.png
```

## Performance Benchmarks

### Expected Results

| Metric | Student | Student Distilled | Teacher |
|--------|---------|-------------------|---------|
| Precision | ~0.82 | ~0.85-0.87 | ~0.88 |
| Recall | ~0.80 | ~0.83-0.85 | ~0.86 |
| mAP50 | ~0.81 | ~0.84-0.86 | ~0.87 |
| Parameters | 6.9M | 6.9M | 25.9M |
| Inference Time | ~45ms | ~45ms | ~85ms |

*Benchmarks based on dental X-ray dataset. Actual results may vary.*

## Troubleshooting

### Model Loading Fails
- Ensure model path exists in `model_config.json`
- Check file permissions
- Verify PyTorch and CUDA compatibility

### Training Runs Out of Memory
- Reduce batch size in YOLO training
- Use gradient accumulation
- Reduce image size (imgsz)

### Poor Distillation Results
- Increase training epochs
- Adjust temperature value (try 3.0-5.0)
- Adjust alpha weight (try 0.5-0.9)
- Check teacher model quality first

### Inference Server Errors
- Check server logs for detailed error messages
- Run `/health` endpoint to verify server status
- Verify model files exist at configured paths

## References

- **YOLO**: Ultralytics YOLOv8 (https://github.com/ultralytics/ultralytics)
- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network"
- **Contourlet Transform**: Do & Vetterli, "The Contourlet Transform"

## Future Improvements

1. **Ensemble Methods**: Combine multiple distilled models
2. **Progressive Distillation**: Use intermediate-sized teachers
3. **Attention Transfer**: Transfer attention maps in addition to logits
4. **Quantization**: Further compress models for edge deployment
5. **Multi-Teacher Distillation**: Learn from multiple specialist teachers

## Citation

If you use this implementation, please cite:

```
@article{rootcanaldiagnosis,
  title={Fusion of Image Filtering and Knowledge-Distilled YOLO Models for Root Canal Failure Diagnosis},
  year={2024}
}
```

## License

This implementation follows the same license as the main project.

---

**Last Updated**: 2024
**Version**: 1.0
