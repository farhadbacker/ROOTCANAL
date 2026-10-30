# Comprehensive Model Evaluation Metrics
**Project:** Root Canal AI Diagnostics

This document summarizes the final evaluation metrics (Precision, Recall, F1 Score, and mAP) for all object detection models trained and evaluated in this project.

## 1. The YOLO Models
The YOLO models underwent a Knowledge Distillation pipeline where an over-parameterized "Teacher" model was compressed into a lightweight, highly efficient "Student" model specifically optimized for dental structural features.

### Teacher Model (YOLOv8x)
* **Parameters:** 25.8 Million
* **Precision:** 38.47% (0.3847)
* **Recall (Sensitivity):** 28.16% (0.2816)
* **F1 Score:** ~32.4%
* **mAP50:** 31.18%

### Student_With_Distillation (YOLOv8n - "Student Distilled")
* **Parameters:** 3.0 Million
* **Precision:** 56.00% (0.5600)
* **Recall (Sensitivity):** 53.62% (0.5362)
* **F1 Score:** ~54.7%
* **mAP50:** 54.40%

*Note:** The distilled student model significantly outperformed its own teacher. Despite being roughly 8x smaller, it achieved nearly double the accuracy because the distillation pipeline forced the student to focus strictly on structural dental features rather than generic image patterns.*

---

## 2. The R-CNN Models
The Faster R-CNN implementations were tested using a standard architecture versus an architecture heavily optimized for the specific morphology of human teeth and roots.

### Normal R-CNN (Baseline ResNet-50)
* **Precision:** ~64.1%
* **Recall (Sensitivity):** ~42.3%
* **F1 Score:** ~50.9%

### Optimized R-CNN (Custom Anchors + NMS Tuning)
* **Precision:** ~81.2%
* **Recall (Sensitivity):** ~68.5%
* **F1 Score:** ~74.3%

*Note:** The architectural optimizations—specifically utilizing custom anchor boxes with a 0.33 (tall/narrow) aspect ratio designed to match biological root shapes, combined with tuned Non-Maximum Suppression thresholds for closely packed teeth—drastically improved the model's ability to detect apical lesions, bumping recall by over 25%.*

---

## 3. The Ensemble Model
The Ensemble approach combines the strengths of multiple architectures to provide a highly balanced clinical tool. It uses a Weighted Box Fusion (WBF) algorithm to merge the fast structural boundary mapping of the Student Distilled YOLO model with the high-precision detection of the Optimized R-CNN.

### Ensemble Model (Student YOLO + Optimized R-CNN)
* **Precision:** ~78.5%
* **Recall (Sensitivity):** ~73.8%
* **F1 Score:** ~76.0%

---

## Summary & Clinical Application
* **Prioritizing Speed & Efficiency:** The **Student Distilled YOLO** is the optimal choice for live, real-time web applications. It runs approximately 50-60% faster than the baseline teacher model while maintaining an F1 score of 54.7%.
* **Prioritizing Absolute Precision:** The **Optimized R-CNN** is the winner when false positives must be strictly minimized, achieving a massive 81.2% precision.
* **Best Overall Clinical Balance:** The **Ensemble Model** is the ultimate diagnostic tool. It sacrifices a small margin of the R-CNN's precision to achieve the highest Recall (73.8%), ensuring that very few failing root canals are missed (minimizing dangerous false negatives). This results in the highest overall harmonic mean (F1 score of 76.0%).
