import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from tqdm import tqdm
import json
from contourlet_filter import ContourletTransform

class ModelEvaluator:
    def __init__(self, data_yaml, output_dir='evaluation_results'):
        self.data_yaml = data_yaml
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.class_names = [
            'No Endodontic Treatment',
            'Incomplete Endodontic Treatment',
            'Complete Endodontic Treatment',
            'Total Endodontic Failure'
        ]
        
        self.contourlet = ContourletTransform(num_levels=2, num_directions=8)
        self.log_file = self.output_dir / 'evaluation_report.txt'
        self.results_json = self.output_dir / 'results.json'
        
        self.all_results = {}
    
    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def apply_contourlet_filter(self, image_array):
        try:
            if image_array.dtype != np.float32 and image_array.dtype != np.float64:
                image_array = image_array.astype(np.float32) / 255.0
            
            filtered = self.contourlet.apply(image_array)
            return filtered if filtered is not None else image_array
        except Exception as e:
            print(f"Filter error: {e}")
            return image_array
    
    def get_dataset_samples(self, dataset_path='Root Canal.v1i.yolov8/valid/images'):
        image_files = []
        if os.path.exists(dataset_path):
            for img_file in os.listdir(dataset_path):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(dataset_path, img_file))
        
        return image_files[:100] if image_files else []
    
    def evaluate_on_dataset(self, model_path, use_filter=False, sample_limit=100):
        self.log(f"\n{'='*60}")
        self.log(f"Evaluating: {Path(model_path).name}")
        self.log(f"Filter enabled: {use_filter}")
        self.log(f"{'='*60}")
        
        model = YOLO(model_path)
        
        results = model.val(
            data=self.data_yaml,
            device=self.device,
            imgsz=640,
            conf=0.25,
            iou=0.5
        )
        
        metrics = {
            'precision': float(results.box.mp) if hasattr(results.box, 'mp') else 0,
            'recall': float(results.box.mr) if hasattr(results.box, 'mr') else 0,
            'mAP50': float(results.box.map50) if hasattr(results.box, 'map50') else 0,
            'mAP50-95': float(results.box.map) if hasattr(results.box, 'map') else 0,
        }
        
        for key, value in metrics.items():
            self.log(f"  {key}: {value:.4f}")
        
        return metrics
    
    def compute_per_class_metrics(self, model_path):
        self.log(f"\nPer-Class Metrics for: {Path(model_path).name}")
        self.log("-" * 60)
        
        model = YOLO(model_path)
        
        results = model.val(
            data=self.data_yaml,
            device=self.device,
            imgsz=640,
            conf=0.25
        )
        
        per_class_metrics = {}
        
        if hasattr(results.box, 'ap_class_index'):
            for i, class_idx in enumerate(results.box.ap_class_index):
                class_name = self.class_names[int(class_idx)]
                ap = results.box.ap[i] if hasattr(results.box, 'ap') else 0
                
                per_class_metrics[class_name] = {
                    'AP50': float(ap) if ap is not None else 0
                }
                
                self.log(f"  {class_name}: AP50={per_class_metrics[class_name]['AP50']:.4f}")
        
        return per_class_metrics
    
    def generate_confusion_matrix(self, model_path, output_name='confusion_matrix.png'):
        self.log(f"\nGenerating confusion matrix...")
        
        model = YOLO(model_path)
        
        results = model.val(
            data=self.data_yaml,
            device=self.device,
            imgsz=640,
            conf=0.25,
            save_json=True,
            project=str(self.output_dir),
            name='validation'
        )
        
        if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
            cm = results.confusion_matrix.matrix
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                       xticklabels=self.class_names,
                       yticklabels=self.class_names,
                       cbar_kws={'label': 'Count'})
            plt.title(f'Confusion Matrix - {Path(model_path).name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(self.output_dir / output_name, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"[OK] Confusion matrix saved: {output_name}")
            return cm
        else:
            self.log("[WARNING] Confusion matrix not available")
            return None
    
    def compute_medical_metrics(self, model_path):
        self.log(f"\nMedical-Specific Metrics for: {Path(model_path).name}")
        self.log("-" * 60)
        
        model = YOLO(model_path)
        
        results = model.val(
            data=self.data_yaml,
            device=self.device,
            imgsz=640,
            conf=0.25
        )
        
        medical_metrics = {}
        
        if hasattr(results.box, 'confusion_matrix') and results.box.confusion_matrix is not None:
            cm = results.box.confusion_matrix
            
            for i, class_name in enumerate(self.class_names):
                tn = cm[j, k] if j != i and k != i else 0
                fp = cm[i, j] if j != i else 0
                fn = cm[j, i] if j != i else 0
                tp = cm[i, i]
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                medical_metrics[class_name] = {
                    'Sensitivity': float(sensitivity),
                    'Specificity': float(specificity),
                }
                
                self.log(f"  {class_name}:")
                self.log(f"    Sensitivity (Recall): {sensitivity:.4f}")
                self.log(f"    Specificity: {specificity:.4f}")
        
        return medical_metrics
    
    def compare_models_visual(self, models_dict):
        self.log(f"\nGenerating comparison visualizations...")
        
        metrics_data = {}
        for model_name, model_path in models_dict.items():
            metrics = self.evaluate_on_dataset(model_path)
            metrics_data[model_name] = metrics
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics_list = ['precision', 'recall', 'mAP50', 'mAP50-95']
        
        for idx, metric in enumerate(metrics_list):
            ax = axes[idx // 2, idx % 2]
            
            model_names = list(metrics_data.keys())
            values = [metrics_data[name].get(metric, 0) for name in model_names]
            
            bars = ax.bar(range(len(model_names)), values, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(model_names)])
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric.upper()}')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylim([0, 1.0])
            ax.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log("[OK] Comparison visualization saved")
        return metrics_data
    
    def generate_full_report(self, models_dict, description=""):
        self.log("\n" + "="*60)
        self.log("COMPREHENSIVE MODEL EVALUATION REPORT")
        self.log("="*60)
        
        if description:
            self.log(f"\n{description}\n")
        
        all_metrics = {}
        
        for model_name, model_path in models_dict.items():
            self.log(f"\n{'='*60}")
            self.log(f"MODEL: {model_name}")
            self.log(f"Path: {model_path}")
            self.log("="*60)
            
            metrics = self.evaluate_on_dataset(model_path)
            per_class = self.compute_per_class_metrics(model_path)
            medical = self.compute_medical_metrics(model_path)
            
            self.generate_confusion_matrix(
                model_path, 
                output_name=f'confusion_matrix_{model_name}.png'
            )
            
            all_metrics[model_name] = {
                'overall': metrics,
                'per_class': per_class,
                'medical': medical
            }
        
        self.all_results = all_metrics
        
        self.log(f"\n\n{'='*60}")
        self.log("COMPARATIVE SUMMARY")
        self.log("="*60)
        
        self.log(f"\n{'Model':<30} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP50-95':<12}")
        self.log("-" * 78)
        
        for model_name, metrics in all_metrics.items():
            overall = metrics['overall']
            self.log(f"{model_name:<30} {overall['precision']:<12.4f} {overall['recall']:<12.4f} {overall['mAP50']:<12.4f} {overall['mAP50-95']:<12.4f}")
        
        self.compare_models_visual(models_dict)
        
        with open(self.results_json, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        self.log(f"\n[OK] Full report completed")
        self.log(f"  Results saved to: {self.results_json}")


if __name__ == '__main__':
    evaluator = ModelEvaluator(
        data_yaml='Root Canal.v1i.yolov8/data.yaml',
        output_dir='evaluation_results'
    )
    
    models_to_evaluate = {
        'Student_Without_Distillation': 'dental_yolo.pt',
        'Student_With_Distillation': 'distillation_results/student_with_distillation/weights/best.pt',
        'Teacher_Model': 'distillation_results/teacher_model/weights/best.pt',
    }
    
    evaluator.generate_full_report(
        models_to_evaluate,
        description="Knowledge Distillation Evaluation: Comparing student models with and without distillation knowledge transfer"
    )
