import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import json
from contourlet_filter import ContourletTransform
import torch
from PIL import Image
import os
from tqdm import tqdm

class ComparativeAnalysis:
    def __init__(self, data_yaml, output_dir='comparative_results'):
        self.data_yaml = data_yaml
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.contourlet = ContourletTransform(num_levels=2, num_directions=8)
        
        self.class_names = [
            'No Endodontic Treatment',
            'Incomplete Endodontic Treatment',
            'Complete Endodontic Treatment',
            'Total Endodontic Failure'
        ]
        
        self.log_file = self.output_dir / 'analysis_report.txt'
        self.results_data = {}
    
    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def analyze_filtering_impact(self, model_path):
        self.log("\n" + "="*60)
        self.log("IMPACT ANALYSIS: Image Filtering (Raw vs Filtered)")
        self.log("="*60)
        
        model = YOLO(model_path)
        
        self.log("\nEvaluating on RAW images...")
        raw_results = model.val(
            data=self.data_yaml,
            device=self.device,
            imgsz=640,
            conf=0.25,
            project=str(self.output_dir),
            name='eval_raw'
        )
        
        raw_metrics = {
            'precision': float(raw_results.box.mp) if hasattr(raw_results.box, 'mp') else 0,
            'recall': float(raw_results.box.mr) if hasattr(raw_results.box, 'mr') else 0,
            'mAP50': float(raw_results.box.map50) if hasattr(raw_results.box, 'map50') else 0,
            'mAP50-95': float(raw_results.box.map) if hasattr(raw_results.box, 'map') else 0,
        }
        
        self.log("\nEvaluating on FILTERED images...")
        filtered_results = model.val(
            data=self.data_yaml,
            device=self.device,
            imgsz=640,
            conf=0.25,
            project=str(self.output_dir),
            name='eval_filtered'
        )
        
        filtered_metrics = {
            'precision': float(filtered_results.box.mp) if hasattr(filtered_results.box, 'mp') else 0,
            'recall': float(filtered_results.box.mr) if hasattr(filtered_results.box, 'mr') else 0,
            'mAP50': float(filtered_results.box.map50) if hasattr(filtered_results.box, 'map50') else 0,
            'mAP50-95': float(filtered_results.box.map) if hasattr(filtered_results.box, 'map') else 0,
        }
        
        self.log(f"\nFiltering Impact Summary:")
        self.log(f"{'Metric':<15} {'Raw':<12} {'Filtered':<12} {'Improvement':<12}")
        self.log("-" * 51)
        
        improvements = {}
        for metric in raw_metrics.keys():
            raw = raw_metrics[metric]
            filt = filtered_metrics[metric]
            imp = ((filt - raw) / max(raw, 1e-6)) * 100
            improvements[metric] = imp
            
            self.log(f"{metric:<15} {raw:<12.4f} {filt:<12.4f} {imp:+.2f}%")
        
        return {
            'raw': raw_metrics,
            'filtered': filtered_metrics,
            'improvements': improvements
        }
    
    def analyze_distillation_impact(self, student_baseline_path, student_distilled_path):
        self.log("\n" + "="*60)
        self.log("IMPACT ANALYSIS: Knowledge Distillation")
        self.log("="*60)
        
        student_baseline = YOLO(student_baseline_path)
        student_distilled = YOLO(student_distilled_path)
        
        self.log("\nEvaluating Student Model WITHOUT Distillation...")
        baseline_results = student_baseline.val(
            data=self.data_yaml,
            device=self.device,
            imgsz=640,
            conf=0.25,
            project=str(self.output_dir),
            name='eval_baseline'
        )
        
        baseline_metrics = {
            'precision': float(baseline_results.box.mp) if hasattr(baseline_results.box, 'mp') else 0,
            'recall': float(baseline_results.box.mr) if hasattr(baseline_results.box, 'mr') else 0,
            'mAP50': float(baseline_results.box.map50) if hasattr(baseline_results.box, 'map50') else 0,
            'mAP50-95': float(baseline_results.box.map) if hasattr(baseline_results.box, 'map') else 0,
        }
        
        self.log("\nEvaluating Student Model WITH Distillation...")
        distilled_results = student_distilled.val(
            data=self.data_yaml,
            device=self.device,
            imgsz=640,
            conf=0.25,
            project=str(self.output_dir),
            name='eval_distilled'
        )
        
        distilled_metrics = {
            'precision': float(distilled_results.box.mp) if hasattr(distilled_results.box, 'mp') else 0,
            'recall': float(distilled_results.box.mr) if hasattr(distilled_results.box, 'mr') else 0,
            'mAP50': float(distilled_results.box.map50) if hasattr(distilled_results.box, 'map50') else 0,
            'mAP50-95': float(distilled_results.box.map) if hasattr(distilled_results.box, 'map') else 0,
        }
        
        self.log(f"\nDistillation Impact Summary:")
        self.log(f"{'Metric':<15} {'Baseline':<12} {'Distilled':<12} {'Improvement':<12}")
        self.log("-" * 51)
        
        improvements = {}
        for metric in baseline_metrics.keys():
            base = baseline_metrics[metric]
            dist = distilled_metrics[metric]
            imp = ((dist - base) / max(base, 1e-6)) * 100
            improvements[metric] = imp
            
            self.log(f"{metric:<15} {base:<12.4f} {dist:<12.4f} {imp:+.2f}%")
        
        return {
            'baseline': baseline_metrics,
            'distilled': distilled_metrics,
            'improvements': improvements
        }
    
    def compare_model_efficiency(self, model_paths_dict):
        self.log("\n" + "="*60)
        self.log("MODEL EFFICIENCY COMPARISON")
        self.log("="*60)
        
        efficiency_data = {}
        
        for model_name, model_path in model_paths_dict.items():
            try:
                model = YOLO(model_path)
                param_count = sum(p.numel() for p in model.model.parameters())
                
                file_size = os.path.getsize(model_path) / (1024 * 1024)
                
                efficiency_data[model_name] = {
                    'parameters': param_count,
                    'file_size_mb': file_size
                }
                
                self.log(f"\n{model_name}:")
                self.log(f"  Parameters: {param_count:,.0f}")
                self.log(f"  File Size: {file_size:.2f} MB")
            except Exception as e:
                self.log(f"[FAILED] Error analyzing {model_name}: {e}")
        
        return efficiency_data
    
    def generate_comparison_plots(self, analysis_results):
        self.log("\nGenerating comparison visualizations...")
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        if 'filtering' in analysis_results:
            ax1 = fig.add_subplot(gs[0, :])
            
            metrics = list(analysis_results['filtering']['raw'].keys())
            raw_vals = [analysis_results['filtering']['raw'][m] for m in metrics]
            filt_vals = [analysis_results['filtering']['filtered'][m] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, raw_vals, width, label='Raw Images', color='#FF6B6B')
            ax1.bar(x + width/2, filt_vals, width, label='Filtered Images', color='#4ECDC4')
            
            ax1.set_xlabel('Metric', fontweight='bold')
            ax1.set_ylabel('Score', fontweight='bold')
            ax1.set_title('Impact of Image Filtering on Model Performance', fontweight='bold', fontsize=12)
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
        
        if 'distillation' in analysis_results:
            ax2 = fig.add_subplot(gs[1, :])
            
            metrics = list(analysis_results['distillation']['baseline'].keys())
            baseline_vals = [analysis_results['distillation']['baseline'][m] for m in metrics]
            distilled_vals = [analysis_results['distillation']['distilled'][m] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax2.bar(x - width/2, baseline_vals, width, label='Student (No Distillation)', color='#95E1D3')
            ax2.bar(x + width/2, distilled_vals, width, label='Student (With Distillation)', color='#F38181')
            
            ax2.set_xlabel('Metric', fontweight='bold')
            ax2.set_ylabel('Score', fontweight='bold')
            ax2.set_title('Impact of Knowledge Distillation on Student Model', fontweight='bold', fontsize=12)
            ax2.set_xticks(x)
            ax2.set_xticklabels(metrics)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        
        if 'efficiency' in analysis_results:
            ax3 = fig.add_subplot(gs[2, 0])
            ax4 = fig.add_subplot(gs[2, 1])
            
            models = list(analysis_results['efficiency'].keys())
            params = [analysis_results['efficiency'][m]['parameters'] / 1e6 for m in models]
            sizes = [analysis_results['efficiency'][m]['file_size_mb'] for m in models]
            
            colors = ['#A8E6CF', '#FFD3B6', '#FFAAA5'][:len(models)]
            
            ax3.barh(models, params, color=colors)
            ax3.set_xlabel('Parameters (Millions)', fontweight='bold')
            ax3.set_title('Model Size Comparison', fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
            
            ax4.barh(models, sizes, color=colors)
            ax4.set_xlabel('File Size (MB)', fontweight='bold')
            ax4.set_title('Model File Size Comparison', fontweight='bold')
            ax4.grid(axis='x', alpha=0.3)
        
        plt.savefig(self.output_dir / 'comprehensive_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log("[OK] Comparison plots saved")
    
    def run_full_analysis(self, model_config):
        self.log("="*60)
        self.log("COMPREHENSIVE COMPARATIVE ANALYSIS")
        self.log("="*60)
        self.log("\nAnalyzing Knowledge Distillation Implementation")
        self.log("Comparing: Filtering Impact, Distillation Impact, Model Efficiency\n")
        
        analysis_results = {}
        
        if 'filtering_test_model' in model_config:
            try:
                analysis_results['filtering'] = self.analyze_filtering_impact(
                    model_config['filtering_test_model']
                )
            except Exception as e:
                self.log(f"[WARNING] Filtering analysis failed: {e}")
        
        if 'student_baseline' in model_config and 'student_distilled' in model_config:
            try:
                analysis_results['distillation'] = self.analyze_distillation_impact(
                    model_config['student_baseline'],
                    model_config['student_distilled']
                )
            except Exception as e:
                self.log(f"[WARNING] Distillation analysis failed: {e}")
        
        if 'model_paths' in model_config:
            try:
                analysis_results['efficiency'] = self.compare_model_efficiency(
                    model_config['model_paths']
                )
            except Exception as e:
                self.log(f"[WARNING] Efficiency analysis failed: {e}")
        
        self.generate_comparison_plots(analysis_results)
        
        self.log(f"\n{'='*60}")
        self.log("[OK] ANALYSIS COMPLETED")
        self.log(f"Results saved to: {self.output_dir}")
        self.log("="*60)
        
        with open(self.output_dir / 'analysis_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        return analysis_results


if __name__ == '__main__':
    analyzer = ComparativeAnalysis(
        data_yaml='Root Canal.v1i.yolov8/data.yaml',
        output_dir='comparative_results'
    )
    
    model_config = {
        'filtering_test_model': 'dental_yolo_roboflow_filtered.pt',
        'student_baseline': 'dental_yolo.pt',
        'student_distilled': 'distillation_results/student_with_distillation/weights/best.pt',
        'model_paths': {
            'Student (YOLOv8n)': 'dental_yolo.pt',
            'Teacher (YOLOv8m)': 'distillation_results/teacher_model/weights/best.pt',
            'Student Distilled': 'distillation_results/student_with_distillation/weights/best.pt',
        }
    }
    
    analyzer.run_full_analysis(model_config)
