#!/usr/bin/env python3
"""
Knowledge Distillation Pipeline Runner
======================================

This script orchestrates the complete knowledge distillation pipeline for
the root canal failure diagnosis system.

Pipeline Stages:
1. Teacher Model Training (YOLOv8m) - Large model with full capacity
2. Student Model Training with Distillation (YOLOv8n) - Smaller model learning from teacher
3. Evaluation & Metrics - Comprehensive model comparison
4. Comparative Analysis - Filtering impact, distillation impact, efficiency

Usage:
  python run_distillation_pipeline.py [--stage STAGE] [--skip-training]
  
  Stages:
    all               - Run complete pipeline (default)
    teacher           - Train teacher model only
    student           - Train student model with distillation only
    evaluate          - Evaluate trained models
    analyze           - Run comparative analysis
    inference         - Start inference server

Examples:
  python run_distillation_pipeline.py --stage all
  python run_distillation_pipeline.py --stage teacher
  python run_distillation_pipeline.py --stage evaluate
  python run_distillation_pipeline.py --stage inference
"""

import argparse
import sys
from pathlib import Path
import subprocess
import time
from knowledge_distillation import DistillationTrainer
from evaluate_models import ModelEvaluator
from comparative_analysis import ComparativeAnalysis

class PipelineRunner:
    def __init__(self, data_yaml='Root Canal.v1i.yolov8/data.yaml'):
        self.data_yaml = data_yaml
        self.start_time = time.time()
        self.log_file = Path('distillation_pipeline.log')
        
        self.print_header()
    
    def print_header(self):
        print("\n" + "="*70)
        print("KNOWLEDGE DISTILLATION PIPELINE FOR ROOT CANAL DIAGNOSIS")
        print("="*70)
        print("\nPaper: 'Fusion of Image Filtering and Knowledge-Distilled YOLO")
        print("       Models for Root Canal Failure Diagnosis'")
        print("\nImplementation includes:")
        print("  [*] Contourlet Image Filtering")
        print("  [*] Teacher-Student Architecture")
        print("  [*] Knowledge Distillation with Temperature Scaling")
        print("  [*] Comprehensive Evaluation Metrics")
        print("  [*] Comparative Analysis")
        print("="*70 + "\n")
    
    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def run_stage(self, stage_name, stage_func):
        self.log(f"\n{'='*70}")
        self.log(f"EXECUTING STAGE: {stage_name.upper()}")
        self.log(f"{'='*70}\n")
        
        try:
            stage_func()
            self.log(f"\n[OK] Stage '{stage_name}' completed successfully\n")
            return True
        except Exception as e:
            self.log(f"\n[FAILED] Stage '{stage_name}' failed: {e}\n")
            import traceback
            traceback.print_exc()
            return False
    
    def stage_teacher_training(self):
        """Train the teacher model (YOLOv8m)"""
        trainer = DistillationTrainer(
            data_yaml=self.data_yaml,
            output_dir='distillation_results'
        )
        
        self.log("Training teacher model (YOLOv8m)...")
        self.log("This is a larger model that serves as knowledge source.\n")
        
        teacher_path = trainer.train_teacher_model(epochs=100, imgsz=640)
        
        self.log(f"\nTeacher model saved to: {teacher_path}")
        return teacher_path
    
    def stage_student_training(self):
        """Train the student model with knowledge distillation"""
        trainer = DistillationTrainer(
            data_yaml=self.data_yaml,
            output_dir='distillation_results'
        )
        
        teacher_path = 'distillation_results/teacher_model/weights/best.pt'
        
        self.log("Training student model with knowledge distillation...")
        self.log("Student model (YOLOv8n) learns from teacher through distillation.\n")
        
        student_path = trainer.train_student_with_distillation(
            teacher_path,
            epochs=100,
            imgsz=640
        )
        
        self.log(f"\nStudent model saved to: {student_path}")
        return student_path
    
    def stage_evaluation(self):
        """Evaluate and compare all models"""
        evaluator = ModelEvaluator(
            data_yaml=self.data_yaml,
            output_dir='evaluation_results'
        )
        
        self.log("Evaluating all models with comprehensive metrics...\n")
        
        models_to_evaluate = {
            'Student_Baseline': 'dental_yolo.pt',
            'Student_Distilled': 'distillation_results/student_with_distillation/weights/best.pt',
            'Teacher_Model': 'distillation_results/teacher_model/weights/best.pt',
        }
        
        evaluator.generate_full_report(
            models_to_evaluate,
            description="Knowledge Distillation Evaluation Report"
        )
        
        self.log(f"\nEvaluation results saved to: {Path('evaluation_results').resolve()}")
    
    def stage_comparative_analysis(self):
        """Run comprehensive comparative analysis"""
        analyzer = ComparativeAnalysis(
            data_yaml=self.data_yaml,
            output_dir='comparative_results'
        )
        
        self.log("Running comprehensive comparative analysis...\n")
        
        model_config = {
            'filtering_test_model': 'dental_yolo_roboflow_filtered.pt',
            'student_baseline': 'dental_yolo.pt',
            'student_distilled': 'distillation_results/student_with_distillation/weights/best.pt',
            'model_paths': {
                'Student_Baseline': 'dental_yolo.pt',
                'Teacher_Model': 'distillation_results/teacher_model/weights/best.pt',
                'Student_Distilled': 'distillation_results/student_with_distillation/weights/best.pt',
            }
        }
        
        analyzer.run_full_analysis(model_config)
        
        self.log(f"\nAnalysis results saved to: {Path('comparative_results').resolve()}")
    
    def stage_inference_server(self):
        """Start the inference server"""
        self.log("Starting inference server...\n")
        
        self.log("Loading model configuration from model_config.json")
        self.log("Server will use distilled student model by default\n")
        
        self.log("Available API endpoints:")
        self.log("  GET  /health              - Server health and model status")
        self.log("  GET  /models              - List all available models")
        self.log("  POST /models/<model_name> - Switch to different model")
        self.log("  POST /detect              - Run detection on image\n")
        
        self.log("Starting Flask server on http://0.0.0.0:5000\n")
        self.log("="*70 + "\n")
        
        try:
            subprocess.run([sys.executable, 'inference_server.py'], check=True)
        except KeyboardInterrupt:
            self.log("\nServer stopped by user")
        except Exception as e:
            self.log(f"Error starting server: {e}")
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        stages = [
            ("Teacher Training", self.stage_teacher_training),
            ("Student Training", self.stage_student_training),
            ("Model Evaluation", self.stage_evaluation),
            ("Comparative Analysis", self.stage_comparative_analysis),
        ]
        
        completed = 0
        failed = 0
        
        for stage_name, stage_func in stages:
            if self.run_stage(stage_name, stage_func):
                completed += 1
            else:
                failed += 1
        
        self.print_summary(completed, failed, len(stages))
    
    def print_summary(self, completed, failed, total):
        elapsed = time.time() - self.start_time
        
        self.log("\n" + "="*70)
        self.log("PIPELINE EXECUTION SUMMARY")
        self.log("="*70)
        self.log(f"\nCompleted: {completed}/{total} stages")
        self.log(f"Failed: {failed}/{total} stages")
        self.log(f"Total time: {elapsed/60:.1f} minutes\n")
        
        self.log("Results and logs saved to:")
        self.log(f"  - distillation_results/     (Training outputs)")
        self.log(f"  - evaluation_results/       (Model evaluation metrics)")
        self.log(f"  - comparative_results/      (Analysis and comparisons)")
        self.log(f"  - {self.log_file}         (Pipeline log)")
        
        self.log("\nNext steps:")
        self.log("  1. Review metrics in evaluation_results/")
        self.log("  2. Check comparative_results/ for visualizations")
        self.log("  3. Update model_config.json if needed")
        self.log("  4. Run inference server: python inference_server.py")
        
        self.log("\n" + "="*70 + "\n")
        
        if failed > 0:
            print(f"[WARNING] {failed} stage(s) failed. Check logs for details.")
            return False
        else:
            print("[OK] All stages completed successfully!")
            return True


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--stage',
        choices=['all', 'teacher', 'student', 'evaluate', 'analyze', 'inference'],
        default='all',
        help='Pipeline stage to run (default: all)'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training stages and go directly to evaluation'
    )
    
    parser.add_argument(
        '--teacher-epochs',
        type=int,
        default=100,
        help='Number of epochs for teacher training (default: 100)'
    )
    
    parser.add_argument(
        '--student-epochs',
        type=int,
        default=100,
        help='Number of epochs for student training (default: 100)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='Root Canal.v1i.yolov8/data.yaml',
        help='Path to data.yaml file (default: Root Canal.v1i.yolov8/data.yaml)'
    )
    
    args = parser.parse_args()
    
    runner = PipelineRunner(data_yaml=args.data)
    
    try:
        if args.stage == 'all':
            runner.run_full_pipeline()
        elif args.stage == 'teacher':
            runner.run_stage("Teacher Training", runner.stage_teacher_training)
        elif args.stage == 'student':
            runner.run_stage("Student Training", runner.stage_student_training)
        elif args.stage == 'evaluate':
            runner.run_stage("Model Evaluation", runner.stage_evaluation)
        elif args.stage == 'analyze':
            runner.run_stage("Comparative Analysis", runner.stage_comparative_analysis)
        elif args.stage == 'inference':
            runner.run_stage("Inference Server", runner.stage_inference_server)
    
    except KeyboardInterrupt:
        runner.log("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        runner.log(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
