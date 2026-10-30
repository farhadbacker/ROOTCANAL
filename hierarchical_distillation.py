import os
import torch
from ultralytics import YOLO
from knowledge_distillation import DistillationTrainer as BaseTrainer
from advanced_distillation import DistillationTrainer as CustomTrainer
from pathlib import Path

class HierarchicalDistillation:
    def __init__(self, data_yaml, output_dir='hierarchical_results'):
        self.data_yaml = data_yaml
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = 0 if torch.cuda.is_available() else 'cpu'

    def log(self, msg):
        print(f"\n[HIERARCHY] {msg}")

    def run_pipeline(self, epochs=10):
        """
        The Model Ladder:
        YOLOv8m (Teacher) -> YOLOv8n (Intermediate) -> YOLOv5n (Final)
        """
        
        # 1. Step 1: Teacher Training (YOLOv8m)
        self.log("Step 1: Training Teacher Model (YOLOv8m)...")
        teacher = YOLO('yolov8m.pt')
        teacher.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=640,
            device=self.device,
            project=str(self.output_dir),
            name='1_teacher_v8m',
            exist_ok=True
        )
        teacher_path = self.output_dir / '1_teacher_v8m' / 'weights' / 'best.pt'

        # 2. Step 2: Intermediate Distillation (v8m -> v8n)
        self.log("Step 2: Distilling YOLOv8m (Teacher) to YOLOv8n (Intermediate)...")
        intermediate_trainer = CustomTrainer(
            teacher_model_path=str(teacher_path),
            overrides={
                'data': self.data_yaml,
                'epochs': epochs,
                'imgsz': 640,
                'device': self.device,
                'project': str(self.output_dir),
                'name': '2_intermediate_v8n',
                'model': 'yolov8n.pt',
                'exist_ok': True
            }
        )
        intermediate_trainer.train()
        intermediate_path = self.output_dir / '2_intermediate_v8n' / 'weights' / 'best.pt'

        # 3. Step 3: Final Distillation (v8n -> v5n)
        self.log("Step 3: Distilling YOLOv8n (Intermediate) to YOLOv5n (Final Target)...")
        # NOTE: YOLOv5n in ultralytics package usually has compatible head structure 
        # for training-time logits if using same imgsz.
        final_trainer = CustomTrainer(
            teacher_model_path=str(intermediate_path),
            overrides={
                'data': self.data_yaml,
                'epochs': epochs,
                'imgsz': 640,
                'device': self.device,
                'project': str(self.output_dir),
                'name': '3_final_v5n',
                'model': 'yolov5n.pt', # Using YOLOv5 as target as per paper
                'exist_ok': True
            }
        )
        final_trainer.train()
        final_path = self.output_dir / '3_final_v5n' / 'weights' / 'best.pt'

        self.log(f"Hierarchical Pipeline Complete!")
        self.log(f"Final High-Precision Model: {final_path}")
        
        return str(final_path)

if __name__ == "__main__":
    # Test with small epochs
    pipeline = HierarchicalDistillation(data_yaml='Root Canal.v1i.yolov8/data.yaml')
    pipeline.run_pipeline(epochs=100)
