from evaluate_models import ModelEvaluator
import json

def main():
    evaluator = ModelEvaluator(
        data_yaml=r'c:\Users\PRO\Desktop\Root Canal\Root-canal-AI-diagnostics\Root Canal.v1i.yolov8\data.yaml',
        output_dir='v2_medical_eval'
    )
    
    model_path = r'c:\Users\PRO\Desktop\Root Canal\Root-canal-AI-diagnostics\distillation_results_v2\student_kd_true\weights\best.pt'
    
    print(f"Calculating Medical Metrics for: {model_path}")
    
    # helper to print dict nicely
    def print_metrics(metrics):
        print(json.dumps(metrics, indent=2))

    try:
        med_metrics = evaluator.compute_medical_metrics(model_path)
        print("\nSUCCESS: Medical Metrics Computed")
        print_metrics(med_metrics)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
