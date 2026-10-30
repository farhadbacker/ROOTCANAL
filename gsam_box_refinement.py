import os
from pathlib import Path

def refine_yolo_labels(original_labels_dir, gsam_labels_dir, output_dir, confidence_threshold=0.8):
    """
    Merges original (manual/CLIP) labels with Super-Teacher (GSAM) labels.
    Prioritizes GSAM labels as they are higher fidelity.
    """
    original_path = Path(original_labels_dir)
    gsam_path = Path(gsam_labels_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    gsam_files = list(gsam_path.glob('*.txt'))
    
    print(f"Refining labels using {len(gsam_files)} GSAM detections...")
    
    for gsam_file in gsam_files:
        orig_file = original_path / gsam_file.name
        
        refined_content = []
        if orig_file.exists():
            with open(orig_file, 'r') as f:
                refined_content.append("# Original Labels\n")
                refined_content.extend(f.readlines())
        
        with open(gsam_file, 'r') as f:
            refined_content.append("# GSAM Refinements (High Fidelity)\n")
            refined_content.extend(f.readlines())
            
        with open(output_path / gsam_file.name, 'w') as f:
            f.writelines(refined_content)

if __name__ == "__main__":
    print("GSAM Box Refinement Tool ready.")
