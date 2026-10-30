#!/usr/bin/env python3

import csv
import os
from pathlib import Path
from collections import Counter
import pandas as pd

def main():
    base_dir = Path(__file__).parent.absolute()
    csv_path = base_dir / 'image_labels.csv'
    
    if not csv_path.exists():
        print(f"✗ Labels file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    df = df.drop_duplicates(subset=['image_path'], keep='first')
    
    class_names = {
        0: 'No Endodontic Treatment',
        1: 'Incomplete Endodontic Treatment',
        2: 'Complete Endodontic Treatment',
        3: 'Total Endodontic Failure'
    }
    
    print("\n" + "=" * 80)
    print("LABEL VERIFICATION REPORT")
    print("=" * 80 + "\n")
    
    print(f"Total images labeled: {len(df)}\n")
    
    label_counts = df['label'].value_counts().sort_index()
    print("Label Distribution:")
    print("-" * 60)
    for label in range(4):
        count = label_counts.get(label, 0)
        percentage = (count / len(df)) * 100
        confidence_avg = df[df['label'] == label]['confidence'].mean() if count > 0 else 0
        print(f"  {label}: {class_names[label]:40} {count:3d} ({percentage:5.1f}%) [avg conf: {confidence_avg:.3f}]")
    print("-" * 60 + "\n")
    
    print("Low Confidence Images (confidence < 0.30):")
    print("-" * 60)
    low_conf = df[df['confidence'] < 0.30].sort_values('confidence')
    if len(low_conf) > 0:
        for idx, row in low_conf.head(20).iterrows():
            print(f"  {row['image_path']:20} | Label: {row['label']} ({class_names[row['label']]}) | Conf: {row['confidence']:.4f}")
        if len(low_conf) > 20:
            print(f"  ... and {len(low_conf) - 20} more")
    else:
        print("  None (all confidences >= 0.30)")
    print("-" * 60 + "\n")
    
    df_sorted = df.sort_values(['label', 'confidence'], ascending=[True, False])
    print("Sample Images by Label (Top 5 most confident per category):")
    print("-" * 60)
    for label in range(4):
        label_df = df_sorted[df_sorted['label'] == label].head(5)
        if len(label_df) > 0:
            print(f"\n{label}: {class_names[label]}")
            for idx, row in label_df.iterrows():
                print(f"    {row['image_path']:20} (confidence: {row['confidence']:.4f})")
    print("-" * 60 + "\n")
    
    print("=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Review the labeled images and check if labels are correct")
    print("2. If accuracy is low, consider:")
    print("   - Manually correcting incorrect labels in image_labels.csv")
    print("   - Using a GUI tool to review images (see review_gui.py)")
    print("3. Once satisfied, use the cleaned labels for training")
    print("=" * 80 + "\n")
    
    save_path = base_dir / 'image_labels_clean.csv'
    df.to_csv(save_path, index=False)
    print(f"✓ Clean labels saved to: {save_path}")

if __name__ == "__main__":
    main()
