import os
import cv2
import numpy as np
from pathlib import Path
from contourlet_filter import ContourletTransform
import shutil


def preprocess_images_with_contourlet(
    input_dir="dataset/images",
    output_dir="dataset/images_contourlet",
    num_levels=2,
    num_directions=8,
    use_original=False
):
    """
    Apply Contourlet transform to all images in a directory
    
    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save filtered images
        num_levels: Number of pyramid levels for Contourlet
        num_directions: Number of directional filters
        use_original: If True, keep original; if False, replace with filtered
    
    Returns:
        Dictionary with processing statistics
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not use_original:
        output_path = input_path
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    ct = ContourletTransform(num_levels=num_levels, num_directions=num_directions)
    
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")) + \
                  list(input_path.glob("*.JPG")) + list(input_path.glob("*.PNG"))
    
    processed_count = 0
    failed_count = 0
    stats = {
        'processed': 0,
        'failed': 0,
        'failed_files': []
    }
    
    print(f"Found {len(image_files)} images to process")
    print(f"Processing with Contourlet Transform (levels={num_levels}, directions={num_directions})")
    print(f"Output directory: {output_path}")
    
    for idx, image_file in enumerate(image_files, 1):
        try:
            image = cv2.imread(str(image_file))
            
            if image is None:
                print(f"  [{idx}/{len(image_files)}] ✗ Failed to read: {image_file.name}")
                failed_count += 1
                stats['failed_files'].append(str(image_file.name))
                continue
            
            filtered = ct.apply(image)
            
            output_file = output_path / image_file.name
            cv2.imwrite(str(output_file), filtered)
            
            if (idx) % 10 == 0:
                print(f"  [{idx}/{len(image_files)}] ✓ Processed: {image_file.name}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"  [{idx}/{len(image_files)}] ✗ Error processing {image_file.name}: {str(e)}")
            failed_count += 1
            stats['failed_files'].append(str(image_file.name))
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {processed_count}/{len(image_files)}")
    print(f"Failed: {failed_count}/{len(image_files)}")
    
    if stats['failed_files']:
        print(f"\nFailed files:")
        for f in stats['failed_files']:
            print(f"  - {f}")
    
    stats['processed'] = processed_count
    stats['failed'] = failed_count
    
    return stats


def backup_original_dataset(original_dir="dataset/images", backup_dir="dataset/images_original"):
    """
    Create a backup of original images before applying filters
    """
    original_path = Path(original_dir)
    backup_path = Path(backup_dir)
    
    if backup_path.exists():
        print(f"Backup already exists at {backup_dir}")
        return False
    
    print(f"Creating backup of original images...")
    backup_path.mkdir(parents=True, exist_ok=True)
    
    image_files = list(original_path.glob("*.*"))
    for image_file in image_files:
        if image_file.is_file():
            shutil.copy2(image_file, backup_path / image_file.name)
    
    print(f"Backup created at {backup_dir}")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Apply Contourlet Transform to dental X-ray images"
    )
    parser.add_argument(
        "--input_dir",
        default="dataset/images",
        help="Input directory with original images (default: dataset/images)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of original images before filtering"
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=2,
        help="Number of pyramid levels (default: 2)"
    )
    parser.add_argument(
        "--directions",
        type=int,
        default=8,
        help="Number of directional filters (default: 8)"
    )
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Don't replace original images; save to separate directory"
    )
    
    args = parser.parse_args()
    
    if args.backup:
        backup_original_dataset(args.input_dir)
    
    stats = preprocess_images_with_contourlet(
        input_dir=args.input_dir,
        num_levels=args.levels,
        num_directions=args.directions,
        use_original=args.no_replace
    )
    
    if stats['processed'] > 0:
        print(f"\n✓ Successfully preprocessed {stats['processed']} images with Contourlet Transform")


if __name__ == "__main__":
    main()
