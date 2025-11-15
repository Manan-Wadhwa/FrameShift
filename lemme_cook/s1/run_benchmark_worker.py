#!/usr/bin/env python
#
# CELL 7.8: MAIN DETECTION BENCHMARK HARNESS (WORKER)
#
# This script is designed to be run in parallel.
# It runs *one* combination of (detector + pipeline)
# across *all* test cases and saves a single JSON result.
#

print("--- [Starting] Benchmark Worker ---")

import numpy as np
import time
import json
import argparse
from pathlib import Path
import sys

# Import all benchmark components
from config_and_utils import (
    get_test_cases, 
    standardize_image, 
    load_ground_truth, 
    evaluate_detection,
    visualize_detection_result
)
from preprocessing import get_pipelines
from detectors_group_a import get_detectors as get_group_a
from detectors_group_b import get_detectors as get_group_b
from detectors_group_c import get_detectors as get_group_c

# --- Helper to find a component by name ---
def find_by_name(component_list, name):
    for component in component_list:
        if component.name == name:
            return component
    return None

def main(args):
    
    # --- 1. Load All Components ---
    print("\nLoading all detectors and pipelines...")
    all_pipelines = get_pipelines()
    all_detectors = get_group_a() + get_group_b() + get_group_c()
    all_test_cases = get_test_cases()
    
    # --- 2. Find Requested Job ---
    pipeline = find_by_name(all_pipelines, args.pipeline)
    detector = find_by_name(all_detectors, args.detector)
    
    if not pipeline:
        print(f"Error: Pipeline '{args.pipeline}' not found!")
        print("Available pipelines:", [p.name for p in all_pipelines])
        sys.exit(1)
        
    if not detector:
        print(f"Error: Detector '{args.detector}' not found!")
        print("Available detectors:", [d.name for d in all_detectors])
        sys.exit(1)
        
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING JOB:")
    print(f"  - Detector:   {detector.name}")
    print(f"  - Pipeline:   {pipeline.name}")
    print(f"  - Test Cases: {len(all_test_cases)}")
    print(f"  - Output Dir: {args.output_dir}")
    print(f"{'='*80}")

    # --- 3. Create Output Directory ---
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a sub-directory for visualizations from this worker
    vis_path = output_path / "visualizations"
    vis_path.mkdir(exist_ok=True)

    # --- 4. Run Benchmark Loop ---
    worker_results = []
    
    for i, test_case in enumerate(all_test_cases):
        print(f"\n--- Test Case {i+1}/{len(all_test_cases)}: {test_case['name']} ---")
        
        # Load images
        img1 = standardize_image(test_case['ref_path'])
        img2 = standardize_image(test_case['test_path'])
        gt_mask = load_ground_truth(test_case['mask_path'])
        
        if img1 is None or img2 is None:
            print("  Failed to load images, skipping.")
            continue
            
        # Apply preprocessing
        try:
            img1_proc, img2_proc = pipeline.process(img1, img2)
        except Exception as e:
            print(f"  Preprocessing failed: {e}, using raw.")
            img1_proc, img2_proc = img1, img2
            
        # Run detection
        result = {
            'preprocessing': pipeline.name,
            'detector': detector.name,
            'test_case': test_case['name'],
            'defect_type': test_case['defect_type'],
            'has_ground_truth': gt_mask is not None,
        }

        try:
            predicted_mask, confidence_map, elapsed_time = detector.detect(img1_proc, img2_proc)
            
            # Evaluate
            if gt_mask is not None:
                metrics = evaluate_detection(predicted_mask, gt_mask)
                print(f"  Result: F1={metrics['f1']:.3f}, IoU={metrics['iou']:.3f}, Time={elapsed_time*1000:.0f}ms")
            else:
                metrics = {'iou': np.nan, 'f1': np.nan, 'precision': np.nan, 'recall': np.nan, 'dice': np.nan}
                print(f"  Result: No GT, Time={elapsed_time*1000:.0f}ms")
            
            result.update(metrics)
            result['time'] = elapsed_time
            
            # Save visualization
            if gt_mask is not None:
                vis_filename = f"{test_case['name']}_{pipeline.name}_{detector.name}.png".replace('+', '_')
                save_path = vis_path / vis_filename
                try:
                    visualize_detection_result(
                        img1_proc, img2_proc, predicted_mask, gt_mask,
                        f"{pipeline.name} + {detector.name}",
                        metrics, save_path=save_path
                    )
                except Exception as e:
                    print(f"  Warning: Visualization failed: {e}")

        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            result.update({
                'time': np.nan, 'iou': np.nan, 'f1': np.nan, 
                'precision': np.nan, 'recall': np.nan, 'dice': np.nan,
                'error': str(e)
            })
        
        worker_results.append(result)

    # --- 5. Save Worker's Final Result ---
    job_name = f"results_{pipeline.name}_{detector.name}".replace('+', '_')
    result_file = output_path / f"{job_name}.json"
    
    with open(result_file, 'w') as f:
        json.dump(worker_results, f, indent=2)
        
    print(f"\n{'='*80}")
    print(f"✅ JOB COMPLETE.")
    print(f"  Results for {len(worker_results)} test cases saved to:")
    print(f"  {result_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single benchmark worker.")
    parser.add_argument("-d", "--detector", required=True, help="Name of the detector to run (e.g., 'SSIM+MultiThreshold')")
    parser.add_argument("-p", "--pipeline", required=True, help="Name of the preprocessing pipeline (e.g., 'Median+SIFT')")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory to save the final JSON result file")
    
    args = parser.parse_args()
    main(args)
