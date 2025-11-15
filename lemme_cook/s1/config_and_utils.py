#!/usr/bin/env python
#
# CELL 7.2: DETECTION CREATE TEST DATASET
# CELL 7.7: EVALUATION METRICS
# This file also includes missing helper functions like standardize_image
#

print("--- [Loading] Config, Helpers, and Test Cases ---")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time

#%%
#
# MANDATORY SETUP (Standardization & Helpers from user's example)
#

TARGET_SIZE = (1024, 1024)

def standardize_image(image_path):
    """Loads, resizes, and normalizes an image."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    
    # 1. Resize
    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    
    # 2. Normalize Bit Depth (to 0.0 - 1.0 float)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    return img_normalized

def load_ground_truth(mask_path):
    """Loads and standardizes a ground truth mask."""
    if mask_path is None:
        return None
    
    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"Warning: Could not load mask at {mask_path}")
        return None
    
    gt_mask = cv2.resize(gt_mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    return gt_mask

#%%
#
# CELL 7.2: DETECTION CREATE TEST DATASET
#

print("\n--- [CELL 7.2] Creating Detection Test Dataset ---")
# We'll create a comprehensive test set combining:
# 1. MVTec AD samples (you already have)
# 2. Synthetic change samples (if added)
# 3. CARDD samples (as requested)

# Define test cases structure
DETECTION_TEST_CASES = []

# Helper to add test cases
def add_test_case(name, ref_path, test_path, mask_path, defect_type):
    """Add a test case to the benchmark"""
    # Check for paths during loading, not definition
    DETECTION_TEST_CASES.append({
        'name': name,
        'ref_path': ref_path,
        'test_path': test_path,
        'mask_path': mask_path,
        'defect_type': defect_type
    })

# 
# ==============================================================================
# IMPORTANT: UPDATE THESE PATHS TO MATCH YOUR DATASET LOCATIONS
# ==============================================================================
#

# Add MVTec samples
# Bottle structural defect
add_test_case(
    'bottle_broken_small',
    'datasets/mvtec_ad/bottle/train/good/000.png',
    'datasets/mvtec_ad/bottle/test/broken_small/000.png',
    'datasets/mvtec_ad/bottle/ground_truth/broken_small/000_mask.png',
    'structural'
)

# Carpet texture defect
add_test_case(
    'carpet_hole',
    'datasets/mvtec_ad/carpet/train/good/000.png',
    'datasets/mvtec_ad/carpet/test/hole/000.png',
    'datasets/mvtec_ad/carpet/ground_truth/hole/000_mask.png',
    'texture'
)

# Hazelnut - crack (your color space winner)
add_test_case(
    'hazelnut_crack',
    'datasets/mvtec_ad/hazelnut/train/good/000.png',
    'datasets/mvtec_ad/hazelnut/test/crack/000.png',
    'datasets/mvtec_ad/hazelnut/ground_truth/crack/000_mask.png',
    'crack'
)

# Grid alignment test case
add_test_case(
    'grid_bent',
    'datasets/mvtec_ad/grid/train/good/000.png',
    'datasets/mvtec_ad/grid/test/bent/000.png',
    'datasets/mvtec_ad/grid/ground_truth/bent/000_mask.png',
    'deformation'
)

#
# ==============================================================================
# ADD YOUR CARDD DATASET SAMPLES HERE
# ==============================================================================
#

# FOR CARDD DATASET:
# After downloading CARDD:
# In Cell 7.2, add CARDD test cases:
cardd_path = 'datasets/cardd'
add_test_case(
    'cardd_building_change',
    f'{cardd_path}/reference/img1.png',
    f'{cardd_path}/query/img2.png',
    f'{cardd_path}/gt/mask.png',
    'building_change'
)

print(f"Loaded {len(DETECTION_TEST_CASES)} test cases.")
for tc in DETECTION_TEST_CASES:
    has_mask = "✓" if tc['mask_path'] else "x"
    print(f"  [{has_mask}] {tc['name']} ({tc['defect_type']})")


def get_test_cases():
    """Returns the globally defined list of test cases."""
    return DETECTION_TEST_CASES

def get_normal_samples():
    """Collects normal samples for training."""
    print("\nCollecting normal samples for training...")
    normal_samples = []
    # Use first 3 reference images
    for test_case in DETECTION_TEST_CASES[:3]: 
        ref_img = standardize_image(test_case['ref_path'])
        if ref_img is not None:
            normal_samples.append(ref_img)
    print(f"Found {len(normal_samples)} normal samples.")
    return normal_samples

#%%
#
# CELL 7.7: EVALUATION METRICS
#
print("\n--- [CELL 7.7] Evaluation Metrics ---")

def evaluate_detection(predicted_mask, ground_truth_mask):
    """Compute comprehensive evaluation metrics"""
    
    # Flatten masks
    pred_flat = predicted_mask.flatten().astype(bool)
    gt_flat = ground_truth_mask.flatten().astype(bool)
    
    # Handle edge cases
    if gt_flat.sum() == 0 and pred_flat.sum() == 0:
        # Both empty - perfect match
        return {'iou': 1.0, 'f1': 1.0, 'precision': 1.0, 'recall': 1.0, 'dice': 1.0}
    
    elif gt_flat.sum() == 0:
        # No ground truth but predicted something - all false positives
        return {'iou': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 1.0, 'dice': 0.0}
        
    elif pred_flat.sum() == 0:
        # Ground truth exists but predicted nothing - all false negatives
        return {'iou': 0.0, 'f1': 0.0, 'precision': 1.0, 'recall': 0.0, 'dice': 0.0}

    # Standard metrics
    intersection = np.logical_and(pred_flat, gt_flat).sum()
    union = np.logical_or(pred_flat, gt_flat).sum()
    
    tp = float(intersection)
    fp = float(pred_flat.sum() - intersection)
    fn = float(gt_flat.sum() - intersection)
    
    iou = tp / union if union > 0 else 0.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    dice = 2 * intersection / (pred_flat.sum() + gt_flat.sum()) if (pred_flat.sum() + gt_flat.sum()) > 0 else 0.0
    
    return {
        'iou': float(iou),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'dice': float(dice)
    }

def visualize_detection_result(img1, img2, predicted_mask,
                             ground_truth_mask, method_name, metrics,
                             save_path=None):
    """Create comprehensive visualization"""
    
    # Ensure proper format for visualization
    img1_vis = (img1 * 255).astype(np.uint8) if img1.max() <= 1.0 else img1.astype(np.uint8)
    img2_vis = (img2 * 255).astype(np.uint8) if img2.max() <= 1.0 else img2.astype(np.uint8)

    # Convert to RGB if grayscale
    if len(img1_vis.shape) == 2:
        img1_vis = cv2.cvtColor(img1_vis, cv2.COLOR_GRAY2BGR)
    if len(img2_vis.shape) == 2:
        img2_vis = cv2.cvtColor(img2_vis, cv2.COLOR_GRAY2BGR)

    # Create overlay
    overlay = img2_vis.copy()
    overlay[predicted_mask > 0] = [0, 255, 0]  # Green for prediction
    
    comparison = np.zeros_like(img2_vis)

    if ground_truth_mask is not None:
        # Red for ground truth
        overlay[ground_truth_mask > 0] = [255, 0, 0]
        # Combine (Blue/Purple for TP)
        overlay[np.logical_and(predicted_mask > 0, ground_truth_mask > 0)] = [255, 0, 255]

        tp = np.logical_and(predicted_mask > 0, ground_truth_mask > 0)
        fp = np.logical_and(predicted_mask > 0, ground_truth_mask == 0)
        fn = np.logical_and(predicted_mask == 0, ground_truth_mask > 0)
        
        comparison[tp] = [0, 255, 0]    # Green - True Positive
        comparison[fp] = [255, 255, 0]  # Yellow - False Positive
        comparison[fn] = [255, 0, 0]    # Red - False Negative

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(img1_vis, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Reference Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(img2_vis, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Test Image')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(predicted_mask, cmap='hot')
    axes[0, 2].set_title('Predicted Mask')
    axes[0, 2].axis('off')
    
    if ground_truth_mask is not None:
        axes[1, 0].imshow(ground_truth_mask, cmap='hot')
        axes[1, 0].set_title('Ground Truth Mask')
    else:
        axes[1, 0].text(0.5, 0.5, 'No Ground Truth', ha='center', va='center', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Overlay (Green=Pred, Red=GT, Purple=TP)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Comparison (Green=TP, Yellow=FP, Red=FN)')
    axes[1, 2].axis('off')
    
    # Add metrics text
    metrics_text = f"{method_name}\n"
    if metrics:
        metrics_text += f"F1: {metrics.get('f1', 0):.3f} | IoU: {metrics.get('iou', 0):.3f}\n"
        metrics_text += f"Prec: {metrics.get('precision', 0):.3f} | Rec: {metrics.get('recall', 0):.3f}"
    fig.suptitle(metrics_text, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

print("✅ Config, helpers, and test cases ready.")
