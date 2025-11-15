#!/usr/bin/env python
#
# CELL 7.6: GROUP C ZERO-SHOT DETECTORS
#

print("--- [Loading] Detectors Group C: Zero-Shot ---")

import cv2
import numpy as np
import time

from detectors_base import DetectionMethod

#%%
#
# CELL 7.6: GROUP C ZERO-SHOT DETECTORS
#
print("\n--- [CELL 7.6] Detection Methods - Group C (Zero-Shot) ---")

#
# --- METHOD 10: Template Matching Ensemble ---
#
class TemplateMatchingDetector(DetectionMethod):
    """Multi-scale template matching (zero-shot)"""
    def __init__(self):
        super().__init__("TemplateMatch-ZS")

    def detect(self, img1, img2):
        start_time = time.time()
        img1_gray = self._ensure_grayscale(img1)
        img2_gray = self._ensure_grayscale(img2)
        img1_gray, img2_gray = self._ensure_same_shape(img1_gray, img2_gray)

        # Convert to 8-bit
        img1_8bit = (img1_gray * 255).astype(np.uint8) if img1_gray.max() <= 1.0 else img1_gray.astype(np.uint8)
        img2_8bit = (img2_gray * 255).astype(np.uint8) if img2_gray.max() <= 1.0 else img2_gray.astype(np.uint8)

        patch_size = 64
        stride = 32
        h, w = img1_8bit.shape
        
        mismatch_map = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h - patch_size, stride):
            for x in range(0, w - patch_size, stride):
                template = img1_8bit[y:y+patch_size, x:x+patch_size]
                
                # Define search region with a small margin
                y_start, y_end = max(0, y - 16), min(h, y + patch_size + 16)
                x_start, x_end = max(0, x - 16), min(w, x + patch_size + 16)
                search_region = img2_8bit[y_start:y_end, x_start:x_end]

                if template.shape[0] > search_region.shape[0] or template.shape[1] > search_region.shape[1]:
                    continue  # Patch is invalid

                result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                mismatch_score = 1.0 - max_val
                
                # Update mismatch map
                current_scores = mismatch_map[y:y+patch_size, x:x+patch_size]
                new_scores = np.maximum(current_scores, mismatch_score)
                mismatch_map[y:y+patch_size, x:x+patch_size] = new_scores
        
        binary_mask = (mismatch_map > 0.3).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        elapsed = time.time() - start_time
        return binary_mask, mismatch_map, elapsed

#
# --- METHOD 11: Statistical Outlier Detection ---
#
class StatisticalOutlierDetector(DetectionMethod):
    """Z-score based outlier detection (zero-shot)"""
    def __init__(self):
        super().__init__("StatOutlier-ZS")

    def detect(self, img1, img2):
        start_time = time.time()
        img1_gray = self._ensure_grayscale(img1)
        img2_gray = self._ensure_grayscale(img2)
        img1_gray, img2_gray = self._ensure_same_shape(img1_gray, img2_gray)

        diff = np.abs(img1_gray - img2_gray)
        
        # Using a blurred version for mean/std is faster than a sliding window
        window_size = 31
        mean = cv2.blur(diff, (window_size, window_size))
        
        # (x - mean)^2
        sq_diff = (diff - mean)**2
        # mean of (x - mean)^2 = variance
        variance = cv2.blur(sq_diff, (window_size, window_size))
        std = np.sqrt(variance)
        
        z_score_map = np.zeros_like(diff)
        # Avoid division by zero
        valid_std = std > 1e-6
        z_score_map[valid_std] = np.abs(diff[valid_std] - mean[valid_std]) / std[valid_std]

        binary_mask = (z_score_map > 2.0).astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        max_z = z_score_map.max()
        confidence_map = z_score_map / (max_z + 1e-8) if max_z > 0 else z_score_map
        
        elapsed = time.time() - start_time
        return binary_mask, confidence_map, elapsed

#
# --- METHOD 12: Frequency Domain Analysis ---
#
class FrequencyDomainDetector(DetectionMethod):
    """FFT-based change detection (zero-shot)"""
    def __init__(self):
        super().__init__("Frequency-ZS")

    def detect(self, img1, img2):
        start_time = time.time()
        img1_gray = self._ensure_grayscale(img1)
        img2_gray = self._ensure_grayscale(img2)
        img1_gray, img2_gray = self._ensure_same_shape(img1_gray, img2_gray)

        f1 = np.fft.fft2(img1_gray)
        f2 = np.fft.fft2(img2_gray)
        mag1 = np.abs(f1)
        mag2 = np.abs(f2)
        
        mag_diff = np.abs(mag1 - mag2)
        spatial_diff = np.abs(np.fft.ifft2(mag_diff))
        
        min_val = spatial_diff.min()
        max_val = spatial_diff.max()
        spatial_diff_norm = (spatial_diff - min_val) / (max_val - min_val + 1e-8)

        binary_mask = (spatial_diff_norm > 0.4).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        elapsed = time.time() - start_time
        return binary_mask, spatial_diff_norm.astype(np.float32), elapsed

#
# Create instances
#
ZEROSHOT_DETECTORS = [
    TemplateMatchingDetector(),
    StatisticalOutlierDetector(),
    FrequencyDomainDetector(),
]

def get_detectors():
    return ZEROSHOT_DETECTORS

print(f"Created {len(ZEROSHOT_DETECTORS)} zero-shot detectors:")
for detector in ZEROSHOT_DETECTORS:
    print(f"  - {detector.name}")

print("✅ Detectors Group C ready.")
