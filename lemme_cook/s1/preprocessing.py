#!/usr/bin/env python
#
# CELL 7.3: DETECTION PREPROCESSING PIPELINE
#

print("--- [Loading] Preprocessing Pipeline ---")

import cv2
import numpy as np
import time
from pathlib import Path

# Ensure opencv-contrib-python is installed for SIFT
# pip install opencv-contrib-python

#%%
#
# CELL 7.3: DETECTION PREPROCESSING PIPELINE
#
print("\n--- [CELL 7.3] Detection Preprocessing Pipeline ---")

class PreprocessingPipeline:
    """Apply your winning preprocessing methods"""
    def __init__(self, use_median=True, use_sift=True, use_blue_channel=False):
        self.use_median = use_median
        self.use_sift = use_sift
        self.use_blue_channel = use_blue_channel
        self.name = self._generate_name()

    def _generate_name(self):
        parts = []
        if self.use_median:
            parts.append("Median")
        if self.use_sift:
            parts.append("SIFT")
        if self.use_blue_channel:
            parts.append("Blue")
        return "+".join(parts) if parts else "Raw"

    def process(self, img1, img2):
        """Apply preprocessing pipeline"""
        
        # Ensure images are standardized (float 0-1)
        if img1.max() > 1.0:
            img1 = img1.astype(np.float32) / 255.0
        if img2.max() > 1.0:
            img2 = img2.astype(np.float32) / 255.0

        # Step 1: Noise reduction
        if self.use_median:
            img1_step1 = self._median_filter(img1)
            img2_step1 = self._median_filter(img2)
        else:
            img1_step1, img2_step1 = img1, img2

        # Step 2: Alignment
        if self.use_sift:
            img2_aligned = self._align_sift(img1_step1, img2_step1)
        else:
            img2_aligned = img2_step1
        
        img1_step2 = img1_step1

        # Step 3: Color channel extraction
        if self.use_blue_channel:
            # Handle grayscale images
            img1_proc = img1_step2[:, :, 0] if len(img1_step2.shape) == 3 else img1_step2
            img2_proc = img2_aligned[:, :, 0] if len(img2_aligned.shape) == 3 else img2_aligned
        else:
            img1_proc = img1_step2
            img2_proc = img2_aligned
            
        return img1_proc, img2_proc

    def _median_filter(self, img):
        """Your winning noise filter"""
        # Convert 0-1 float to 0-255 uint8
        img_8bit = (img * 255).astype(np.uint8)
        filtered = cv2.medianBlur(img_8bit, 5)
        # Convert back to 0-1 float
        return filtered.astype(np.float32) / 255.0

    def _align_sift(self, img1, img2):
        """Your winning alignment method"""
        
        # Convert to grayscale and 8-bit
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2

        gray1_8bit = (gray1 * 255).astype(np.uint8)
        gray2_8bit = (gray2 * 255).astype(np.uint8)
        h, w = gray1_8bit.shape

        try:
            sift = cv2.SIFT_create(nfeatures=5000)
            kp1, des1 = sift.detectAndCompute(gray1_8bit, None)
            kp2, des2 = sift.detectAndCompute(gray2_8bit, None)

            if des1 is None or des2 is None or len(kp1) < 10 or len(des2) < 2:
                return img2  # Not enough features, return original
            
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) < 10:
                return img2  # Not enough matches

            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
            if H is None:
                return img2

            # Warp the *original* image (color or gray)
            if len(img2.shape) == 3:
                aligned = cv2.warpPerspective(img2, H, (w, h))
            else:
                aligned_8bit = cv2.warpPerspective(gray2_8bit, H, (w, h))
                aligned = aligned_8bit.astype(np.float32) / 255.0
            
            return aligned
        except cv2.error as e:
            print(f"! SIFT failed: {e}. Returning unaligned.")
            return img2
        except Exception as e:
            print(f"! SIFT failed with general error: {e}. Returning unaligned.")
            return img2

# Create preprocessing pipeline variants
PREPROCESSING_PIPELINES = [
    PreprocessingPipeline(use_median=False, use_sift=False, use_blue_channel=False),  # Raw
    PreprocessingPipeline(use_median=True, use_sift=True, use_blue_channel=False),   # Best general
    PreprocessingPipeline(use_median=True, use_sift=True, use_blue_channel=True),    # Best for cracks
]

def get_pipelines():
    """Returns the list of pipeline instances."""
    return PREPROCESSING_PIPELINES

print(f"Created {len(PREPROCESSING_PIPELINES)} preprocessing pipelines:")
for pipeline in PREPROCESSING_PIPELINES:
    print(f"  - {pipeline.name}")

print("✅ Preprocessing module ready.")
