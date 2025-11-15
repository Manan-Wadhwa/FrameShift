#!/usr/bin/env python
#
# CELL 7.4: DETECTION METHODS - BASE CLASS
#

print("--- [Loading] Detector Base Class ---")

import cv2
import numpy as np
import time

#%%
#
# CELL 7.4: DETECTION METHODS (BASE)
#
print("\n--- [CELL 7.4] Detection Methods - Base Class ---")

class DetectionMethod:
    """Base class for all detection methods"""
    def __init__(self, name):
        self.name = name

    def detect(self, img1, img2):
        """
        Returns: (binary_mask, confidence_map, time_elapsed)
        """
        raise NotImplementedError

    def _ensure_grayscale(self, img):
        """Helper to ensure image is grayscale"""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _ensure_same_shape(self, img1, img2):
        """Ensure both images have same shape"""
        if img1.shape != img2.shape:
            print("Warning: Mismatched shapes, resizing img2.")
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        return img1, img2

print("✅ Detector base class ready.")
