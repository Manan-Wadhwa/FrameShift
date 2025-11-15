#!/usr/bin/env python
#
# CELL 7.4: GROUP A FAST PIXEL-LEVEL DETECTORS
#

print("--- [Loading] Detectors Group A: Fast Pixel-Level ---")

import cv2
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim

from detectors_base import DetectionMethod

#%%
#
# CELL 7.4: GROUP A FAST PIXEL-LEVEL DETECTORS
#
print("\n--- [CELL 7.4] Detection Methods - Group A (Fast Pixel-Level) ---")

#
# METHOD 1: Multi-Threshold SSIM
#
class SSIMMultiThresholdDetector(DetectionMethod):
    """SSIM with adaptive thresholding"""
    def __init__(self):
        super().__init__("SSIM+MultiThreshold")

    def detect(self, img1, img2):
        start_time = time.time()
        
        img1_gray = self._ensure_grayscale(img1)
        img2_gray = self._ensure_grayscale(img2)
        img1_gray, img2_gray = self._ensure_same_shape(img1_gray, img2_gray)

        # Compute SSIM
        try:
            score, diff_map = ssim(img1_gray, img2_gray, data_range=1.0, full=True)
        except ValueError as e:
            # Handle case where images are too small
            print(f"SSIM failed: {e}. Using simple diff.")
            diff_map = np.abs(img1_gray - img2_gray)

        diff_map = 1.0 - diff_map  # Invert: 0=same, 1=different
        
        best_threshold = 0.3  # For demo, use single good threshold
        binary_mask = (diff_map > best_threshold).astype(np.uint8)

        # Post-processing: morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Remove small regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if stats.shape[0] > 1:
            min_area_threshold = 100
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_area_threshold:
                    binary_mask[labels == i] = 0
        
        elapsed = time.time() - start_time
        return binary_mask, diff_map, elapsed

#
# --- METHOD 2: Multi-Scale SSIM ---
#
class MultiScaleSSIMDetector(DetectionMethod):
    """SSIM at multiple scales"""
    def __init__(self):
        super().__init__("MultiScale-SSIM")
        self.scales = [1.0, 0.5, 0.25]
        self.weights = [0.5, 0.3, 0.2]

    def detect(self, img1, img2):
        start_time = time.time()
        img1_gray = self._ensure_grayscale(img1)
        img2_gray = self._ensure_grayscale(img2)
        img1_gray, img2_gray = self._ensure_same_shape(img1_gray, img2_gray)
        
        h, w = img1_gray.shape
        detection_maps = []

        for scale in self.scales:
            if scale != 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                img1_scaled = cv2.resize(img1_gray, (new_w, new_h))
                img2_scaled = cv2.resize(img2_gray, (new_w, new_h))
            else:
                img1_scaled, img2_scaled = img1_gray, img2_gray
            
            try:
                _, diff_map = ssim(img1_scaled, img2_scaled, data_range=1.0, full=True)
                diff_map = 1.0 - diff_map
            except ValueError:
                diff_map = np.abs(img1_scaled - img2_scaled)

            if scale != 1.0:
                diff_map = cv2.resize(diff_map, (w, h))
            detection_maps.append(diff_map)

        # Weighted fusion
        fused_map = np.zeros_like(detection_maps[0])
        for map_i, weight in zip(detection_maps, self.weights):
            fused_map += weight * map_i

        binary_mask = (fused_map > 0.3).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        elapsed = time.time() - start_time
        return binary_mask, fused_map, elapsed

#
# --- METHOD 3: Edge Delta (Your Blue Channel Winner!)
#
class EdgeDeltaDetector(DetectionMethod):
    """Edge-based detection using Canny"""
    def __init__(self, use_blue_channel=True):
        name = "EdgeDelta" + ("-Blue" if use_blue_channel else "")
        super().__init__(name)
        self.use_blue_channel = use_blue_channel

    def detect(self, img1, img2):
        start_time = time.time()
        
        if self.use_blue_channel and len(img1.shape) == 3:
            img1_proc = img1[:, :, 0]  # Blue channel
            img2_proc = img2[:, :, 0]
        else:
            img1_proc = self._ensure_grayscale(img1)
            img2_proc = self._ensure_grayscale(img2)
            
        img1_proc, img2_proc = self._ensure_same_shape(img1_proc, img2_proc)

        img1_8bit = (img1_proc * 255).astype(np.uint8)
        img2_8bit = (img2_proc * 255).astype(np.uint8)

        edges1_low = cv2.Canny(img1_8bit, 30, 100)
        edges1_high = cv2.Canny(img1_8bit, 100, 200)
        edges2_low = cv2.Canny(img2_8bit, 30, 100)
        edges2_high = cv2.Canny(img2_8bit, 100, 200)

        edge_diff_low = cv2.bitwise_xor(edges1_low, edges2_low)
        edge_diff_high = cv2.bitwise_xor(edges1_high, edges2_high)
        binary_mask = cv2.bitwise_or(edge_diff_low, edge_diff_high)
        binary_mask = (binary_mask > 0).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        confidence_map = binary_mask.astype(np.float32)
        elapsed = time.time() - start_time
        return binary_mask, confidence_map, elapsed

#
# --- METHOD 4: Color Histogram Delta ---
#
class ColorHistogramDetector(DetectionMethod):
    """Block-wise histogram comparison"""
    def __init__(self, block_size=32):
        super().__init__(f"ColorHist-{block_size}")
        self.block_size = block_size

    def detect(self, img1, img2):
        start_time = time.time()
        img1, img2 = self._ensure_same_shape(img1, img2)
        h, w = img1.shape[:2]
        
        change_mask = np.zeros((h, w), dtype=np.uint8)
        confidence_map = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                y_end = min(y + self.block_size, h)
                x_end = min(x + self.block_size, w)
                
                block1 = img1[y:y_end, x:x_end]
                block2 = img2[y:y_end, x:x_end]
                
                if block1.size == 0 or block2.size == 0:
                    continue

                # Compute histogram for blue channel (if color) or grayscale
                if len(block1.shape) == 3:
                    hist1 = cv2.calcHist([block1], [0], None, [32], [0, 1])  # Blue
                    hist2 = cv2.calcHist([block2], [0], None, [32], [0, 1])
                else:
                    # Convert 0-1 float to 0-255 uint8 for hist
                    b1_8bit = (block1 * 255).astype(np.uint8)
                    b2_8bit = (block2 * 255).astype(np.uint8)
                    hist1 = cv2.calcHist([b1_8bit], [0], None, [32], [0, 256])
                    hist2 = cv2.calcHist([b2_8bit], [0], None, [32], [0, 256])

                hist1 = cv2.normalize(hist1, hist1).flatten()
                hist2 = cv2.normalize(hist2, hist2).flatten()
                
                similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                
                if similarity < 0.85:
                    change_mask[y:y_end, x:x_end] = 1
                confidence_map[y:y_end, x:x_end] = 1.0 - similarity

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)
        
        elapsed = time.time() - start_time
        return change_mask, confidence_map, elapsed

#
# --- METHOD 5: Optical Flow ---
#
class OpticalFlowDetector(DetectionMethod):
    """Farneback optical flow"""
    def __init__(self):
        super().__init__("OpticalFlow")

    def detect(self, img1, img2):
        start_time = time.time()
        img1_gray = self._ensure_grayscale(img1)
        img2_gray = self._ensure_grayscale(img2)
        img1_gray, img2_gray = self._ensure_same_shape(img1_gray, img2_gray)

        # Farneback needs 0-255 uint8
        img1_8bit = (img1_gray * 255).astype(np.uint8)
        img2_8bit = (img2_gray * 255).astype(np.uint8)
        
        flow = cv2.calcOpticalFlowFarneback(
            img1_8bit, img2_8bit, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        threshold = np.percentile(magnitude, 95) if magnitude.size > 0 else 0
        binary_mask = (magnitude > threshold).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        max_mag = magnitude.max()
        confidence_map = magnitude / (max_mag + 1e-8) if max_mag > 0 else magnitude
        
        elapsed = time.time() - start_time
        return binary_mask, confidence_map, elapsed

#
# --- METHOD 6: Gradient Magnitude Difference ---
#
class GradientMagnitudeDetector(DetectionMethod):
    """Sobel gradient-based detection"""
    def __init__(self):
        super().__init__("GradientMag")

    def detect(self, img1, img2):
        start_time = time.time()
        img1_gray = self._ensure_grayscale(img1)
        img2_gray = self._ensure_grayscale(img2)
        img1_gray, img2_gray = self._ensure_same_shape(img1_gray, img2_gray)

        # Compute gradients (use 0-1 float images)
        grad1_x = cv2.Sobel(img1_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad1_y = cv2.Sobel(img1_gray, cv2.CV_64F, 0, 1, ksize=3)
        grad2_x = cv2.Sobel(img2_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad2_y = cv2.Sobel(img2_gray, cv2.CV_64F, 0, 1, ksize=3)

        mag1 = np.sqrt(grad1_x**2 + grad1_y**2)
        mag2 = np.sqrt(grad2_x**2 + grad2_y**2)
        mag_diff = np.abs(mag1 - mag2)
        
        max_diff = mag_diff.max()
        mag_diff_norm = mag_diff / (max_diff + 1e-8) if max_diff > 0 else mag_diff

        binary_mask = (mag_diff_norm > 0.3).astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        elapsed = time.time() - start_time
        return binary_mask, mag_diff_norm, elapsed

#
# Create instances
#
FAST_DETECTORS = [
    SSIMMultiThresholdDetector(),
    MultiScaleSSIMDetector(),
    EdgeDeltaDetector(use_blue_channel=True),
    ColorHistogramDetector(block_size=32),
    OpticalFlowDetector(),
    GradientMagnitudeDetector(),
]

def get_detectors():
    return FAST_DETECTORS

print(f"Created {len(FAST_DETECTORS)} fast pixel-level detectors:")
for detector in FAST_DETECTORS:
    print(f"  - {detector.name}")

print("✅ Detectors Group A ready.")
