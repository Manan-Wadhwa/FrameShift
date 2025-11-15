#!/usr/bin/env python
#
# CELL 7.5: GROUP B LEARNING-BASED DETECTORS
#

print("--- [Loading] Detectors Group B: Learning-Based ---")

import cv2
import numpy as np
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
from pathlib import Path

from detectors_base import DetectionMethod

# Helper to ensure 3-channel image for models
def _ensure_3channel(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    return img

#%%
#
# CELL 7.5: GROUP B LEARNING-BASED DETECTORS
#
print("\n--- [CELL 7.5] Detection Methods - Group B (Learning-Based) ---")

#
# --- METHOD 7: Simplified PatchCore ---
#
class SimplePatchCoreDetector(DetectionMethod):
    """Lightweight PatchCore implementation. MODIFIED to load pre-trained bank."""
    def __init__(self, memory_bank_path="patchcore_lite_memory.npy"):
        super().__init__("PatchCore-Lite")
        self.memory_bank_path = memory_bank_path
        self.memory_bank = None
        self.is_trained = False
        
        try:
            # Use lightweight backbone
            self.model = timm.create_model(
                'resnet18',
                pretrained=True,
                features_only=True,
                out_indices=[2]  # layer3 output
            )
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not init PatchCore model: {e}. Timm/Torch install?")
            self.model = None

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.load_memory_bank()

    def load_memory_bank(self):
        if Path(self.memory_bank_path).exists():
            try:
                self.memory_bank = np.load(self.memory_bank_path)
                self.is_trained = True
                print(f"  - PatchCore-Lite: Loaded memory bank ({self.memory_bank.shape})")
            except Exception as e:
                print(f"Warning: Failed to load PatchCore memory bank: {e}")
                self.is_trained = False
        else:
            print(f"Warning: {self.memory_bank_path} not found. PatchCore will be skipped.")
            print("  Run `python train_models.py` to create it.")

    def train_and_save(self, normal_images, save_path=None):
        """Train on normal samples and save the memory bank."""
        if save_path is None:
            save_path = self.memory_bank_path
            
        print(f"\nTraining PatchCore on {len(normal_images)} normal images...")
        if self.model is None:
            print("Error: PatchCore model not initialized. Cannot train.")
            return

        features_list = []
        with torch.no_grad():
            for img in normal_images:
                img_3c = _ensure_3channel(img)
                img_tensor = self.transform(img_3c).unsqueeze(0)
                features = self.model(img_tensor)[0]
                pooled = F.adaptive_avg_pool2d(features, (7, 7))
                flattened = pooled.flatten().numpy()
                features_list.append(flattened)
        
        self.memory_bank = np.vstack(features_list)
        np.save(save_path, self.memory_bank)
        self.is_trained = True
        print(f"Memory bank created and saved to {save_path}: {self.memory_bank.shape}")

    def detect(self, img1, img2):
        if not self.is_trained or self.model is None:
            raise RuntimeError("PatchCore must be trained and loaded first! Run train_models.py")
        
        start_time = time.time()
        
        img_3c = _ensure_3channel(img2)
        img_tensor = self.transform(img_3c).unsqueeze(0)

        with torch.no_grad():
            features = self.model(img_tensor)[0]
        
        h, w = features.shape[2:]
        anomaly_map = np.zeros((h, w))

        # Inefficient loop, but matches PDF.
        # A vectorized KNN would be much faster.
        for i in range(h):
            for j in range(w):
                feat_vec = features[0, :, i, j].numpy()
                distances = np.linalg.norm(self.memory_bank - feat_vec, axis=1)
                anomaly_map[i, j] = np.min(distances)
        
        original_size = (img2.shape[1], img2.shape[0])
        anomaly_map_resized = cv2.resize(anomaly_map, original_size)
        
        # Normalize
        min_val = anomaly_map_resized.min()
        max_val = anomaly_map_resized.max()
        anomaly_map_norm = (anomaly_map_resized - min_val) / (max_val - min_val + 1e-8)

        binary_mask = (anomaly_map_norm > 0.5).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        elapsed = time.time() - start_time
        return binary_mask, anomaly_map_norm, elapsed

#
# --- METHOD 8: Background Subtraction (MOG2) ---
#
class BackgroundSubtractionDetector(DetectionMethod):
    """MOG2 background subtraction"""
    def __init__(self):
        super().__init__("BGS-MOG2")
        # Re-init for each call
    
    def detect(self, img1, img2):
        start_time = time.time()
        img1, img2 = self._ensure_same_shape(img1, img2)
        
        # MOG2 needs 0-255 uint8
        img1_8bit = (img1 * 255).astype(np.uint8) if img1.max() <= 1.0 else img1.astype(np.uint8)
        img2_8bit = (img2 * 255).astype(np.uint8) if img2.max() <= 1.0 else img2.astype(np.uint8)

        # Create a new subtractor for each call
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=16, detectShadows=False)

        bg_subtractor.apply(img1_8bit, learningRate=1.0)  # Learn background
        fg_mask = bg_subtractor.apply(img2_8bit, learningRate=0)  # Detect foreground

        binary_mask = (fg_mask > 0).astype(np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        confidence_map = fg_mask.astype(np.float32) / 255.0
        elapsed = time.time() - start_time
        return binary_mask, confidence_map, elapsed

#
# --- METHOD 9: Simple Autoencoder (Reconstruction-based) ---
#
class SimpleAutoencoderDetector(DetectionMethod):
    """Reconstruction error-based detection"""
    def __init__(self):
        super().__init__("Autoencoder-Recon")
        self.is_trained = True  # It's just a proxy
        print("  - Autoencoder: Using simple reconstruction proxy (AbsDiff + Gaussian)")

    def train(self, normal_images):
        """Simulate training"""
        pass  # Not implemented in lightweight version

    def detect(self, img1, img2):
        start_time = time.time()
        img1_gray = self._ensure_grayscale(img1)
        img2_gray = self._ensure_grayscale(img2)
        img1_gray, img2_gray = self._ensure_same_shape(img1_gray, img2_gray)
        
        # Simple proxy: Use img1 as "reconstruction" of img2
        reconstruction_error = np.abs(img1_gray - img2_gray)
        reconstruction_error = gaussian_filter(reconstruction_error, sigma=2)
        
        max_err = reconstruction_error.max()
        recon_norm = reconstruction_error / (max_err + 1e-8) if max_err > 0 else reconstruction_error

        binary_mask = (recon_norm > 0.3).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        elapsed = time.time() - start_time
        return binary_mask, recon_norm, elapsed

#
# Create instances
#
LEARNING_DETECTORS = [
    SimplePatchCoreDetector(),
    BackgroundSubtractionDetector(),
    SimpleAutoencoderDetector(),
]

def get_detectors():
    return LEARNING_DETECTORS

print(f"Created {len(LEARNING_DETECTORS)} learning-based detectors:")
for detector in LEARNING_DETECTORS:
    print(f"  - {detector.name}")

print("✅ Detectors Group B ready.")
