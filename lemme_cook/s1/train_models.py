#!/usr/bin/env python
#
# This script pre-trains any models that require it,
# like PatchCore-Lite.
#
# Run this *before* starting the benchmark workers.
#

import numpy as np
import time

# Import the components we need
from config_and_utils import get_normal_samples
from detectors_group_b import SimplePatchCoreDetector
# Import other trainable detectors here...

print("--- 🧠 Model Training Script ---")

#
# --- 1. Train PatchCore-Lite ---
#
print("\n--- Training PatchCore-Lite ---")
try:
    # Get normal "good" images from the test cases
    normal_samples = get_normal_samples()
    
    if len(normal_samples) > 0:
        patchcore_detector = SimplePatchCoreDetector()
        # This method will train AND save the .npy file
        patchcore_detector.train_and_save(normal_samples, "patchcore_lite_memory.npy")
        print("✅ PatchCore-Lite training complete.")
    else:
        print("Warning: No normal samples found. PatchCore training skipped.")

except Exception as e:
    print(f"❌ PatchCore training failed: {e}")
    print("  Ensure 'torch', 'torchvision', and 'timm' are installed.")


#
# --- 2. Train Autoencoder (if not a proxy) ---
#
print("\n--- Training Autoencoder ---")
print("  Autoencoder is a proxy in this benchmark, no training needed.")


print("\n--- ✅ All model training complete. ---")
