# %%
# ============================================================================
# CELL 1: Setup & Imports
# ============================================================================
# F1 DRIVER ONBOARD MOTION TRACKER V2.0
# Professional driver motion masking with elegant visualization
# Based on FrameShift V1.1 architecture
# ============================================================================

import cv2
import numpy as np
import sys
from collections import deque
import tkinter as tk
from tkinter import filedialog

print("🏎️ F1 DRIVER ONBOARD MOTION TRACKER V2.0")
print("="*70)
print("✅ Imports loaded")

# %%
# ============================================================================
# CELL 2: Configuration
# ============================================================================

CONFIG = {
    # Background subtraction settings
    'bg_history': 500,              # Frames to learn background
    'bg_var_threshold': 25,         # Sensitivity (lower = more sensitive)
    'detect_shadows': False,        # Ignore shadows
    'learning_rate': 0.001,         # How fast to adapt to changes
    
    # Preprocessing
    'denoise_strength': 5,          # Reduce camera noise
    'apply_clahe': True,            # Enhance contrast
    'clahe_clip_limit': 2.0,
    'clahe_grid_size': (8, 8),
    
    # Mask refinement
    'morphology_iterations': 2,     # Clean up mask
    'open_kernel_size': (3, 3),     # Remove small noise
    'close_kernel_size': (9, 9),    # Fill holes
    'dilate_kernel_size': (5, 5),   # Expand mask slightly
    
    # Motion filtering
    'min_motion_area': 200,         # Minimum pixels for motion
    'max_motion_area': 50000,       # Maximum pixels (full hands)
    'temporal_smoothing': True,     # Smooth over time
    'smooth_window': 5,             # Frames to average
    
    # ROI (Region of Interest) - Focus on driver area
    'use_roi': True,                # Enable ROI
    'roi_coords': None,             # Auto-detect or manual (x, y, w, h)
    'roi_padding': 0.1,             # 10% padding around detected area
    
    # Visualization
    'output_mode': 'overlay',       # 'mask', 'overlay', 'side_by_side', 'heatmap'
    'mask_color': (0, 255, 0),      # Green for motion
    'overlay_alpha': 0.6,           # Transparency
    'show_contours': True,          # Draw motion boundaries
    'contour_color': (0, 255, 255), # Yellow contours
    'contour_thickness': 2,
    'show_trails': True,            # Show motion trails
    'trail_length': 15,             # Frames in trail
    'trail_color': (255, 0, 255),   # Magenta trails
    
    # Output
    'output_codec': 'mp4v',         # Video codec
    'output_quality': 95,           # JPEG quality for preview
    'show_preview': True,           # Live preview window
    'preview_scale': 0.6,           # Scale for preview
    'save_debug_frames': False,     # Save individual frames
    'debug_interval': 30,           # Save every N frames
}

print("📋 Configuration loaded")
for key, value in list(CONFIG.items())[:5]:
    print(f"   {key}: {value}")
print(f"   ... and {len(CONFIG)-5} more settings")

# %%
# ============================================================================
# CELL 3: Utility Functions
# ============================================================================

def select_video_gui():
    """Open file dialog to select video"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    print("📂 Select F1 onboard video...")
    video_path = filedialog.askopenfilename(
        title="Select F1 Onboard Video",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return video_path

def preprocess_frame(frame):
    """Apply preprocessing to improve motion detection"""
    processed = frame.copy()
    
    # 1. Denoise
    if CONFIG['denoise_strength'] > 0:
        processed = cv2.fastNlMeansDenoisingColored(
            processed, None, 
            CONFIG['denoise_strength'], 
            CONFIG['denoise_strength'], 7, 21
        )
    
    # 2. Enhance contrast with CLAHE
    if CONFIG['apply_clahe']:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(
            clipLimit=CONFIG['clahe_clip_limit'],
            tileGridSize=CONFIG['clahe_grid_size']
        )
        l = clahe.apply(l)
        
        processed = cv2.merge([l, a, b])
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
    
    return processed

def refine_mask(mask):
    """Clean up and refine the motion mask"""
    refined = mask.copy()
    
    # 1. Morphological operations
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CONFIG['open_kernel_size'])
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CONFIG['close_kernel_size'])
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CONFIG['dilate_kernel_size'])
    
    # Remove small noise
    for _ in range(CONFIG['morphology_iterations']):
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_open)
    
    # Fill holes
    for _ in range(CONFIG['morphology_iterations']):
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel_close)
    
    # Expand slightly to capture full motion
    refined = cv2.dilate(refined, kernel_dilate, iterations=1)
    
    # 2. Filter by area
    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create new mask with only valid contours
    filtered_mask = np.zeros_like(refined)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if CONFIG['min_motion_area'] < area < CONFIG['max_motion_area']:
            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
    
    return filtered_mask, contours

def apply_temporal_smoothing(mask, mask_history):
    """Smooth mask over time to reduce flicker"""
    mask_history.append(mask.astype(float) / 255.0)
    
    # Average recent masks
    avg_mask = np.mean(mask_history, axis=0)
    
    # Threshold back to binary
    smoothed = (avg_mask > 0.3).astype(np.uint8) * 255
    
    return smoothed

def detect_roi_auto(cap, bg_subtractor):
    """Automatically detect driver area by finding motion region"""
    print("🔍 Auto-detecting driver region...")
    
    # Read first frame to get dimensions
    ret, first_frame = cap.read()
    if not ret:
        print("⚠️ Could not read frame for ROI detection")
        return None
    
    motion_accumulator = np.zeros(first_frame.shape[:2], dtype=np.float32)
    
    # Process first frame
    fg_mask = bg_subtractor.apply(first_frame)
    motion_accumulator += fg_mask.astype(float) / 255.0
    
    # Sample more frames to find consistent motion area
    for _ in range(49):  # 49 more frames to total 50
        ret, sample_frame = cap.read()
        if not ret:
            break
        
        fg_mask = bg_subtractor.apply(sample_frame)
        motion_accumulator += fg_mask.astype(float) / 255.0
    
    # Find bounding box of motion
    motion_map = (motion_accumulator > 10).astype(np.uint8) * 255
    contours, _ = cv2.findContours(motion_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get bounding box of largest motion area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        pad_x = int(w * CONFIG['roi_padding'])
        pad_y = int(h * CONFIG['roi_padding'])
        
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(first_frame.shape[1] - x, w + 2*pad_x)
        h = min(first_frame.shape[0] - y, h + 2*pad_y)
        
        print(f"✅ ROI detected: ({x}, {y}, {w}, {h})")
        return (x, y, w, h)
    
    print("⚠️ Could not auto-detect ROI, using full frame")
    return None

print("✅ Utility functions defined")

# %%
# ============================================================================
# CELL 4: Visualization Class
# ============================================================================

class DriverMotionVisualizer:
    """Elegant visualization of driver motion"""
    
    def __init__(self):
        self.motion_trails = deque(maxlen=CONFIG['trail_length'])
        self.contour_history = deque(maxlen=5)
    
    def create_overlay(self, frame, mask):
        """Create overlay visualization"""
        overlay = frame.copy()
        
        # Apply colored mask
        colored_mask = np.zeros_like(frame)
        colored_mask[mask > 0] = CONFIG['mask_color']
        
        # Blend with original
        result = cv2.addWeighted(frame, 1.0, colored_mask, CONFIG['overlay_alpha'], 0)
        
        return result
    
    def create_heatmap(self, frame, mask):
        """Create heatmap visualization"""
        # Apply colormap to mask
        heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        
        # Blend with original
        result = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        
        return result
    
    def create_side_by_side(self, frame, mask):
        """Create side-by-side comparison"""
        # Convert mask to 3-channel
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Stack horizontally
        result = np.hstack([frame, mask_3ch])
        
        return result
    
    def draw_contours(self, frame, contours):
        """Draw motion contours"""
        if CONFIG['show_contours'] and contours:
            valid_contours = [c for c in contours if cv2.contourArea(c) > CONFIG['min_motion_area']]
            cv2.drawContours(frame, valid_contours, -1, 
                           CONFIG['contour_color'], CONFIG['contour_thickness'])
    
    def draw_trails(self, frame, contours):
        """Draw motion trails"""
        if not CONFIG['show_trails']:
            return
        
        # Get centroids of motion regions
        centroids = []
        for contour in contours:
            if cv2.contourArea(contour) > CONFIG['min_motion_area']:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    centroids.append((cx, cy))
        
        if centroids:
            self.motion_trails.append(centroids)
        
        # Draw trails
        for i, trail_points in enumerate(self.motion_trails):
            alpha = (i + 1) / len(self.motion_trails)
            for point in trail_points:
                radius = max(2, int(5 * alpha))
                color = tuple(int(c * alpha) for c in CONFIG['trail_color'])
                cv2.circle(frame, point, radius, color, -1)
    
    def add_info_panel(self, frame, frame_num, total_frames, motion_percentage):
        """Add information overlay"""
        h, w = frame.shape[:2]
        
        # Semi-transparent panel
        panel = frame.copy()
        cv2.rectangle(panel, (10, h - 120), (400, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.5, frame, 0.5, 0, frame)
        
        # Text info
        info_lines = [
            f"Frame: {frame_num}/{total_frames}",
            f"Progress: {(frame_num/total_frames)*100:.1f}%",
            f"Motion: {motion_percentage:.1f}%",
        ]
        
        y_offset = h - 95
        for line in info_lines:
            cv2.putText(frame, line, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
    
    def visualize(self, frame, mask, contours, frame_num, total_frames):
        """Create final visualization"""
        # Calculate motion percentage
        motion_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        motion_percentage = (motion_pixels / total_pixels) * 100
        
        # Create visualization based on mode
        if CONFIG['output_mode'] == 'mask':
            result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        elif CONFIG['output_mode'] == 'overlay':
            result = self.create_overlay(frame, mask)
            self.draw_contours(result, contours)
            self.draw_trails(result, contours)
        elif CONFIG['output_mode'] == 'heatmap':
            result = self.create_heatmap(frame, mask)
        elif CONFIG['output_mode'] == 'side_by_side':
            result = self.create_side_by_side(frame, mask)
        else:
            result = frame
        
        # Add info panel
        self.add_info_panel(result, frame_num, total_frames, motion_percentage)
        
        return result

print("✅ Visualizer class defined")

# %%
# ============================================================================
# CELL 5: Main Processing Function
# ============================================================================

def process_driver_onboard(input_video_path, output_video_path):
    """
    Professional driver motion masking system
    """
    global cap  # For ROI auto-detection
    
    # 1. Open video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video: {input_video_path}")
        return False
    
    # 2. Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_width == 0 or frame_height == 0:
        print("❌ Error: Invalid video dimensions")
        cap.release()
        return False
    
    print(f"\n📹 Video Properties:")
    print(f"   Resolution: {frame_width}x{frame_height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f}s")
    
    # 3. Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=CONFIG['bg_history'],
        varThreshold=CONFIG['bg_var_threshold'],
        detectShadows=CONFIG['detect_shadows']
    )
    
    # 4. Auto-detect ROI if enabled
    roi = None
    if CONFIG['use_roi'] and CONFIG['roi_coords'] is None:
        roi = detect_roi_auto(cap, bg_subtractor)
        CONFIG['roi_coords'] = roi
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    elif CONFIG['use_roi']:
        roi = CONFIG['roi_coords']
    
    # 5. Setup output video
    output_width = frame_width
    output_height = frame_height
    
    if CONFIG['output_mode'] == 'side_by_side':
        output_width = frame_width * 2
    
    fourcc = cv2.VideoWriter_fourcc(*CONFIG['output_codec'])
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
    
    print(f"\n✅ Output: {output_video_path}")
    print(f"   Mode: {CONFIG['output_mode']}")
    print(f"   ROI: {'Enabled' if roi else 'Disabled'}")
    
    # 6. Initialize systems
    visualizer = DriverMotionVisualizer()
    mask_history = deque(maxlen=CONFIG['smooth_window'])
    
    print(f"\n🚀 Processing video...")
    print("="*70)
    
    frame_count = 0
    
    # 7. Process frame by frame
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Preprocess
            processed = preprocess_frame(frame)
            
            # Apply ROI if set
            process_region = processed
            if roi:
                x, y, w, h = roi
                process_region = processed[y:y+h, x:x+w]
            
            # Background subtraction
            fg_mask = bg_subtractor.apply(
                process_region, 
                learningRate=CONFIG['learning_rate']
            )
            
            # Refine mask
            refined_mask, contours = refine_mask(fg_mask)
            
            # Temporal smoothing
            if CONFIG['temporal_smoothing']:
                refined_mask = apply_temporal_smoothing(refined_mask, mask_history)
            
            # Restore full frame size if ROI was used
            if roi:
                full_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                full_mask[y:y+h, x:x+w] = refined_mask
                refined_mask = full_mask
            
            # Visualize
            output_frame = visualizer.visualize(
                frame, refined_mask, contours, 
                frame_count + 1, total_frames
            )
            
            # Write frame
            out.write(output_frame)
            
            # Show preview
            if CONFIG['show_preview']:
                preview = cv2.resize(output_frame, None, 
                                   fx=CONFIG['preview_scale'], 
                                   fy=CONFIG['preview_scale'])
                cv2.imshow('F1 Driver Motion Tracker', preview)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️ Stopped by user")
                    break
                elif key == ord('s'):
                    cv2.imwrite(f'frame_{frame_count}.jpg', output_frame)
                    print(f"💾 Saved frame_{frame_count}.jpg")
            
            # Save debug frames
            if CONFIG['save_debug_frames'] and frame_count % CONFIG['debug_interval'] == 0:
                cv2.imwrite(f'debug_frame_{frame_count:05d}.jpg', output_frame)
            
            # Progress
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    
    finally:
        # 8. Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print(f"✅ Processing complete!")
    print(f"   Processed {frame_count} frames")
    print(f"   Output saved: {output_video_path}")
    print("="*70)
    
    return True

print("✅ Main processing function defined")

# %%
# ============================================================================
# CELL 6: Main Execution (Run this to start)
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🏁 F1 DRIVER ONBOARD MOTION TRACKER V2.0")
    print("   Professional driver motion masking system")
    print("="*70 + "\n")
    
    # Configuration options
    print("⚙️ QUICK CONFIG PRESETS:")
    print("   1. High Quality (best results, slower)")
    print("   2. Balanced (good quality, moderate speed)")
    print("   3. Fast Preview (lower quality, faster)")
    print("   4. Custom (use CONFIG settings)")
    
    preset = input("\nSelect preset (1-4) or press Enter for default [2]: ").strip()
    
    if preset == "1":
        # High Quality
        CONFIG['denoise_strength'] = 7
        CONFIG['morphology_iterations'] = 3
        CONFIG['temporal_smoothing'] = True
        CONFIG['smooth_window'] = 7
        print("✅ Using HIGH QUALITY preset")
    elif preset == "3":
        # Fast Preview
        CONFIG['denoise_strength'] = 3
        CONFIG['morphology_iterations'] = 1
        CONFIG['temporal_smoothing'] = False
        CONFIG['learning_rate'] = 0.005
        print("✅ Using FAST PREVIEW preset")
    elif preset == "4":
        print("✅ Using CUSTOM CONFIG settings")
    else:
        # Balanced (default)
        print("✅ Using BALANCED preset")
    
    # Select visualization mode
    print("\n🎨 VISUALIZATION MODES:")
    print("   1. Overlay (green motion overlay)")
    print("   2. Heatmap (thermal-style visualization)")
    print("   3. Side-by-Side (original + mask)")
    print("   4. Mask Only (black & white)")
    
    viz_mode = input("\nSelect mode (1-4) or press Enter for default [1]: ").strip()
    
    if viz_mode == "2":
        CONFIG['output_mode'] = 'heatmap'
    elif viz_mode == "3":
        CONFIG['output_mode'] = 'side_by_side'
    elif viz_mode == "4":
        CONFIG['output_mode'] = 'mask'
    else:
        CONFIG['output_mode'] = 'overlay'
    
    print(f"✅ Using {CONFIG['output_mode'].upper()} mode\n")
    
    # Get video file
    use_gui = input("📂 Use file dialog to select video? (y/n) [y]: ").strip().lower()
    
    if use_gui != 'n':
        INPUT_VIDEO = select_video_gui()
        if not INPUT_VIDEO:
            print("❌ No video selected. Exiting.")
            sys.exit(1)
    else:
        INPUT_VIDEO = input("Enter video path: ").strip().strip('"')
    
    # Generate output filename
    import os
    base_name = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
    OUTPUT_VIDEO = f"{base_name}_motion_tracked_2.mp4"
    
    print(f"\n📹 Input: {INPUT_VIDEO}")
    print(f"💾 Output: {OUTPUT_VIDEO}")
    
    # Confirm
    proceed = input("\n▶️ Start processing? (y/n) [y]: ").strip().lower()
    
    if proceed == 'n':
        print("❌ Processing cancelled.")
        sys.exit(0)
    
    # Process video
    try:
        success = process_driver_onboard(INPUT_VIDEO, OUTPUT_VIDEO)
        
        if success:
            print("\n🎉 SUCCESS! Your driver motion tracking video is ready!")
            print(f"   Location: {OUTPUT_VIDEO}")
            print("\n💡 Tips:")
            print("   • Try different visualization modes for better results")
            print("   • Adjust CONFIG settings for fine-tuning")
            print("   • Use ROI to focus on driver area only")
        else:
            print("\n❌ Processing failed. Check error messages above.")
    
    except FileNotFoundError:
        print(f"\n❌ ERROR: Video file not found: {INPUT_VIDEO}")
        print("   Please check the file path and try again.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("🏁 F1 DRIVER ONBOARD MOTION TRACKER - Session Complete")
    print("="*70)
# %%
