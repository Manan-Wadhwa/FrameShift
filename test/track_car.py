# %%
# ============================================================================
# F1 RACE POSITION TRACKER V1.0
# Detect overtakes and track car positions in race footage
# ============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
from datetime import timedelta

print("🏁 F1 Race Position Tracker V1.0 Ready!")
print("   • Car Detection & Tracking")
print("   • Overtake Detection")
print("   • Position Analysis")
print("   • Race Timeline Generation")

# %%
# ============================================================================
# CELL 1: Configuration
# ============================================================================

CONFIG = {
    # Video input
    'video_path': None,  # Set to video path or None for file dialog
    'use_webcam': False,  # True to use webcam for testing
    
    # Detection settings
    'min_car_area': 500,  # Minimum pixels for car detection
    'max_car_area': 50000,  # Maximum pixels for car detection
    'detection_roi': None,  # (x, y, w, h) or None for full frame
    
    # Tracking settings
    'max_track_age': 30,  # Frames before track is lost
    'min_track_confidence': 5,  # Minimum frames to confirm track
    'position_threshold': 50,  # Pixels to consider position change
    
    # Overtake detection
    'overtake_cooldown': 60,  # Frames between same overtake events
    'lateral_threshold': 30,  # Horizontal movement for overtake
    'overtake_duration': 10,  # Frames to confirm overtake
    
    # Visualization
    'show_trails': True,  # Show car movement trails
    'trail_length': 30,  # Number of points in trail
    'show_speed_estimate': True,  # Show relative speed
    'output_video': True,  # Save annotated video
    'output_path': 'f1_race_analysis.mp4',
    
    # Analysis
    'save_telemetry': True,  # Save position data to JSON
    'telemetry_path': 'race_telemetry.json',
    'generate_timeline': True,  # Create overtake timeline
}

print("📋 Current Configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")

# %%
# ============================================================================
# CELL 2: Data Structures
# ============================================================================

@dataclass
class CarTrack:
    """Represents a tracked car"""
    id: int
    name: str
    color: Tuple[int, int, int]
    positions: deque  # Recent positions
    bboxes: deque  # Recent bounding boxes
    last_seen: int  # Frame number
    confidence: int  # Tracking confidence
    total_frames: int  # Total frames tracked
    
    def __post_init__(self):
        if not isinstance(self.positions, deque):
            self.positions = deque(maxlen=CONFIG['trail_length'])
        if not isinstance(self.bboxes, deque):
            self.bboxes = deque(maxlen=10)
    
    def update(self, position, bbox, frame_num):
        """Update track with new detection"""
        self.positions.append(position)
        self.bboxes.append(bbox)
        self.last_seen = frame_num
        self.confidence = min(self.confidence + 1, 100)
        self.total_frames += 1
    
    def get_velocity(self):
        """Estimate velocity from recent positions"""
        if len(self.positions) < 2:
            return (0, 0)
        
        recent = list(self.positions)[-5:]
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        return (dx / len(recent), dy / len(recent))
    
    def get_current_position(self):
        """Get most recent position"""
        return self.positions[-1] if self.positions else None

@dataclass
class OvertakeEvent:
    """Represents an overtake"""
    frame: int
    timestamp: float
    overtaking_car: str
    overtaken_car: str
    position_before: Tuple[int, int]
    position_after: Tuple[int, int]
    confidence: float

class RaceTracker:
    """Main tracking system"""
    
    def __init__(self):
        self.tracks: Dict[int, CarTrack] = {}
        self.next_id = 0
        self.frame_count = 0
        self.overtakes: List[OvertakeEvent] = []
        self.last_overtake = defaultdict(int)  # Cooldown tracking
        
        # Car colors for visualization (expand as needed)
        self.car_colors = [
            (255, 0, 0),    # Red
            (0, 0, 255),    # Blue
            (0, 255, 0),    # Green
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]
        
        # Telemetry data
        self.telemetry = defaultdict(list)
    
    def create_track(self, position, bbox):
        """Create new car track"""
        track_id = self.next_id
        color = self.car_colors[track_id % len(self.car_colors)]
        
        track = CarTrack(
            id=track_id,
            name=f"Car {track_id + 1}",
            color=color,
            positions=deque(maxlen=CONFIG['trail_length']),
            bboxes=deque(maxlen=10),
            last_seen=self.frame_count,
            confidence=1,
            total_frames=0
        )
        
        track.update(position, bbox, self.frame_count)
        self.tracks[track_id] = track
        self.next_id += 1
        
        return track_id
    
    def update_tracks(self, detections):
        """Update all tracks with new detections"""
        self.frame_count += 1
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        for track_id, track in self.tracks.items():
            if track.last_seen < self.frame_count - CONFIG['max_track_age']:
                continue
            
            best_match = None
            best_distance = float('inf')
            
            last_pos = track.get_current_position()
            if last_pos is None:
                continue
            
            for i, (position, bbox) in enumerate(detections):
                if i in matched_detections:
                    continue
                
                distance = np.linalg.norm(np.array(position) - np.array(last_pos))
                
                if distance < best_distance and distance < 100:  # Max matching distance
                    best_distance = distance
                    best_match = i
            
            if best_match is not None:
                track.update(detections[best_match][0], detections[best_match][1], self.frame_count)
                matched_tracks.add(track_id)
                matched_detections.add(best_match)
        
        # Create new tracks for unmatched detections
        for i, (position, bbox) in enumerate(detections):
            if i not in matched_detections:
                self.create_track(position, bbox)
        
        # Record telemetry
        for track_id, track in self.tracks.items():
            if track.last_seen == self.frame_count:
                pos = track.get_current_position()
                vel = track.get_velocity()
                self.telemetry[track.name].append({
                    'frame': self.frame_count,
                    'position': pos,
                    'velocity': vel,
                    'bbox': track.bboxes[-1] if track.bboxes else None
                })
    
    def detect_overtakes(self):
        """Detect overtake events between cars"""
        active_tracks = [t for t in self.tracks.values() 
                        if t.last_seen >= self.frame_count - 5 and 
                        t.confidence >= CONFIG['min_track_confidence']]
        
        if len(active_tracks) < 2:
            return
        
        # Check all pairs of cars
        for i, track1 in enumerate(active_tracks):
            for track2 in active_tracks[i+1:]:
                self._check_overtake(track1, track2)
    
    def _check_overtake(self, track1, track2):
        """Check if one car is overtaking another"""
        if len(track1.positions) < 5 or len(track2.positions) < 5:
            return
        
        # Get recent positions
        pos1_old = list(track1.positions)[0]
        pos1_new = list(track1.positions)[-1]
        pos2_old = list(track2.positions)[0]
        pos2_new = list(track2.positions)[-1]
        
        # Check cooldown
        overtake_key = f"{track1.id}-{track2.id}"
        if self.frame_count - self.last_overtake[overtake_key] < CONFIG['overtake_cooldown']:
            return
        
        # Detect position swap (assuming horizontal racing line)
        x1_change = pos1_new[0] - pos1_old[0]
        x2_change = pos2_new[0] - pos2_old[0]
        
        # Check if cars crossed positions
        was_behind = pos1_old[0] < pos2_old[0]
        now_ahead = pos1_new[0] > pos2_new[0]
        
        lateral_movement = abs(x1_change - x2_change)
        
        if was_behind and now_ahead and lateral_movement > CONFIG['lateral_threshold']:
            # Track1 overtook Track2
            overtake = OvertakeEvent(
                frame=self.frame_count,
                timestamp=self.frame_count / 30.0,  # Assuming 30 fps
                overtaking_car=track1.name,
                overtaken_car=track2.name,
                position_before=pos1_old,
                position_after=pos1_new,
                confidence=min(track1.confidence, track2.confidence) / 100.0
            )
            
            self.overtakes.append(overtake)
            self.last_overtake[overtake_key] = self.frame_count
            
            print(f"🏁 OVERTAKE! {track1.name} passed {track2.name} at frame {self.frame_count}")

print("✅ Data structures initialized")

# %%
# ============================================================================
# CELL 3: Car Detection
# ============================================================================

class CarDetector:
    """Detect cars in video frames"""
    
    def __init__(self):
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=True
        )
        
        # Color ranges for car detection (expand for specific teams)
        self.color_ranges = [
            # Red (Ferrari, Red Bull)
            ((0, 100, 100), (10, 255, 255)),
            ((170, 100, 100), (180, 255, 255)),
            # Blue (Mercedes, Alpine)
            ((100, 100, 100), (130, 255, 255)),
            # Silver/Gray
            ((0, 0, 100), (180, 50, 200)),
        ]
    
    def detect_by_motion(self, frame):
        """Detect cars using motion/background subtraction"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if CONFIG['min_car_area'] < area < CONFIG['max_car_area']:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (cars are wider than tall)
                aspect_ratio = w / h if h > 0 else 0
                if 0.8 < aspect_ratio < 4.0:
                    center = (x + w//2, y + h//2)
                    detections.append((center, (x, y, w, h)))
        
        return detections
    
    def detect_by_color(self, frame):
        """Detect cars using color information"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if CONFIG['min_car_area'] < area < CONFIG['max_car_area']:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w//2, y + h//2)
                detections.append((center, (x, y, w, h)))
        
        return detections
    
    def detect(self, frame):
        """Combine detection methods"""
        motion_detections = self.detect_by_motion(frame)
        color_detections = self.detect_by_color(frame)
        
        # Merge detections (simple union for now)
        all_detections = motion_detections + color_detections
        
        # Remove duplicates (detections close to each other)
        unique_detections = []
        for det in all_detections:
            is_duplicate = False
            for unique_det in unique_detections:
                distance = np.linalg.norm(np.array(det[0]) - np.array(unique_det[0]))
                if distance < 50:  # Merge threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_detections.append(det)
        
        return unique_detections

print("✅ Car detector initialized")

# %%
# ============================================================================
# CELL 4: Visualization
# ============================================================================

class RaceVisualizer:
    """Visualize tracking and overtakes"""
    
    def __init__(self, tracker):
        self.tracker = tracker
        self.overtake_flash = {}  # Flash effect for overtakes
    
    def draw_track(self, frame, track):
        """Draw single car track"""
        if not track.positions:
            return
        
        # Draw trail
        if CONFIG['show_trails'] and len(track.positions) > 1:
            points = np.array(list(track.positions), dtype=np.int32)
            for i in range(len(points) - 1):
                alpha = (i + 1) / len(points)
                thickness = max(1, int(3 * alpha))
                cv2.line(frame, tuple(points[i]), tuple(points[i+1]), 
                        track.color, thickness)
        
        # Draw current position
        pos = track.get_current_position()
        if pos and track.bboxes:
            x, y, w, h = track.bboxes[-1]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), track.color, 2)
            
            # Draw car name and info
            label = track.name
            if CONFIG['show_speed_estimate']:
                vel = track.get_velocity()
                speed = np.linalg.norm(vel)
                label += f" ({speed:.1f})"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y), 
                         track.color, -1)
            cv2.putText(frame, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw position marker
            cv2.circle(frame, pos, 5, track.color, -1)
            cv2.circle(frame, pos, 7, (255, 255, 255), 2)
    
    def draw_overtake_notification(self, frame, overtake):
        """Flash overtake notification"""
        if self.tracker.frame_count - overtake.frame < 60:  # Show for 2 seconds
            h, w = frame.shape[:2]
            
            # Create notification box
            text = f"OVERTAKE! {overtake.overtaking_car} > {overtake.overtaken_car}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_BOLD, 1.0, 2)
            
            box_x = (w - text_w) // 2 - 20
            box_y = 50
            
            # Pulsing effect
            alpha = 0.5 + 0.5 * np.sin((self.tracker.frame_count - overtake.frame) * 0.3)
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, box_y), (box_x + text_w + 40, box_y + text_h + 20),
                         (0, 255, 0), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            cv2.putText(frame, text, (box_x + 20, box_y + text_h + 10),
                       cv2.FONT_HERSHEY_BOLD, 1.0, (255, 255, 255), 2)
    
    def draw_race_info(self, frame):
        """Draw race statistics"""
        h, w = frame.shape[:2]
        
        # Info panel
        info_lines = [
            f"Frame: {self.tracker.frame_count}",
            f"Cars: {len([t for t in self.tracker.tracks.values() if t.confidence > CONFIG['min_track_confidence']])}",
            f"Overtakes: {len(self.tracker.overtakes)}",
        ]
        
        y_offset = h - 100
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, line, (20, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    def draw_frame(self, frame):
        """Draw all visualizations"""
        viz_frame = frame.copy()
        
        # Draw all tracks
        for track in self.tracker.tracks.values():
            if track.confidence >= CONFIG['min_track_confidence']:
                self.draw_track(viz_frame, track)
        
        # Draw recent overtakes
        for overtake in self.tracker.overtakes[-5:]:  # Last 5 overtakes
            self.draw_overtake_notification(viz_frame, overtake)
        
        # Draw info panel
        self.draw_race_info(viz_frame)
        
        return viz_frame

print("✅ Visualizer initialized")

# %%
# ============================================================================
# CELL 5: Load Video
# ============================================================================

import tkinter as tk
from tkinter import filedialog

def select_video():
    """Open file dialog to select video"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    print("📂 Select race video...")
    video_path = filedialog.askopenfilename(
        title="Select Race Video",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return video_path

# Get video source
if CONFIG['use_webcam']:
    video_source = 0
    print("📹 Using webcam")
elif CONFIG['video_path']:
    video_source = CONFIG['video_path']
    print(f"📹 Using video: {CONFIG['video_path']}")
else:
    video_source = select_video()
    if not video_source:
        print("❌ No video selected")
        raise FileNotFoundError("No video selected")

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"❌ Failed to open video: {video_source}")
    raise IOError("Failed to open video")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"✅ Video loaded:")
print(f"   Resolution: {frame_width}x{frame_height}")
print(f"   FPS: {fps}")
print(f"   Total frames: {total_frames}")

# Setup output video writer
if CONFIG['output_video']:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(CONFIG['output_path'], fourcc, fps, 
                          (frame_width, frame_height))
    print(f"✅ Output video: {CONFIG['output_path']}")

# %%
# ============================================================================
# CELL 6: Process Video
# ============================================================================

print("\n" + "="*70)
print("🏁 STARTING RACE ANALYSIS")
print("="*70)
print("\nPress 'q' to stop, 'p' to pause, 's' to save current frame\n")

# Initialize systems
tracker = RaceTracker()
detector = CarDetector()
visualizer = RaceVisualizer(tracker)

frame_skip = 1  # Process every Nth frame (1 = process all)
paused = False

try:
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("\n✅ Video processing complete")
                break
            
            # Process frame
            if tracker.frame_count % frame_skip == 0:
                # Apply ROI if configured
                process_frame = frame
                if CONFIG['detection_roi']:
                    x, y, w, h = CONFIG['detection_roi']
                    process_frame = frame[y:y+h, x:x+w]
                
                # Detect cars
                detections = detector.detect(process_frame)
                
                # Update tracks
                tracker.update_tracks(detections)
                
                # Detect overtakes
                tracker.detect_overtakes()
            
            # Visualize
            viz_frame = visualizer.draw_frame(frame)
            
            # Save frame
            if CONFIG['output_video']:
                out.write(viz_frame)
            
            # Display
            display_frame = cv2.resize(viz_frame, (1280, 720))
            cv2.imshow('F1 Race Tracker', display_frame)
            
            # Progress
            if tracker.frame_count % 30 == 0:
                progress = (tracker.frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {progress:.1f}% | Frame: {tracker.frame_count} | Cars: {len(tracker.tracks)} | Overtakes: {len(tracker.overtakes)}")
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n⏹️ Stopped by user")
            break
        elif key == ord('p'):
            paused = not paused
            print("⏸️ Paused" if paused else "▶️ Resumed")
        elif key == ord('s'):
            cv2.imwrite(f'frame_{tracker.frame_count}.jpg', viz_frame)
            print(f"💾 Saved frame_{tracker.frame_count}.jpg")

except KeyboardInterrupt:
    print("\n⏹️ Interrupted by user")

finally:
    cap.release()
    if CONFIG['output_video']:
        out.release()
    cv2.destroyAllWindows()

print("\n" + "="*70)
print("📊 RACE ANALYSIS COMPLETE")
print("="*70)

# %%
# ============================================================================
# CELL 7: Generate Reports
# ============================================================================

print("\n📈 Generating analysis reports...\n")

# Overtake Timeline
if CONFIG['generate_timeline'] and tracker.overtakes:
    print("🏁 OVERTAKE TIMELINE:")
    print("-" * 70)
    
    for i, overtake in enumerate(tracker.overtakes):
        timestamp = timedelta(seconds=overtake.timestamp)
        print(f"{i+1}. [{timestamp}] {overtake.overtaking_car} overtook {overtake.overtaken_car}")
        print(f"   Confidence: {overtake.confidence:.2f}")
    
    print()

# Track Statistics
print("🏎️ CAR STATISTICS:")
print("-" * 70)

for track in sorted(tracker.tracks.values(), key=lambda t: t.total_frames, reverse=True):
    if track.confidence >= CONFIG['min_track_confidence']:
        print(f"{track.name}:")
        print(f"   Frames tracked: {track.total_frames}")
        print(f"   Confidence: {track.confidence}")
        
        if track.positions:
            positions = list(track.positions)
            total_distance = sum(
                np.linalg.norm(np.array(positions[i+1]) - np.array(positions[i]))
                for i in range(len(positions) - 1)
            )
            print(f"   Distance traveled: {total_distance:.1f} pixels")

print()

# Save telemetry
if CONFIG['save_telemetry']:
    telemetry_data = {
        'metadata': {
            'total_frames': tracker.frame_count,
            'fps': fps,
            'resolution': [frame_width, frame_height],
        },
        'cars': {
            name: [
                {
                    'frame': entry['frame'],
                    'position': entry['position'],
                    'velocity': entry['velocity']
                }
                for entry in data
            ]
            for name, data in tracker.telemetry.items()
        },
        'overtakes': [
            {
                'frame': ov.frame,
                'timestamp': ov.timestamp,
                'overtaking_car': ov.overtaking_car,
                'overtaken_car': ov.overtaken_car,
                'confidence': ov.confidence
            }
            for ov in tracker.overtakes
        ]
    }
    
    with open(CONFIG['telemetry_path'], 'w') as f:
        json.dump(telemetry_data, f, indent=2)
    
    print(f"💾 Telemetry saved to: {CONFIG['telemetry_path']}")

# %%
# ============================================================================
# CELL 8: Visualization Plots
# ============================================================================

if tracker.telemetry:
    print("\n📊 Generating position plots...\n")
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Position over time
    ax1 = axes[0]
    for car_name, data in tracker.telemetry.items():
        frames = [entry['frame'] for entry in data]
        positions = [entry['position'][0] for entry in data]  # X position
        
        if len(frames) > 10:
            ax1.plot(frames, positions, label=car_name, linewidth=2)
    
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('X Position (pixels)')
    ax1.set_title('Car Positions Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mark overtakes
    for overtake in tracker.overtakes:
        ax1.axvline(x=overtake.frame, color='red', linestyle='--', alpha=0.5)
        ax1.text(overtake.frame, ax1.get_ylim()[1], '🏁', 
                ha='center', va='bottom', fontsize=12)
    
    # Speed estimates
    ax2 = axes[1]
    for car_name, data in tracker.telemetry.items():
        frames = [entry['frame'] for entry in data]
        speeds = [np.linalg.norm(entry['velocity']) for entry in data]
        
        if len(frames) > 10:
            ax2.plot(frames, speeds, label=car_name, linewidth=2)
    
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Speed (pixels/frame)')
    ax2.set_title('Relative Speed Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('race_analysis_plots.png', dpi=150, bbox_inches='tight')
    print("💾 Saved race_analysis_plots.png")
    plt.show()

# %%
# ============================================================================
# CELL 9: Usage Guide
# ============================================================================

print("\n" + "="*70)
print("📖 F1 RACE TRACKER - USAGE GUIDE")
print("="*70)

print("""
🎯 CONFIGURATION TIPS:

📹 VIDEO INPUT:
   CONFIG['video_path'] = 'race.mp4'  # Specific video
   CONFIG['use_webcam'] = True         # Use webcam for testing

🔍 DETECTION TUNING:
   CONFIG['min_car_area'] = 500        # Smaller for distant cars
   CONFIG['max_car_area'] = 50000      # Larger for close-up shots
   CONFIG['detection_roi'] = (x,y,w,h) # Focus on specific track area

🏁 OVERTAKE SENSITIVITY:
   CONFIG['lateral_threshold'] = 30    # Lower = more sensitive
   CONFIG['overtake_cooldown'] = 60    # Prevent duplicate detections

📊 OUTPUT OPTIONS:
   CONFIG['output_video'] = True       # Save annotated video
   CONFIG['save_telemetry'] = True     # Save position data (JSON)
   CONFIG['generate_timeline'] = True  # Print overtake events

🎨 VISUALIZATION:
   CONFIG['show_trails'] = True        # Car movement trails
   CONFIG['show_speed_estimate'] = True# Speed indicators
   CONFIG['trail_length'] = 30         # Trail length in frames

💡 BEST PRACTICES:
   • Use stable camera angles (broadcast views work best)
   • Avoid rapid camera cuts
   • Good lighting and contrast helps detection
   • Set detection_roi to focus on main racing line
   • Adjust min/max_car_area based on camera distance

🔧 TROUBLESHOOTING:
   • Cars not detected? Adjust min_car_area
   • Too many false detections? Increase min_car_area
   • Missed overtakes? Lower lateral_threshold
   • Too many false overtakes? Increase overtake_cooldown

🚀 MODIFY CONFIG IN CELL 1, THEN RUN FROM CELL 5!
""")

print("="*70)
print("✅ F1 RACE TRACKER READY!")
print("="*70)

# %%
# ============================================================================
# END
# ============================================================================
# F1 RACE POSITION TRACKER V1.0
# Detect overtakes and track car positions in race footage
# ============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
from datetime import timedelta

print("🏁 F1 Race Position Tracker V1.0 Ready!")
print("   • Car Detection & Tracking")
print("   • Overtake Detection")
print("   • Position Analysis")
print("   • Race Timeline Generation")

# %%
# ============================================================================
# CELL 1: Configuration
# ============================================================================

CONFIG = {
    # Video input
    'video_path': None,  # Set to video path or None for file dialog
    'use_webcam': False,  # True to use webcam for testing
    
    # Detection settings
    'min_car_area': 500,  # Minimum pixels for car detection
    'max_car_area': 50000,  # Maximum pixels for car detection
    'detection_roi': None,  # (x, y, w, h) or None for full frame
    
    # Tracking settings
    'max_track_age': 30,  # Frames before track is lost
    'min_track_confidence': 5,  # Minimum frames to confirm track
    'position_threshold': 50,  # Pixels to consider position change
    
    # Overtake detection
    'overtake_cooldown': 60,  # Frames between same overtake events
    'lateral_threshold': 30,  # Horizontal movement for overtake
    'overtake_duration': 10,  # Frames to confirm overtake
    
    # Visualization
    'show_trails': True,  # Show car movement trails
    'trail_length': 30,  # Number of points in trail
    'show_speed_estimate': True,  # Show relative speed
    'output_video': True,  # Save annotated video
    'output_path': 'f1_race_analysis.mp4',
    
    # Analysis
    'save_telemetry': True,  # Save position data to JSON
    'telemetry_path': 'race_telemetry.json',
    'generate_timeline': True,  # Create overtake timeline
}

print("📋 Current Configuration:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")

# %%
# ============================================================================
# CELL 2: Data Structures
# ============================================================================

@dataclass
class CarTrack:
    """Represents a tracked car"""
    id: int
    name: str
    color: Tuple[int, int, int]
    positions: deque  # Recent positions
    bboxes: deque  # Recent bounding boxes
    last_seen: int  # Frame number
    confidence: int  # Tracking confidence
    total_frames: int  # Total frames tracked
    
    def __post_init__(self):
        if not isinstance(self.positions, deque):
            self.positions = deque(maxlen=CONFIG['trail_length'])
        if not isinstance(self.bboxes, deque):
            self.bboxes = deque(maxlen=10)
    
    def update(self, position, bbox, frame_num):
        """Update track with new detection"""
        self.positions.append(position)
        self.bboxes.append(bbox)
        self.last_seen = frame_num
        self.confidence = min(self.confidence + 1, 100)
        self.total_frames += 1
    
    def get_velocity(self):
        """Estimate velocity from recent positions"""
        if len(self.positions) < 2:
            return (0, 0)
        
        recent = list(self.positions)[-5:]
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        return (dx / len(recent), dy / len(recent))
    
    def get_current_position(self):
        """Get most recent position"""
        return self.positions[-1] if self.positions else None

@dataclass
class OvertakeEvent:
    """Represents an overtake"""
    frame: int
    timestamp: float
    overtaking_car: str
    overtaken_car: str
    position_before: Tuple[int, int]
    position_after: Tuple[int, int]
    confidence: float

class RaceTracker:
    """Main tracking system"""
    
    def __init__(self):
        self.tracks: Dict[int, CarTrack] = {}
        self.next_id = 0
        self.frame_count = 0
        self.overtakes: List[OvertakeEvent] = []
        self.last_overtake = defaultdict(int)  # Cooldown tracking
        
        # Car colors for visualization (expand as needed)
        self.car_colors = [
            (255, 0, 0),    # Red
            (0, 0, 255),    # Blue
            (0, 255, 0),    # Green
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]
        
        # Telemetry data
        self.telemetry = defaultdict(list)
    
    def create_track(self, position, bbox):
        """Create new car track"""
        track_id = self.next_id
        color = self.car_colors[track_id % len(self.car_colors)]
        
        track = CarTrack(
            id=track_id,
            name=f"Car {track_id + 1}",
            color=color,
            positions=deque(maxlen=CONFIG['trail_length']),
            bboxes=deque(maxlen=10),
            last_seen=self.frame_count,
            confidence=1,
            total_frames=0
        )
        
        track.update(position, bbox, self.frame_count)
        self.tracks[track_id] = track
        self.next_id += 1
        
        return track_id
    
    def update_tracks(self, detections):
        """Update all tracks with new detections"""
        self.frame_count += 1
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        for track_id, track in self.tracks.items():
            if track.last_seen < self.frame_count - CONFIG['max_track_age']:
                continue
            
            best_match = None
            best_distance = float('inf')
            
            last_pos = track.get_current_position()
            if last_pos is None:
                continue
            
            for i, (position, bbox) in enumerate(detections):
                if i in matched_detections:
                    continue
                
                distance = np.linalg.norm(np.array(position) - np.array(last_pos))
                
                if distance < best_distance and distance < 100:  # Max matching distance
                    best_distance = distance
                    best_match = i
            
            if best_match is not None:
                track.update(detections[best_match][0], detections[best_match][1], self.frame_count)
                matched_tracks.add(track_id)
                matched_detections.add(best_match)
        
        # Create new tracks for unmatched detections
        for i, (position, bbox) in enumerate(detections):
            if i not in matched_detections:
                self.create_track(position, bbox)
        
        # Record telemetry
        for track_id, track in self.tracks.items():
            if track.last_seen == self.frame_count:
                pos = track.get_current_position()
                vel = track.get_velocity()
                self.telemetry[track.name].append({
                    'frame': self.frame_count,
                    'position': pos,
                    'velocity': vel,
                    'bbox': track.bboxes[-1] if track.bboxes else None
                })
    
    def detect_overtakes(self):
        """Detect overtake events between cars"""
        active_tracks = [t for t in self.tracks.values() 
                        if t.last_seen >= self.frame_count - 5 and 
                        t.confidence >= CONFIG['min_track_confidence']]
        
        if len(active_tracks) < 2:
            return
        
        # Check all pairs of cars
        for i, track1 in enumerate(active_tracks):
            for track2 in active_tracks[i+1:]:
                self._check_overtake(track1, track2)
    
    def _check_overtake(self, track1, track2):
        """Check if one car is overtaking another"""
        if len(track1.positions) < 5 or len(track2.positions) < 5:
            return
        
        # Get recent positions
        pos1_old = list(track1.positions)[0]
        pos1_new = list(track1.positions)[-1]
        pos2_old = list(track2.positions)[0]
        pos2_new = list(track2.positions)[-1]
        
        # Check cooldown
        overtake_key = f"{track1.id}-{track2.id}"
        if self.frame_count - self.last_overtake[overtake_key] < CONFIG['overtake_cooldown']:
            return
        
        # Detect position swap (assuming horizontal racing line)
        x1_change = pos1_new[0] - pos1_old[0]
        x2_change = pos2_new[0] - pos2_old[0]
        
        # Check if cars crossed positions
        was_behind = pos1_old[0] < pos2_old[0]
        now_ahead = pos1_new[0] > pos2_new[0]
        
        lateral_movement = abs(x1_change - x2_change)
        
        if was_behind and now_ahead and lateral_movement > CONFIG['lateral_threshold']:
            # Track1 overtook Track2
            overtake = OvertakeEvent(
                frame=self.frame_count,
                timestamp=self.frame_count / 30.0,  # Assuming 30 fps
                overtaking_car=track1.name,
                overtaken_car=track2.name,
                position_before=pos1_old,
                position_after=pos1_new,
                confidence=min(track1.confidence, track2.confidence) / 100.0
            )
            
            self.overtakes.append(overtake)
            self.last_overtake[overtake_key] = self.frame_count
            
            print(f"🏁 OVERTAKE! {track1.name} passed {track2.name} at frame {self.frame_count}")

print("✅ Data structures initialized")

# %%
# ============================================================================
# CELL 3: Car Detection
# ============================================================================

class CarDetector:
    """Detect cars in video frames"""
    
    def __init__(self):
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=True
        )
        
        # Color ranges for car detection (expand for specific teams)
        self.color_ranges = [
            # Red (Ferrari, Red Bull)
            ((0, 100, 100), (10, 255, 255)),
            ((170, 100, 100), (180, 255, 255)),
            # Blue (Mercedes, Alpine)
            ((100, 100, 100), (130, 255, 255)),
            # Silver/Gray
            ((0, 0, 100), (180, 50, 200)),
        ]
    
    def detect_by_motion(self, frame):
        """Detect cars using motion/background subtraction"""
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if CONFIG['min_car_area'] < area < CONFIG['max_car_area']:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (cars are wider than tall)
                aspect_ratio = w / h if h > 0 else 0
                if 0.8 < aspect_ratio < 4.0:
                    center = (x + w//2, y + h//2)
                    detections.append((center, (x, y, w, h)))
        
        return detections
    
    def detect_by_color(self, frame):
        """Detect cars using color information"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if CONFIG['min_car_area'] < area < CONFIG['max_car_area']:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w//2, y + h//2)
                detections.append((center, (x, y, w, h)))
        
        return detections
    
    def detect(self, frame):
        """Combine detection methods"""
        motion_detections = self.detect_by_motion(frame)
        color_detections = self.detect_by_color(frame)
        
        # Merge detections (simple union for now)
        all_detections = motion_detections + color_detections
        
        # Remove duplicates (detections close to each other)
        unique_detections = []
        for det in all_detections:
            is_duplicate = False
            for unique_det in unique_detections:
                distance = np.linalg.norm(np.array(det[0]) - np.array(unique_det[0]))
                if distance < 50:  # Merge threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_detections.append(det)
        
        return unique_detections

print("✅ Car detector initialized")

# %%
# ============================================================================
# CELL 4: Visualization
# ============================================================================

class RaceVisualizer:
    """Visualize tracking and overtakes"""
    
    def __init__(self, tracker):
        self.tracker = tracker
        self.overtake_flash = {}  # Flash effect for overtakes
    
    def draw_track(self, frame, track):
        """Draw single car track"""
        if not track.positions:
            return
        
        # Draw trail
        if CONFIG['show_trails'] and len(track.positions) > 1:
            points = np.array(list(track.positions), dtype=np.int32)
            for i in range(len(points) - 1):
                alpha = (i + 1) / len(points)
                thickness = max(1, int(3 * alpha))
                cv2.line(frame, tuple(points[i]), tuple(points[i+1]), 
                        track.color, thickness)
        
        # Draw current position
        pos = track.get_current_position()
        if pos and track.bboxes:
            x, y, w, h = track.bboxes[-1]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), track.color, 2)
            
            # Draw car name and info
            label = track.name
            if CONFIG['show_speed_estimate']:
                vel = track.get_velocity()
                speed = np.linalg.norm(vel)
                label += f" ({speed:.1f})"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y), 
                         track.color, -1)
            cv2.putText(frame, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw position marker
            cv2.circle(frame, pos, 5, track.color, -1)
            cv2.circle(frame, pos, 7, (255, 255, 255), 2)
    
    def draw_overtake_notification(self, frame, overtake):
        """Flash overtake notification"""
        if self.tracker.frame_count - overtake.frame < 60:  # Show for 2 seconds
            h, w = frame.shape[:2]
            
            # Create notification box
            text = f"OVERTAKE! {overtake.overtaking_car} > {overtake.overtaken_car}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_BOLD, 1.0, 2)
            
            box_x = (w - text_w) // 2 - 20
            box_y = 50
            
            # Pulsing effect
            alpha = 0.5 + 0.5 * np.sin((self.tracker.frame_count - overtake.frame) * 0.3)
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, box_y), (box_x + text_w + 40, box_y + text_h + 20),
                         (0, 255, 0), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            cv2.putText(frame, text, (box_x + 20, box_y + text_h + 10),
                       cv2.FONT_HERSHEY_BOLD, 1.0, (255, 255, 255), 2)
    
    def draw_race_info(self, frame):
        """Draw race statistics"""
        h, w = frame.shape[:2]
        
        # Info panel
        info_lines = [
            f"Frame: {self.tracker.frame_count}",
            f"Cars: {len([t for t in self.tracker.tracks.values() if t.confidence > CONFIG['min_track_confidence']])}",
            f"Overtakes: {len(self.tracker.overtakes)}",
        ]
        
        y_offset = h - 100
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, line, (20, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    def draw_frame(self, frame):
        """Draw all visualizations"""
        viz_frame = frame.copy()
        
        # Draw all tracks
        for track in self.tracker.tracks.values():
            if track.confidence >= CONFIG['min_track_confidence']:
                self.draw_track(viz_frame, track)
        
        # Draw recent overtakes
        for overtake in self.tracker.overtakes[-5:]:  # Last 5 overtakes
            self.draw_overtake_notification(viz_frame, overtake)
        
        # Draw info panel
        self.draw_race_info(viz_frame)
        
        return viz_frame

print("✅ Visualizer initialized")

# %%
# ============================================================================
# CELL 5: Load Video
# ============================================================================

import tkinter as tk
from tkinter import filedialog

def select_video():
    """Open file dialog to select video"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    print("📂 Select race video...")
    video_path = filedialog.askopenfilename(
        title="Select Race Video",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return video_path

# Get video source
if CONFIG['use_webcam']:
    video_source = 0
    print("📹 Using webcam")
elif CONFIG['video_path']:
    video_source = CONFIG['video_path']
    print(f"📹 Using video: {CONFIG['video_path']}")
else:
    video_source = select_video()
    if not video_source:
        print("❌ No video selected")
        raise FileNotFoundError("No video selected")

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"❌ Failed to open video: {video_source}")
    raise IOError("Failed to open video")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"✅ Video loaded:")
print(f"   Resolution: {frame_width}x{frame_height}")
print(f"   FPS: {fps}")
print(f"   Total frames: {total_frames}")

# Setup output video writer
if CONFIG['output_video']:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(CONFIG['output_path'], fourcc, fps, 
                          (frame_width, frame_height))
    print(f"✅ Output video: {CONFIG['output_path']}")

# %%
# ============================================================================
# CELL 6: Process Video
# ============================================================================

print("\n" + "="*70)
print("🏁 STARTING RACE ANALYSIS")
print("="*70)
print("\nPress 'q' to stop, 'p' to pause, 's' to save current frame\n")

# Initialize systems
tracker = RaceTracker()
detector = CarDetector()
visualizer = RaceVisualizer(tracker)

frame_skip = 1  # Process every Nth frame (1 = process all)
paused = False

try:
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("\n✅ Video processing complete")
                break
            
            # Process frame
            if tracker.frame_count % frame_skip == 0:
                # Apply ROI if configured
                process_frame = frame
                if CONFIG['detection_roi']:
                    x, y, w, h = CONFIG['detection_roi']
                    process_frame = frame[y:y+h, x:x+w]
                
                # Detect cars
                detections = detector.detect(process_frame)
                
                # Update tracks
                tracker.update_tracks(detections)
                
                # Detect overtakes
                tracker.detect_overtakes()
            
            # Visualize
            viz_frame = visualizer.draw_frame(frame)
            
            # Save frame
            if CONFIG['output_video']:
                out.write(viz_frame)
            
            # Display
            display_frame = cv2.resize(viz_frame, (1280, 720))
            cv2.imshow('F1 Race Tracker', display_frame)
            
            # Progress
            if tracker.frame_count % 30 == 0:
                progress = (tracker.frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {progress:.1f}% | Frame: {tracker.frame_count} | Cars: {len(tracker.tracks)} | Overtakes: {len(tracker.overtakes)}")
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n⏹️ Stopped by user")
            break
        elif key == ord('p'):
            paused = not paused
            print("⏸️ Paused" if paused else "▶️ Resumed")
        elif key == ord('s'):
            cv2.imwrite(f'frame_{tracker.frame_count}.jpg', viz_frame)
            print(f"💾 Saved frame_{tracker.frame_count}.jpg")

except KeyboardInterrupt:
    print("\n⏹️ Interrupted by user")

finally:
    cap.release()
    if CONFIG['output_video']:
        out.release()
    cv2.destroyAllWindows()

print("\n" + "="*70)
print("📊 RACE ANALYSIS COMPLETE")
print("="*70)

# %%
# ============================================================================
# CELL 7: Generate Reports
# ============================================================================

print("\n📈 Generating analysis reports...\n")

# Overtake Timeline
if CONFIG['generate_timeline'] and tracker.overtakes:
    print("🏁 OVERTAKE TIMELINE:")
    print("-" * 70)
    
    for i, overtake in enumerate(tracker.overtakes):
        timestamp = timedelta(seconds=overtake.timestamp)
        print(f"{i+1}. [{timestamp}] {overtake.overtaking_car} overtook {overtake.overtaken_car}")
        print(f"   Confidence: {overtake.confidence:.2f}")
    
    print()

# Track Statistics
print("🏎️ CAR STATISTICS:")
print("-" * 70)

for track in sorted(tracker.tracks.values(), key=lambda t: t.total_frames, reverse=True):
    if track.confidence >= CONFIG['min_track_confidence']:
        print(f"{track.name}:")
        print(f"   Frames tracked: {track.total_frames}")
        print(f"   Confidence: {track.confidence}")
        
        if track.positions:
            positions = list(track.positions)
            total_distance = sum(
                np.linalg.norm(np.array(positions[i+1]) - np.array(positions[i]))
                for i in range(len(positions) - 1)
            )
            print(f"   Distance traveled: {total_distance:.1f} pixels")

print()

# Save telemetry
if CONFIG['save_telemetry']:
    telemetry_data = {
        'metadata': {
            'total_frames': tracker.frame_count,
            'fps': fps,
            'resolution': [frame_width, frame_height],
        },
        'cars': {
            name: [
                {
                    'frame': entry['frame'],
                    'position': entry['position'],
                    'velocity': entry['velocity']
                }
                for entry in data
            ]
            for name, data in tracker.telemetry.items()
        },
        'overtakes': [
            {
                'frame': ov.frame,
                'timestamp': ov.timestamp,
                'overtaking_car': ov.overtaking_car,
                'overtaken_car': ov.overtaken_car,
                'confidence': ov.confidence
            }
            for ov in tracker.overtakes
        ]
    }
    
    with open(CONFIG['telemetry_path'], 'w') as f:
        json.dump(telemetry_data, f, indent=2)
    
    print(f"💾 Telemetry saved to: {CONFIG['telemetry_path']}")

# %%
# ============================================================================
# CELL 8: Visualization Plots
# ============================================================================

if tracker.telemetry:
    print("\n📊 Generating position plots...\n")
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Position over time
    ax1 = axes[0]
    for car_name, data in tracker.telemetry.items():
        frames = [entry['frame'] for entry in data]
        positions = [entry['position'][0] for entry in data]  # X position
        
        if len(frames) > 10:
            ax1.plot(frames, positions, label=car_name, linewidth=2)
    
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('X Position (pixels)')
    ax1.set_title('Car Positions Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mark overtakes
    for overtake in tracker.overtakes:
        ax1.axvline(x=overtake.frame, color='red', linestyle='--', alpha=0.5)
        ax1.text(overtake.frame, ax1.get_ylim()[1], '🏁', 
                ha='center', va='bottom', fontsize=12)
    
    # Speed estimates
    ax2 = axes[1]
    for car_name, data in tracker.telemetry.items():
        frames = [entry['frame'] for entry in data]
        speeds = [np.linalg.norm(entry['velocity']) for entry in data]
        
        if len(frames) > 10:
            ax2.plot(frames, speeds, label=car_name, linewidth=2)
    
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Speed (pixels/frame)')
    ax2.set_title('Relative Speed Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('race_analysis_plots.png', dpi=150, bbox_inches='tight')
    print("💾 Saved race_analysis_plots.png")
    plt.show()

# %%
# ============
# %%
