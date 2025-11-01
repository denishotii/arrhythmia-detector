"""
Remote Photoplethysmography (rPPG) Extraction from Video
Extracts heart rate signals from facial video for arrhythmia detection.
"""

import numpy as np
import cv2
from scipy import signal
from scipy.ndimage import gaussian_filter1d


def extract_roi(frame, face_landmarks=None):
    """
    Extract Region of Interest (ROI) for rPPG extraction.
    Best ROI: forehead region (stable, good blood perfusion).
    
    Args:
        frame: Video frame (BGR image)
        face_landmarks: Optional face landmarks for precise ROI
        
    Returns:
        roi: Region of interest
        roi_coords: Coordinates of ROI
    """
    h, w = frame.shape[:2]
    
    # Default: use forehead region (top 20% of face)
    # This is a simplified version - in production, use face detection landmarks
    roi_y1 = int(h * 0.1)  # Top 10% of frame
    roi_y2 = int(h * 0.3)  # Top 30% of frame
    roi_x1 = int(w * 0.3)  # Middle-left
    roi_x2 = int(w * 0.7)  # Middle-right
    
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_coords = (roi_x1, roi_y1, roi_x2, roi_y2)
    
    return roi, roi_coords


def extract_rppg_signal(video_frames, roi_extractor=extract_roi, fps=30):
    """
    Extract rPPG signal from video frames.
    
    This function analyzes color changes in facial skin due to blood perfusion.
    The green channel is typically most sensitive to blood volume changes.
    
    Args:
        video_frames: List of video frames (BGR format)
        roi_extractor: Function to extract ROI from frame
        fps: Frames per second of video
        
    Returns:
        rppg_signal: Extracted PPG-like signal (1D array)
        sampling_rate: Effective sampling rate (equals fps)
    """
    if not video_frames:
        return None, None
    
    # Extract ROI mean values for each frame
    roi_values = []
    
    for frame in video_frames:
        roi, _ = roi_extractor(frame)
        if roi.size > 0:
            # Extract green channel (most sensitive to blood volume changes)
            green_channel = roi[:, :, 1]  # BGR format: index 1 is green
            mean_value = np.mean(green_channel)
            roi_values.append(mean_value)
        else:
            roi_values.append(0)
    
    rppg_signal = np.array(roi_values)
    
    # Apply signal processing to enhance rPPG
    rppg_signal = enhance_rppg_signal(rppg_signal, fps)
    
    return rppg_signal, fps


def enhance_rppg_signal(signal_data, sampling_rate=30):
    """
    Enhance rPPG signal quality.
    
    Steps:
    1. Remove DC component (detrend)
    2. Bandpass filter (0.7-4 Hz: typical heart rate range 42-240 bpm)
    3. Smooth with Gaussian filter
    
    Args:
        signal_data: Raw rPPG signal
        sampling_rate: Sampling rate in Hz
        
    Returns:
        enhanced_signal: Processed signal
    """
    # Remove DC component (mean)
    signal_centered = signal_data - np.mean(signal_data)
    
    # Bandpass filter: 0.7-4 Hz (heart rate range)
    nyquist = sampling_rate / 2
    low = 0.7 / nyquist
    high = 4.0 / nyquist
    
    try:
        b, a = signal.butter(4, [low, high], btype='band')
        signal_filtered = signal.filtfilt(b, a, signal_centered)
    except:
        # If filtering fails, use original signal
        signal_filtered = signal_centered
    
    # Smooth with Gaussian filter
    signal_smoothed = gaussian_filter1d(signal_filtered, sigma=2)
    
    return signal_smoothed


def extract_rppg_from_video_file(video_path, fps=None, duration=None):
    """
    Extract rPPG signal from video file.
    
    Args:
        video_path: Path to video file
        fps: Target FPS (if None, uses video's native FPS)
        duration: Extract only first N seconds (if None, uses entire video)
        
    Returns:
        rppg_signal: Extracted signal
        sampling_rate: Effective sampling rate
        frame_count: Number of frames processed
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps is None:
        fps = video_fps
    
    # Calculate frames to extract
    if duration:
        frames_to_extract = int(duration * video_fps)
        frames_to_extract = min(frames_to_extract, total_frames)
    else:
        frames_to_extract = total_frames
    
    frames = []
    frame_count = 0
    
    while len(frames) < frames_to_extract:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize if needed (for faster processing)
        if frame.shape[0] > 480:
            scale = 480 / frame.shape[0]
            new_width = int(frame.shape[1] * scale)
            frame = cv2.resize(frame, (new_width, 480))
        
        frames.append(frame)
        frame_count += 1
        
        # Skip frames if video FPS > target FPS
        if video_fps > fps:
            skip = video_fps / fps
            for _ in range(int(skip) - 1):
                cap.read()
    
    cap.release()
    
    # Extract rPPG signal
    rppg_signal, sampling_rate = extract_rppg_signal(frames, fps=fps)
    
    return rppg_signal, sampling_rate, frame_count


def extract_rppg_from_api_frames(video_frames, fps=30):
    """
    Extract rPPG from frames received from Cloud API.
    
    This is the function to use when receiving frames from the challenge's Cloud API.
    
    Args:
        video_frames: List of frames from API (format: numpy arrays or base64 encoded)
        fps: Frames per second
        
    Returns:
        rppg_signal: Extracted signal
        sampling_rate: Sampling rate
    """
    # Convert API frames to OpenCV format if needed
    processed_frames = []
    for frame in video_frames:
        if isinstance(frame, str):
            # If base64 encoded, decode it
            import base64
            frame_bytes = base64.b64decode(frame)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(frame, np.ndarray):
            # Already numpy array
            pass
        else:
            continue
        processed_frames.append(frame)
    
    # Extract rPPG
    rppg_signal, sampling_rate = extract_rppg_signal(processed_frames, fps=fps)
    
    return rppg_signal, sampling_rate


def real_time_rppg_buffer(buffer_duration=30, fps=30):
    """
    Create a buffer for real-time rPPG extraction.
    Maintains a sliding window of video frames for continuous monitoring.
    
    Args:
        buffer_duration: Duration of buffer in seconds (default: 30s for model input)
        fps: Frames per second
        
    Returns:
        buffer: Buffer object with add_frame() and get_signal() methods
    """
    max_frames = int(buffer_duration * fps)
    
    class RPPGBuffer:
        def __init__(self):
            self.frames = []
            self.max_frames = max_frames
            
        def add_frame(self, frame):
            """Add new frame to buffer."""
            self.frames.append(frame)
            # Keep only last max_frames
            if len(self.frames) > self.max_frames:
                self.frames.pop(0)
        
        def get_signal(self):
            """Extract rPPG signal from current buffer."""
            if len(self.frames) < 10:  # Need minimum frames
                return None, None
            return extract_rppg_signal(self.frames, fps=fps)
        
        def is_ready(self):
            """Check if buffer has enough data for prediction."""
            return len(self.frames) >= max_frames
    
    return RPPGBuffer()


if __name__ == '__main__':
    # Example usage
    print("rPPG Extractor for Camera-based Arrhythmia Detection")
    print("=" * 60)
    print("\nThis module extracts heart rate signals from facial video.")
    print("Use with video files or Cloud API frames.")
    print("\nExample usage:")
    print("  rppg_signal, fps = extract_rppg_from_video_file('video.mp4')")
    print("  rppg_signal, fps = extract_rppg_from_api_frames(api_frames, fps=30)")

