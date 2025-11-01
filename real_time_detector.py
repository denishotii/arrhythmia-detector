"""
Real-time Arrhythmia Detection from Video
Processes video frames in real-time to detect arrhythmia.
"""

import numpy as np
from video_rppg_extractor import real_time_rppg_buffer, extract_rppg_from_api_frames
from data_preprocessing import preprocess_signal
from model import ArrhythmiaPredictor
import time


class RealTimeArrhythmiaDetector:
    """
    Real-time arrhythmia detector from video stream.
    Processes video frames, extracts rPPG, and predicts arrhythmia.
    """
    
    def __init__(self, model_path='model.pkl', buffer_duration=30, fps=30):
        """
        Initialize real-time detector.
        
        Args:
            model_path: Path to trained model
            buffer_duration: Duration of buffer in seconds (default: 30s)
            fps: Expected frames per second
        """
        # Load model
        self.predictor = ArrhythmiaPredictor()
        self.predictor.load(model_path)
        
        # Create rPPG buffer
        self.buffer = real_time_rppg_buffer(buffer_duration, fps)
        self.fps = fps
        self.buffer_duration = buffer_duration
        
        # Prediction history
        self.prediction_history = []
        self.alert_threshold = 0.7  # Probability threshold for alert
        
    def add_frame(self, frame):
        """
        Add new video frame to processing buffer.
        
        Args:
            frame: Video frame (numpy array or API format)
        """
        self.buffer.add_frame(frame)
    
    def process_and_predict(self):
        """
        Process current buffer and make prediction.
        
        Returns:
            prediction: 0 (normal) or 1 (arrhythmia)
            probability: Confidence score (0-1)
            rppg_signal: Extracted rPPG signal (for visualization)
            None if buffer not ready
        """
        if not self.buffer.is_ready():
            return None
        
        # Extract rPPG signal from buffer
        rppg_signal, sampling_rate = self.buffer.get_signal()
        
        if rppg_signal is None or len(rppg_signal) < 100:
            return None
        
        try:
            # Preprocess signal (extract features)
            features = preprocess_signal(rppg_signal, sampling_rate=sampling_rate)
            features = features.reshape(1, -1)
            
            # Predict
            prediction, probability = self.predictor.predict(features)
            
            # Store in history
            self.prediction_history.append({
                'timestamp': time.time(),
                'prediction': prediction[0],
                'probability': probability[0],
                'signal_length': len(rppg_signal)
            })
            
            # Keep only last 60 predictions (2 minutes at 1 prediction per 2 seconds)
            if len(self.prediction_history) > 60:
                self.prediction_history.pop(0)
            
            return {
                'prediction': prediction[0],
                'probability': probability[0],
                'signal': rppg_signal,
                'sampling_rate': sampling_rate,
                'status': 'ARRHYTHMIA' if prediction[0] == 1 else 'NORMAL',
                'confidence': probability[0]
            }
            
        except Exception as e:
            print(f"Error processing signal: {e}")
            return None
    
    def check_alert(self, recent_predictions=5):
        """
        Check if alert should be triggered based on recent predictions.
        
        Args:
            recent_predictions: Number of recent predictions to check
            
        Returns:
            alert: True if alert should be triggered
            details: Alert details
        """
        if len(self.prediction_history) < recent_predictions:
            return False, None
        
        # Get recent predictions
        recent = self.prediction_history[-recent_predictions:]
        
        # Count arrhythmia detections
        arrhythmia_count = sum(1 for p in recent if p['prediction'] == 1)
        avg_probability = np.mean([p['probability'] for p in recent if p['prediction'] == 1])
        
        # Alert if majority of recent predictions are arrhythmia
        if arrhythmia_count >= (recent_predictions * 0.6) and avg_probability > self.alert_threshold:
            return True, {
                'severity': 'HIGH' if avg_probability > 0.9 else 'MEDIUM',
                'confidence': avg_probability,
                'consecutive_detections': arrhythmia_count,
                'message': f'ARRHYTHMIA DETECTED: {arrhythmia_count}/{recent_predictions} recent predictions indicate arrhythmia'
            }
        
        return False, None
    
    def get_statistics(self):
        """Get detection statistics."""
        if not self.prediction_history:
            return None
        
        total = len(self.prediction_history)
        arrhythmia = sum(1 for p in self.prediction_history if p['prediction'] == 1)
        normal = total - arrhythmia
        
        return {
            'total_predictions': total,
            'arrhythmia_detections': arrhythmia,
            'normal_detections': normal,
            'arrhythmia_rate': arrhythmia / total if total > 0 else 0,
            'avg_confidence': np.mean([p['probability'] for p in self.prediction_history])
        }


def process_video_api_stream(api_client, detector, callback=None):
    """
    Process video stream from Cloud API.
    
    Args:
        api_client: Cloud API client (implemented based on challenge API)
        detector: RealTimeArrhythmiaDetector instance
        callback: Optional callback function(prediction_result)
    """
    print("Starting real-time video processing...")
    print(f"Buffer duration: {detector.buffer_duration}s")
    print(f"Expected FPS: {detector.fps}")
    print("-" * 60)
    
    frame_count = 0
    
    try:
        # Main processing loop
        while True:
            # Get frame from API (implement based on challenge API)
            frame = api_client.get_next_frame()
            
            if frame is None:
                break
            
            # Add frame to buffer
            detector.add_frame(frame)
            frame_count += 1
            
            # Process every 2 seconds (or adjust based on buffer)
            if frame_count % (detector.fps * 2) == 0:
                result = detector.process_and_predict()
                
                if result:
                    # Check for alert
                    alert, alert_details = detector.check_alert()
                    
                    # Print results
                    status = result['status']
                    confidence = result['confidence']
                    
                    print(f"[{frame_count//detector.fps}s] {status} (confidence: {confidence:.2%})")
                    
                    if alert:
                        print(f"⚠️  ALERT: {alert_details['message']}")
                        print(f"   Severity: {alert_details['severity']}")
                    
                    # Call callback if provided
                    if callback:
                        callback(result, alert, alert_details)
    
    except KeyboardInterrupt:
        print("\nStopping video processing...")
    
    # Print final statistics
    stats = detector.get_statistics()
    if stats:
        print("\n" + "=" * 60)
        print("Final Statistics:")
        print(f"  Total predictions: {stats['total_predictions']}")
        print(f"  Arrhythmia detections: {stats['arrhythmia_detections']}")
        print(f"  Normal detections: {stats['normal_detections']}")
        print(f"  Arrhythmia rate: {stats['arrhythmia_rate']:.2%}")
        print(f"  Avg confidence: {stats['avg_confidence']:.2%}")


# Example Cloud API integration template
class CloudAPIClient:
    """
    Template for Cloud API client integration.
    Implement based on challenge's actual API specification.
    """
    
    def __init__(self, api_url, api_key=None):
        """
        Initialize API client.
        
        Args:
            api_url: API endpoint URL
            api_key: API key if required
        """
        self.api_url = api_url
        self.api_key = api_key
        # Initialize connection based on API spec
    
    def get_next_frame(self):
        """
        Get next video frame from API.
        
        Returns:
            frame: Video frame (numpy array or API format)
        """
        # Implement based on challenge API
        # Example:
        # response = requests.get(f"{self.api_url}/frame", headers={"Authorization": self.api_key})
        # frame = decode_frame(response.content)
        # return frame
        pass
    
    def start_stream(self):
        """Start video stream."""
        # Implement based on API
        pass
    
    def stop_stream(self):
        """Stop video stream."""
        # Implement based on API
        pass


if __name__ == '__main__':
    print("Real-time Arrhythmia Detection from Video")
    print("=" * 60)
    print("\nThis module processes video frames in real-time to detect arrhythmia.")
    print("\nUsage:")
    print("  1. Initialize detector: detector = RealTimeArrhythmiaDetector('model.pkl')")
    print("  2. Add frames: detector.add_frame(frame)")
    print("  3. Process: result = detector.process_and_predict()")
    print("  4. Check alerts: alert, details = detector.check_alert()")

