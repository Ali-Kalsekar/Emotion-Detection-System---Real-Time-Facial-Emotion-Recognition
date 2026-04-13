"""
Emotion Detection System
Real-time emotion recognition from webcam or video
Main application entry point
"""
import cv2
import yaml
import sys
import os
from collections import deque
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_detection.face_detector import FaceDetector
from emotion_recognition.emotion_classifier import EmotionClassifier
from utils.fps import FPSCounter
from utils.logger import EmotionLogger
from utils.draw import Visualizer


class EmotionDetectionSystem:
    """Main emotion detection system combining face detection and emotion classification."""
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize emotion detection system.
        
        Args:
            config_path: Path to configuration file
        """
        # Make config path absolute if it's relative
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("Initializing Emotion Detection System...")
        
        # Initialize components
        print("  - Initializing face detector...")
        self.face_detector = FaceDetector(
            method=self.config['detection']['face_detector'],
            min_face_size=self.config['detection']['min_face_size']
        )
        
        print("  - Initializing emotion classifier...")
        # Make model path absolute if it's relative
        model_path = self.config['model']['path']
        if not os.path.isabs(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, model_path)
        
        self.emotion_classifier = EmotionClassifier(
            model_path=model_path,
            input_size=self.config['model']['input_size']
        )
        
        print("  - Initializing visualization...")
        self.visualizer = Visualizer(
            font_scale=self.config['display']['font_scale'],
            thickness=self.config['display']['thickness']
        )
        
        print("  - Initializing logger...")
        # Make log file path absolute if it's relative
        log_file = self.config['logging']['output_file']
        if not os.path.isabs(log_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_file = os.path.join(script_dir, log_file)
        
        self.logger = EmotionLogger(
            log_file=log_file
        )
        
        print("  - Initializing FPS counter...")
        self.fps_counter = FPSCounter()
        
        # Initialize emotion history for statistics
        self.emotion_history = deque(maxlen=self.config['features']['history_window'])
        
        print("System initialized successfully!\n")
    
    def _smooth_confidence(self, emotion, confidence):
        """
        Apply confidence smoothing using history window.
        
        Args:
            emotion: Emotion label
            confidence: Current confidence
        
        Returns:
            float: Smoothed confidence
        """
        if not self.config['features']['confidence_smoothing_enabled']:
            return confidence
        
        # This is handled by the classifier
        return confidence
    
    def _update_emotion_history(self, emotion):
        """Update emotion history for statistics."""
        self.emotion_history.append(emotion)
        self.logger.emotion_counts[emotion] += 1
        self.logger.total_predictions += 1
    
    def _get_emotion_statistics(self):
        """Get current emotion statistics."""
        stats = {}
        for emotion in self.emotion_classifier.EMOTIONS:
            count = self.logger.emotion_counts[emotion]
            percentage = (count / self.logger.total_predictions * 100) if self.logger.total_predictions > 0 else 0
            stats[emotion] = {
                'count': count,
                'percentage': percentage
            }
        stats['total'] = self.logger.total_predictions
        return stats
    
    def run(self, source=0):
        """
        Run emotion detection system.
        
        Args:
            source: Video source (0 for webcam, or path to video file)
        """
        print("="*60)
        print(self.config['display']['window_title'])
        print("="*60)
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Show statistics")
        print("  'c' - Clear statistics")
        print("  'f' - Toggle face detector method")
        print("  'h' - Toggle history tracking")
        print("="*60 + "\n")
        
        # Initialize video capture
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video source {source}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['resolution']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['resolution']['height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['camera']['fps'])
        
        # Get GPU acceleration preference
        if self.config['features']['gpu_acceleration']:
            print("GPU acceleration enabled\n")
        
        frame_count = 0
        running = True
        
        print("Starting real-time emotion detection...\n")
        
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for better webcam experience
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = self.face_detector.detect(frame)
            
            # Process each face
            faces_data = []
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Predict emotion
                emotion, confidence = self.emotion_classifier.predict_emotion(face_roi)
                
                # Apply confidence threshold
                if confidence < self.config['detection']['confidence_threshold']:
                    emotion = "unknown"
                
                # Update history
                if emotion != "unknown":
                    self._update_emotion_history(emotion)
                
                # Log prediction
                if self.config['logging']['enabled'] and frame_count % self.config['logging']['log_interval'] == 0:
                    if emotion != "unknown":
                        self.logger.log_prediction(emotion, confidence, len(faces), frame_count)
                
                faces_data.append((x, y, w, h, emotion, confidence))
            
            # Draw faces
            frame = self.visualizer.draw_multiple_faces(frame, faces_data)
            
            # Draw FPS
            self.fps_counter.update()
            if self.config['display']['show_fps']:
                fps = self.fps_counter.get_fps()
                frame = self.visualizer.draw_fps(frame, fps)
            
            # Draw face count
            frame = self.visualizer.draw_face_count(frame, len(faces))
            
            # Draw statistics
            if self.config['features']['show_statistics'] and self.logger.total_predictions > 0:
                stats = self._get_emotion_statistics()
                frame = self.visualizer.draw_statistics(frame, stats, top_offset=60)
            
            # Display frame
            cv2.imshow(self.config['display']['window_title'], frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                running = False
            elif key == ord('s'):
                self.logger.print_statistics()
            elif key == ord('c'):
                self.logger.reset()
                self.emotion_classifier.reset_history()
                print("Statistics cleared\n")
            elif key == ord('f'):
                # Toggle face detector method
                current_method = self.face_detector.get_method()
                new_method = "dnn" if current_method == "haarcascade" else "haarcascade"
                try:
                    self.face_detector.set_method(new_method)
                    print(f"Switched to {new_method} face detector\n")
                except Exception as e:
                    print(f"Error switching detector: {e}\n")
            elif key == ord('h'):
                # Toggle history tracking
                current_state = self.config['features']['emotion_history_enabled']
                self.config['features']['emotion_history_enabled'] = not current_state
                print(f"History tracking {'enabled' if not current_state else 'disabled'}\n")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "="*60)
        print("SYSTEM SHUTDOWN")
        print("="*60)
        self.logger.print_statistics()
        
        # Print total frames processed
        fps = self.fps_counter.get_fps()
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {fps:.1f}\n")


def main():
    """Main entry point."""
    try:
        # Get script directory and construct paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check if model exists
        config_path = os.path.join(script_dir, "config/config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_path = config['model']['path']
        if not os.path.isabs(model_path):
            model_path = os.path.join(script_dir, model_path)
        
        if not os.path.exists(model_path):
            print(f"\n{'='*60}")
            print("WARNING: Emotion model not found!")
            print(f"{'='*60}")
            print(f"\nModel path: {model_path}")
            print("\nTo train a model:")
            print("  1. First, collect training data:")
            print("     python dataset/collect_data.py")
            print("  2. Then train the model:")
            print("     python training/train_model.py")
            print("\nThe system will run with random predictions until a model is trained.\n")
        
        # Initialize system (pass absolute path)
        system = EmotionDetectionSystem(config_path=os.path.join(script_dir, "config/config.yaml"))
        
        # Run system (0 = default webcam)
        system.run(source=0)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
