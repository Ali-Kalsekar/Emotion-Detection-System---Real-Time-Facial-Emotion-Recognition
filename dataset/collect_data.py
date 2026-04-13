"""
Data Collection Script
Captures face images for emotion dataset training
"""
import cv2
import os
import yaml
from pathlib import Path


class DataCollector:
    """Collects face images for emotion dataset."""
    
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize data collector.
        
        Args:
            config_path: Path to configuration file
        """
        # Make config path absolute if it's relative
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "..", config_path)
            config_path = os.path.normpath(config_path)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Make output directory path absolute if it's relative
        self.output_dir = self.config['data_collection']['output_dir']
        if not os.path.isabs(self.output_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_dir = os.path.join(script_dir, "..", self.output_dir)
            self.output_dir = os.path.normpath(self.output_dir)
        
        self.image_size = self.config['data_collection']['image_size']
        self.samples_per_emotion = self.config['data_collection']['samples_per_emotion']
        
        # Initialize output directories
        self._init_directories()
    
    def _init_directories(self):
        """Create emotion subdirectories."""
        for emotion in self.EMOTIONS:
            emotion_dir = os.path.join(self.output_dir, emotion)
            os.makedirs(emotion_dir, exist_ok=True)
    
    def _get_next_filename(self, emotion):
        """
        Get next filename for emotion.
        
        Args:
            emotion: Emotion label
        
        Returns:
            str: Full path to next image file
        """
        emotion_dir = os.path.join(self.output_dir, emotion)
        existing_files = len(os.listdir(emotion_dir))
        filename = f"{emotion}_{existing_files + 1:04d}.jpg"
        return os.path.join(emotion_dir, filename)
    
    def _get_completion_stats(self):
        """Get dataset completion statistics."""
        stats = {}
        total_samples = 0
        
        for emotion in self.EMOTIONS:
            emotion_dir = os.path.join(self.output_dir, emotion)
            count = len(os.listdir(emotion_dir))
            total_samples += count
            stats[emotion] = count
        
        return stats, total_samples
    
    def collect_data(self):
        """Main data collection loop."""
        print("\n" + "="*60)
        print("EMOTION DATASET COLLECTOR")
        print("="*60)
        print("\nSupported emotions:")
        for i, emotion in enumerate(self.EMOTIONS, 1):
            print(f"  {i}. {emotion.upper()}")
        
        print("\nInstructions:")
        print("  - Position your face in the frame")
        print("  - Press the emotion number to capture")
        print("  - Press 'q' to quit")
        print("  - Press 's' to show statistics")
        print("="*60 + "\n")
        
        # Initialize camera
        cap = cv2.VideoCapture(self.config['camera']['index'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['resolution']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['resolution']['height'])
        
        frame_count = 0
        running = True
        
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for better webcam experience
            frame = cv2.flip(frame, 1)
            
            # Add instructions
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (10, 10), (w-10, 100), (0, 0, 0), -1)
            
            cv2.putText(frame, "Press number to capture (1-7)", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' for stats, 'q' to quit", (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add emotion list
            y_offset = 120
            for i, emotion in enumerate(self.EMOTIONS, 1):
                cv2.putText(frame, f"{i}. {emotion.upper()}", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 30
            
            cv2.imshow("Data Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                running = False
            elif key == ord('s'):
                self._print_statistics()
            elif 49 <= key <= 55:  # Numbers 1-7
                emotion_idx = key - 49
                if emotion_idx < len(self.EMOTIONS):
                    emotion = self.EMOTIONS[emotion_idx]
                    filename = self._get_next_filename(emotion)
                    
                    # Resize and save image
                    resized = cv2.resize(frame, (self.image_size, self.image_size))
                    cv2.imwrite(filename, resized)
                    
                    print(f"Captured {emotion} - Saved to {filename}")
                    frame_count += 1
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("DATA COLLECTION COMPLETED")
        self._print_statistics()
        print("="*60 + "\n")
    
    def _print_statistics(self):
        """Print dataset statistics."""
        stats, total = self._get_completion_stats()
        
        print("\n" + "-"*60)
        print("DATASET STATISTICS:")
        print("-"*60)
        
        for emotion in self.EMOTIONS:
            count = stats[emotion]
            percentage = (count / self.samples_per_emotion * 100) if self.samples_per_emotion > 0 else 0
            bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
            print(f"{emotion.upper():10} | {count:3}/{self.samples_per_emotion} | {bar} {percentage:6.1f}%")
        
        print("-"*60)
        print(f"{'TOTAL':10} | {total:3} images")
        print("-"*60 + "\n")


def main():
    """Main entry point."""
    collector = DataCollector()
    collector.collect_data()


if __name__ == "__main__":
    main()
