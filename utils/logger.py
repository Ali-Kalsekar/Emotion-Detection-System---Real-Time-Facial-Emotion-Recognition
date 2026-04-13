"""
Logging Module
Logs emotion predictions to CSV file for analysis and auditing
"""
import csv
import os
from datetime import datetime
from collections import defaultdict


class EmotionLogger:
    """Logs emotion predictions to file and maintains statistics."""
    
    def __init__(self, log_file="output/emotion_log.csv"):
        """
        Initialize emotion logger.
        
        Args:
            log_file: Path to CSV file for logging
        """
        self.log_file = log_file
        self.emotion_counts = defaultdict(int)
        self.total_predictions = 0
        self.log_interval = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        
        # Initialize CSV file with headers
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'emotion',
                    'confidence',
                    'num_faces',
                    'frame_number'
                ])
    
    def log_prediction(self, emotion, confidence, num_faces=1, frame_number=0):
        """
        Log a single emotion prediction.
        
        Args:
            emotion: Predicted emotion label
            confidence: Confidence score (0-1)
            num_faces: Number of faces detected in frame
            frame_number: Current frame number
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Update statistics
        self.emotion_counts[emotion] += 1
        self.total_predictions += 1
        
        # Log to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                emotion,
                f"{confidence:.4f}",
                num_faces,
                frame_number
            ])
    
    def get_statistics(self):
        """
        Get emotion prediction statistics.
        
        Returns:
            dict: Statistics with emotion counts and percentages
        """
        stats = {}
        for emotion, count in self.emotion_counts.items():
            percentage = (count / self.total_predictions * 100) if self.total_predictions > 0 else 0
            stats[emotion] = {
                'count': count,
                'percentage': percentage
            }
        
        stats['total'] = self.total_predictions
        return stats
    
    def print_statistics(self):
        """Print statistics to console."""
        stats = self.get_statistics()
        print("\n" + "="*50)
        print("EMOTION STATISTICS")
        print("="*50)
        print(f"Total Predictions: {stats['total']}")
        print("-"*50)
        
        sorted_emotions = sorted(
            [(e, s['count'], s['percentage']) for e, s in stats.items() if e != 'total'],
            key=lambda x: x[1],
            reverse=True
        )
        
        for emotion, count, percentage in sorted_emotions:
            print(f"{emotion.capitalize():15} | Count: {count:6} | {percentage:6.2f}%")
        print("="*50 + "\n")
    
    def reset(self):
        """Reset statistics."""
        self.emotion_counts.clear()
        self.total_predictions = 0
