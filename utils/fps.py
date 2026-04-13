"""
FPS Counter Module
Tracks and displays frames per second in real-time
"""
import time
from collections import deque


class FPSCounter:
    """Calculate and track FPS for real-time performance monitoring."""
    
    def __init__(self, window_size=30):
        """
        Initialize FPS counter.
        
        Args:
            window_size: Number of frames to average FPS over
        """
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        self.start_time = time.time()
    
    def update(self):
        """Update FPS counter with current timestamp."""
        self.timestamps.append(time.time())
    
    def get_fps(self):
        """
        Calculate current FPS.
        
        Returns:
            float: Current FPS value
        """
        if len(self.timestamps) < 2:
            return 0.0
        
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff == 0:
            return 0.0
        
        fps = (len(self.timestamps) - 1) / time_diff
        return fps
    
    def get_average_fps(self):
        """
        Get average FPS over window.
        
        Returns:
            float: Average FPS value
        """
        return self.get_fps()
    
    def reset(self):
        """Reset FPS counter."""
        self.timestamps.clear()
        self.start_time = time.time()
