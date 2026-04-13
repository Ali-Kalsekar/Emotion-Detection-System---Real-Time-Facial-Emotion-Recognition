"""
Drawing Module
Handles visualization of face bounding boxes, emotions, and confidence scores
"""
import cv2
import numpy as np
from collections import deque


class Visualizer:
    """Handles visualization of detection results on video frames."""
    
    # Color mapping for emotions
    EMOTION_COLORS = {
        'happy': (0, 255, 0),      # Green
        'sad': (255, 0, 0),         # Blue
        'angry': (0, 0, 255),       # Red
        'surprised': (0, 165, 255), # Orange
        'neutral': (128, 128, 128), # Gray
        'fear': (128, 0, 128),      # Purple
        'disgust': (0, 128, 0),     # Dark Green
    }
    
    def __init__(self, font_scale=0.7, thickness=2):
        """
        Initialize visualizer.
        
        Args:
            font_scale: Font scale for text
            thickness: Thickness of lines and text
        """
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness
    
    def draw_face_box(self, frame, x, y, w, h, emotion, confidence):
        """
        Draw face bounding box with emotion and confidence.
        
        Args:
            frame: Input video frame
            x, y, w, h: Face bounding box coordinates and dimensions
            emotion: Predicted emotion label
            confidence: Confidence score (0-1)
        
        Returns:
            frame: Frame with drawn annotations
        """
        # Get color based on emotion
        color = self.EMOTION_COLORS.get(emotion.lower(), (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Prepare text
        emotion_text = f"{emotion.capitalize()}"
        confidence_text = f"Conf: {confidence:.2%}"
        
        # Draw emotion label background
        label_size = cv2.getTextSize(emotion_text, self.font, self.font_scale, self.thickness)[0]
        cv2.rectangle(
            frame,
            (x, y - 50),
            (x + label_size[0] + 10, y - 10),
            color,
            -1
        )
        
        # Draw emotion label
        cv2.putText(
            frame,
            emotion_text,
            (x + 5, y - 30),
            self.font,
            self.font_scale,
            (255, 255, 255),
            self.thickness
        )
        
        # Draw confidence
        cv2.putText(
            frame,
            confidence_text,
            (x + 5, y - 10),
            self.font,
            self.font_scale * 0.8,
            (255, 255, 255),
            self.thickness - 1
        )
        
        return frame
    
    def draw_fps(self, frame, fps):
        """
        Draw FPS counter on frame.
        
        Args:
            frame: Input video frame
            fps: Current FPS value
        
        Returns:
            frame: Frame with FPS counter
        """
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (10, 30),
            self.font,
            self.font_scale + 0.3,
            (0, 255, 0),
            self.thickness
        )
        return frame
    
    def draw_statistics(self, frame, statistics, top_offset=60):
        """
        Draw emotion statistics on frame.
        
        Args:
            frame: Input video frame
            statistics: Dictionary with emotion statistics
            top_offset: Vertical offset from top
        
        Returns:
            frame: Frame with statistics
        """
        y_offset = top_offset
        
        # Draw title
        cv2.putText(
            frame,
            "Emotion Stats:",
            (10, y_offset),
            self.font,
            self.font_scale,
            (255, 255, 255),
            self.thickness
        )
        
        y_offset += 25
        
        # Sort emotions by count
        sorted_stats = sorted(
            [(e, s['count'], s['percentage']) for e, s in statistics.items() if e != 'total'],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Draw top 5 emotions
        for emotion, count, percentage in sorted_stats[:5]:
            text = f"{emotion.capitalize()}: {percentage:.1f}%"
            color = self.EMOTION_COLORS.get(emotion.lower(), (255, 255, 255))
            
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                self.font,
                self.font_scale * 0.8,
                color,
                self.thickness - 1
            )
            y_offset += 20
        
        return frame
    
    def draw_face_count(self, frame, face_count):
        """
        Draw number of detected faces.
        
        Args:
            frame: Input video frame
            face_count: Number of faces detected
        
        Returns:
            frame: Frame with face count
        """
        height = frame.shape[0]
        face_text = f"Faces: {face_count}"
        
        cv2.putText(
            frame,
            face_text,
            (10, height - 20),
            self.font,
            self.font_scale,
            (0, 255, 255),
            self.thickness
        )
        
        return frame
    
    def draw_multiple_faces(self, frame, faces_data):
        """
        Draw multiple faces with their emotions.
        
        Args:
            frame: Input video frame
            faces_data: List of tuples (x, y, w, h, emotion, confidence)
        
        Returns:
            frame: Frame with all faces annotated
        """
        for x, y, w, h, emotion, confidence in faces_data:
            frame = self.draw_face_box(frame, x, y, w, h, emotion, confidence)
        
        return frame
