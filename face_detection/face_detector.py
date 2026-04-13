"""
Face Detection Module
Detects faces in video frames using Haar Cascade or DNN based detection
"""
import cv2
import os


class FaceDetector:
    """Detects faces in images using multiple methods."""
    
    def __init__(self, method="haarcascade", min_face_size=20):
        """
        Initialize face detector.
        
        Args:
            method: Detection method - "haarcascade" or "dnn"
            min_face_size: Minimum face size in pixels
        """
        self.method = method
        self.min_face_size = min_face_size
        self.face_cascade = None
        self.net = None
        self.blob = None
        
        if method == "haarcascade":
            self._init_haarcascade()
        elif method == "dnn":
            self._init_dnn()
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def _init_haarcascade(self):
        """Initialize Haar Cascade classifier."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")
    
    def _init_dnn(self):
        """Initialize DNN based face detector."""
        # Paths to DNN model files
        model_file = cv2.data.haarcascades + '../dnn_face_detector_uint8.pb'
        config_file = cv2.data.haarcascades + '../opencv_face_detector.pbtxt'
        
        # Initialize network
        self.net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    def detect(self, frame, scale_factor=1.1, min_neighbors=5):
        """
        Detect faces in frame.
        
        Args:
            frame: Input video frame
            scale_factor: Scale factor for Haar Cascade
            min_neighbors: Minimum neighbors for Haar Cascade
        
        Returns:
            list: List of face bounding boxes (x, y, w, h)
        """
        if self.method == "haarcascade":
            return self._detect_haarcascade(frame, scale_factor, min_neighbors)
        elif self.method == "dnn":
            return self._detect_dnn(frame)
    
    def _detect_haarcascade(self, frame, scale_factor=1.1, min_neighbors=5):
        """
        Detect faces using Haar Cascade.
        
        Args:
            frame: Input video frame
            scale_factor: Scale factor for cascade
            min_neighbors: Minimum neighbors
        
        Returns:
            list: Detected faces as (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        return list(faces)
    
    def _detect_dnn(self, frame, confidence_threshold=0.5):
        """
        Detect faces using DNN.
        
        Args:
            frame: Input video frame
            confidence_threshold: Confidence threshold
        
        Returns:
            list: Detected faces as (x, y, w, h)
        """
        h, w = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            frame,
            1.0,
            (300, 300),
            [104, 117, 123],
            False,
            False
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Convert to (x, y, w, h) format
                x = max(0, x1)
                y = max(0, y1)
                face_w = min(x2, w) - x
                face_h = min(y2, h) - y
                
                if face_w > self.min_face_size and face_h > self.min_face_size:
                    faces.append((x, y, face_w, face_h))
        
        return faces
    
    def get_method(self):
        """Get current detection method."""
        return self.method
    
    def set_method(self, method):
        """
        Switch detection method.
        
        Args:
            method: New detection method
        """
        if method not in ["haarcascade", "dnn"]:
            raise ValueError(f"Unknown detection method: {method}")
        
        self.method = method
        if method == "haarcascade":
            self._init_haarcascade()
        else:
            self._init_dnn()
