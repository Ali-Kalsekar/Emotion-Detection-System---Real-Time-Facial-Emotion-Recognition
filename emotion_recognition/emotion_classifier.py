"""
Emotion Classification Module
Classifies emotions from face images using trained CNN model
"""
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from collections import deque


class EmotionClassifier:
    """Classifies emotions from facial images using CNN."""
    
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    
    def __init__(self, model_path="models/emotion_model.h5", input_size=224):
        """
        Initialize emotion classifier.
        
        Args:
            model_path: Path to trained emotion model
            input_size: Input image size for model
        """
        self.model_path = model_path
        self.input_size = input_size
        self.model = None
        self.confidence_history = {emotion: deque(maxlen=5) for emotion in self.EMOTIONS}
        
        # Try to load existing model
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"Model not found at {model_path}")
            print("Please train the model first using: python training/train_model.py")
            # Initialize a placeholder model
            self._init_placeholder_model()
    
    def _init_placeholder_model(self):
        """Initialize a placeholder model for demo purposes."""
        # This model will output random predictions
        # It's used when no trained model is available
        self.model = None
    
    def load_model(self):
        """Load trained emotion model from disk."""
        try:
            self.model = load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def save_model(self, save_path=None):
        """
        Save model to disk.
        
        Args:
            save_path: Path to save model (defaults to model_path)
        """
        if self.model is None:
            print("No model to save")
            return
        
        save_path = save_path or self.model_path
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        
        try:
            self.model.save(save_path)
            print(f"Model saved successfully to {save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def preprocess_face(self, face_image):
        """
        Preprocess face image for model input.
        
        Args:
            face_image: Raw face image (BGR grayscale or color)
        
        Returns:
            np.ndarray: Preprocessed image ready for model
        """
        try:
            # Convert to RGB if grayscale
            if len(face_image.shape) == 2:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
            elif face_image.shape[2] == 4:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGRA2BGR)
            
            # Resize to model input size
            resized = cv2.resize(face_image, (self.input_size, self.input_size))
            
            # Normalize to [0, 1]
            normalized = resized.astype('float32') / 255.0
            
            # Add batch dimension
            input_data = np.expand_dims(normalized, axis=0)
            
            return input_data
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None
    
    def predict_emotion(self, face_image, use_smoothing=True):
        """
        Predict emotion from face image.
        
        Args:
            face_image: Face image (BGR or grayscale)
            use_smoothing: Apply confidence smoothing over frames
        
        Returns:
            tuple: (emotion_label, confidence_score)
        """
        if self.model is None:
            # Return random prediction if model not loaded
            idx = np.random.randint(0, len(self.EMOTIONS))
            return self.EMOTIONS[idx], np.random.rand()
        
        try:
            # Preprocess image
            input_data = self.preprocess_face(face_image)
            if input_data is None:
                return "neutral", 0.0
            
            # Predict
            predictions = self.model.predict(input_data, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            emotion = self.EMOTIONS[emotion_idx]
            confidence = float(predictions[0][emotion_idx])
            
            # Apply smoothing if enabled
            if use_smoothing:
                self.confidence_history[emotion].append(confidence)
                confidence = np.mean(self.confidence_history[emotion])
            
            return emotion, confidence
        
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return "neutral", 0.0
    
    def predict_batch(self, face_images):
        """
        Predict emotions for multiple faces.
        
        Args:
            face_images: List of face images
        
        Returns:
            list: List of (emotion, confidence) tuples
        """
        results = []
        for face_image in face_images:
            emotion, confidence = self.predict_emotion(face_image, use_smoothing=False)
            results.append((emotion, confidence))
        
        return results
    
    def get_emotions(self):
        """Get list of supported emotions."""
        return self.EMOTIONS.copy()
    
    def reset_history(self):
        """Reset confidence history."""
        for emotion in self.EMOTIONS:
            self.confidence_history[emotion] = deque(maxlen=5)
    
    def get_model_summary(self):
        """Get model summary."""
        if self.model is None:
            print("No model loaded")
            return
        
        return self.model.summary()
