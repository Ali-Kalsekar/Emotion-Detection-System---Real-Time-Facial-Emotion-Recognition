# System Architecture & Developer Guide

## Overview

The Emotion Detection System is a modular, production-ready deep learning application built with Python, OpenCV, and TensorFlow/Keras. It provides real-time emotion recognition from facial images.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MAIN APPLICATION                         │
│                      (main.py)                               │
└──────────┬────────────────────────────────────────────────┬──┘
           │                                                  │
    ┌──────▼────────┐                              ┌─────────▼──┐
    │  Face         │                              │  Emotion   │
    │  Detection    │                              │  Classification
    │  Module       │                              │  Module    │
    │               │                              │            │
    │ • Detect      │                              │ • Load     │
    │   faces       │                              │   model    │
    │ • Return      │                              │ • Predict  │
    │   bboxes      │                              │   emotion  │
    │ • Support     │                              │ • Score    │
    │   multiple    │                              │   confidence
    │   faces       │                              │            │
    └───────────────┘                              └────────────┘
           │                                              │
           │                    ┌──────────────────────┐  │
           │                    │  Model Files         │  │
           │                    │ (models/emotion_     │  │
           │                    │  model.h5)           │  │
           │                    └──────────────────────┘  │
           │                                              │
    ┌──────▼──────────────────────────────────────────────▼───┐
    │            VISUALIZATION & LOGGING                      │
    │                                                         │
    │  ┌────────────────┐  ┌──────────┐  ┌──────────────┐  │
    │  │  Draw Module   │  │  Logger  │  │  FPS Counter │  │
    │  │                │  │          │  │              │  │
    │  │ • Bounding     │  │ • CSV    │  │ • Track FPS  │  │
    │  │   boxes        │  │   logging│  │ • Real-time  │  │
    │  │ • Emotions     │  │ • Stats  │  │   stats      │  │
    │  │ • Confidence   │  │          │  │              │  │
    │  └────────────────┘  └──────────┘  └──────────────┘  │
    │                                                         │
    └──────────┬──────────────────────────────────────────────┘
               │
      ┌────────▼────────┐
      │  Output Files   │
      │                 │
      │ • Logs (CSV)    │
      │ • Config (YAML) │
      │ • Plots (PNG)   │
      └─────────────────┘
```

## Module Descriptions

### 1. Face Detection Module (`face_detection/face_detector.py`)

**Purpose:** Detect faces in video frames

**Key Class:** `FaceDetector`

**Methods:**
- `__init__(method, min_face_size)` - Initialize detector
- `detect(frame, scale_factor, min_neighbors)` - Detect faces
- `get_method()` / `set_method(method)` - Switch detection method

**Detection Methods:**
- **Haar Cascade**: Fast, built-in OpenCV method
- **DNN**: More accurate but slower

**Example:**
```python
from face_detection.face_detector import FaceDetector

detector = FaceDetector(method="haarcascade")
faces = detector.detect(frame)  # Returns [(x, y, w, h), ...]
```

### 2. Emotion Classification Module (`emotion_recognition/emotion_classifier.py`)

**Purpose:** Classify emotions from face images

**Key Class:** `EmotionClassifier`

**Methods:**
- `__init__(model_path, input_size)` - Initialize classifier
- `load_model()` - Load trained model
- `preprocess_face(face_image)` - Prepare image for model
- `predict_emotion(face_image, use_smoothing)` - Get emotion prediction
- `predict_batch(face_images)` - Process multiple faces
- `save_model(save_path)` - Save trained model

**Supported Emotions:**
- angry, disgust, fear, happy, neutral, sad, surprised

**Example:**
```python
from emotion_recognition.emotion_classifier import EmotionClassifier

classifier = EmotionClassifier(model_path="models/emotion_model.h5")
emotion, confidence = classifier.predict_emotion(face_image)
```

### 3. Data Collection Module (`dataset/collect_data.py`)

**Purpose:** Collect training dataset from webcam

**Key Class:** `DataCollector`

**Features:**
- Interactive GUI for data collection
- Organize images by emotion
- Real-time progress tracking
- Statistical reporting

**Usage:**
```bash
python dataset/collect_data.py
```

### 4. Training Module (`training/train_model.py`)

**Purpose:** Train CNN model for emotion recognition

**Key Class:** `EmotionModelTrainer`

**Methods:**
- `load_dataset(dataset_path)` - Load images
- `build_model()` - Build CNN architecture
- `train(X_data, y_data)` - Train model
- `evaluate(X_test, y_test)` - Test model
- `save_model()` - Save trained model
- `plot_training_history(save_path)` - Visualize training

**Model Architecture:**
- 4 convolutional blocks (32→64→128→256 filters)
- Batch normalization after each conv layer
- Max pooling (2×2) after each block
- Dropout (0.25-0.5) for regularization
- 3 dense layers (512→256→7)
- Softmax output for 7 emotions

**Usage:**
```bash
python training/train_model.py
```

### 5. Main Application (`main.py`)

**Purpose:** Real-time emotion detection from webcam

**Key Class:** `EmotionDetectionSystem`

**Main Functions:**
- `__init__(config_path)` - Initialize system
- `run(source)` - Start real-time detection

**Features:**
- Real-time face and emotion detection
- Multiple face support
- FPS tracking
- Emotion statistics
- Logging predictions
- Configurable via YAML

**Usage:**
```bash
python main.py
```

### 6. Visualization Module (`utils/draw.py`)

**Purpose:** Draw annotations on video frames

**Key Class:** `Visualizer`

**Methods:**
- `draw_face_box(frame, x, y, w, h, emotion, confidence)`
- `draw_fps(frame, fps)`
- `draw_statistics(frame, statistics, top_offset)`
- `draw_face_count(frame, face_count)`
- `draw_multiple_faces(frame, faces_data)`

**Color Scheme:**
- Happy: Green
- Sad: Blue
- Angry: Red
- Surprised: Orange
- Neutral: Gray
- Fear: Purple
- Disgust: Dark Green

### 7. FPS Counter Module (`utils/fps.py`)

**Purpose:** Track real-time performance

**Key Class:** `FPSCounter`

**Methods:**
- `update()` - Update with current frame
- `get_fps()` - Get current FPS
- `get_average_fps()` - Get averaged FPS
- `reset()` - Reset counter

### 8. Logger Module (`utils/logger.py`)

**Purpose:** Log predictions and generate statistics

**Key Class:** `EmotionLogger`

**Methods:**
- `log_prediction(emotion, confidence, num_faces, frame_number)`
- `get_statistics()` - Get statistics dictionary
- `print_statistics()` - Print formatted stats
- `reset()` - Clear statistics

**Output Format (CSV):**
```
timestamp,emotion,confidence,num_faces,frame_number
2024-01-15 14:30:45.123,happy,0.9234,1,156
```

## Data Flow

### Training Flow
```
Raw Images
    ↓
Data Augmentation (rotation, zoom, shift)
    ↓
Model Training
    ├── Forward Pass
    ├── Loss Calculation
    └── Backpropagation
    ↓
Model Validation
    ├── Accuracy Calculation
    └── Early Stopping
    ↓
Saved Model (emotion_model.h5)
    ↓
Training Plots
```

### Inference Flow
```
Video Frame
    ↓
Face Detection
    ├── Convert to Gray
    ├── Multi-scale Scanning
    └── Bounding Boxes
    ↓
For Each Face:
    ├── Extract ROI
    ├── Preprocess (resize, normalize)
    ├── Model Inference
    ├── Softmax Output
    └── Get Max Probability
    ↓
Visualization
    ├── Draw Boxes
    ├── Add Labels
    ├── Show Confidence
    └── Display Stats
    ↓
Logging
    ├── Record Timestamp
    ├── Save Prediction
    └── Update Statistics
    ↓
Display Frame
```

## Configuration System

The system is configured via `config/config.yaml`:

```yaml
camera:
  index: 0              # Webcam index
  resolution:
    width: 640
    height: 480

model:
  path: "models/emotion_model.h5"
  input_size: 224

detection:
  confidence_threshold: 0.6
  face_detector: "haarcascade"  # or "dnn"

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001

features:
  emotion_history_enabled: true
  confidence_smoothing_enabled: true
  gpu_acceleration: true
```

## CNN Model Architecture

```
Input: (224, 224, 3) RGB Image
    ↓
Block 1: Conv(32) → BN → Conv(32) → BN → MaxPool(2) → Dropout(0.25)
    ↓ (112×112×32)
Block 2: Conv(64) → BN → Conv(64) → BN → MaxPool(2) → Dropout(0.25)
    ↓ (56×56×64)
Block 3: Conv(128) → BN → Conv(128) → BN → MaxPool(2) → Dropout(0.25)
    ↓ (28×28×128)
Block 4: Conv(256) → BN → Conv(256) → BN → MaxPool(2) → Dropout(0.25)
    ↓ (14×14×256)
Flatten
    ↓
Dense(512) → BN → Dropout(0.5)
    ↓
Dense(256) → BN → Dropout(0.5)
    ↓
Dense(7) → Softmax
    ↓
Output: [angry, disgust, fear, happy, neutral, sad, surprised] probabilities
```

## Development Guidelines

### Adding a New Emotion Class

1. **Update `config.yaml`:**
```yaml
emotions:
  - "angry"
  - "new_emotion"  # Add here
```

2. **Retrain Model:**
```bash
python training/train_model.py
```

3. **Update Color Scheme:**
```python
# In utils/draw.py
EMOTION_COLORS = {
    'new_emotion': (100, 150, 200),  # Add color
}
```

### Custom Face Detection Algorithm

```python
from face_detection.face_detector import FaceDetector

class CustomDetector(FaceDetector):
    def detect(self, frame, **kwargs):
        # Your custom detection logic
        return faces
```

### Custom Preprocessing

```python
def custom_preprocess(face_image):
    # Your preprocessing pipeline
    resized = cv2.resize(face_image, (224, 224))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)
```

## Performance Optimization

### 1. GPU Acceleration
```python
# TensorFlow automatically uses GPU if available
# Configure in config.yaml:
features:
  gpu_acceleration: true
```

### 2. Batch Processing
```python
classifier = EmotionClassifier()
emotions = classifier.predict_batch(face_images)
```

### 3. Model Quantization
```python
# For 5-10x speedup on edge devices
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
```

### 4. Multi-threading
```python
from threading import Thread

def detection_thread(frame_queue, result_queue):
    detector = FaceDetector()
    while True:
        frame = frame_queue.get()
        faces = detector.detect(frame)
        result_queue.put(faces)
```

## Testing

Run comprehensive tests:
```bash
python test_system.py
```

Tests coverage:
- ✓ Import verification
- ✓ Project structure
- ✓ Configuration
- ✓ Face detector
- ✓ Emotion classifier
- ✓ Utilities
- ✓ System requirements

## Deployment Considerations

### Edge Devices (Raspberry Pi, Jetson)
1. Use TensorFlow Lite for quantization
2. Switch to Haar Cascade detector
3. Reduce image resolution
4. Use smaller model (MobileNet base)

### Web Service
1. Create REST API endpoints
2. Implement request queuing
3. Add authentication/authorization
4. Monitor performance metrics

### Cloud Deployment
1. Containerize with Docker
2. Deploy on Kubernetes
3. Use managed ML services
4. Implement auto-scaling

## Troubleshooting

### Low Detection Rate
- Ensure good lighting conditions
- Adjust `detection.confidence_threshold`
- Retrain with more diverse data

### Slow Performance
- Use Haar Cascade detector
- Reduce input resolution
- Enable GPU acceleration
- Process in batches

### Memory Issues
- Reduce batch size
- Clear model cache periodically
- Use model pruning/quantization

## Future Enhancements

- [ ] Multi-GPU support
- [ ] Real-time model updates
- [ ] Emotion intensity classification
- [ ] Cross-frame temporal analysis
- [ ] REST API for remote inference
- [ ] Mobile app integration
- [ ] Advanced data augmentation
- [ ] Model ensemble methods

---

**Last Updated:** 2024
**Version:** 1.0
