# Emotion Detection System
> Last automated login update: 2026-04-14 12:41:38

A complete, production-ready emotion detection system using OpenCV and Deep Learning. The system detects faces in real-time, classifies emotions, and provides comprehensive analysis with logging and statistics.

## Features

Ã¢Å“â€¦ **Real-time Face Detection**
- Haar Cascade and DNN-based detection methods
- Multiple face support
- Configurable detection parameters

Ã¢Å“â€¦ **Emotion Classification**
- 7 emotion classes: Happy, Sad, Angry, Surprised, Neutral, Fear, Disgust
- CNN-based deep learning model
- Confidence scoring and smoothing
- GPU acceleration support

Ã¢Å“â€¦ **Data Collection & Training**
- Interactive dataset collection from webcam
- Automated model training with data augmentation
- Training history visualization
- Model evaluation and metrics

Ã¢Å“â€¦ **Real-time Visualization**
- Face bounding boxes with emotion labels
- Confidence scores displayed
- FPS counter
- Emotion statistics overlay
- Color-coded emotions for easy interpretation

Ã¢Å“â€¦ **Logging & Analysis**
- CSV logging of predictions
- Real-time emotion statistics
- Emotion frequency tracking
- Comprehensive reporting

Ã¢Å“â€¦ **Advanced Features**
- Emotion history tracking
- Confidence smoothing
- Multiple GPU acceleration options
- Easy configuration via YAML

## System Requirements

- Python 3.8 or higher
- Webcam (for real-time inference) or video file
- GPU support recommended for faster inference

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

Ensure all packages are installed:
```bash
pip list | grep -E "opencv|tensorflow|keras|numpy"
```

## Quick Start

### Option 1: Run with Pre-trained Model (If Available)

```bash
python main.py
```

### Option 2: Train a New Model

#### Step 1: Collect Training Data
```bash
python dataset/collect_data.py
```

**Instructions:**
- Press number 1-7 to capture images for each emotion
- Position your face in the center of the frame
- Collect at least 100 samples per emotion
- Press 's' to see collection progress
- Press 'q' to exit

#### Step 2: Train the Model
```bash
python training/train_model.py
```

This will:
- Load your collected dataset
- Build and train a CNN model
- Display training progress
- Save the trained model
- Generate training history plot

#### Step 3: Run Inference
```bash
python main.py
```

## Usage

### Main Application Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `s` | Show emotion statistics |
| `c` | Clear statistics |
| `f` | Toggle face detector (Haar Cascade Ã¢â€ â€ DNN) |
| `h` | Toggle emotion history tracking |

### Run with Video File

To run emotion detection on a video file instead of webcam:

```python
from main import EmotionDetectionSystem
import cv2

system = EmotionDetectionSystem()
system.run(source="path/to/video.mp4")
```

## Project Structure

```
emotion_detection_system/
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ main.py                          # Main application entry point
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ requirements.txt                 # Python dependencies
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ README.md                        # This file
Ã¢â€â€š
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ config/
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ config.yaml                  # Configuration file
Ã¢â€â€š
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ face_detection/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ __init__.py
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ face_detector.py             # Face detection module
Ã¢â€â€š
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ emotion_recognition/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ __init__.py
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ emotion_classifier.py        # Emotion classification module
Ã¢â€â€š
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ dataset/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ __init__.py
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ collect_data.py              # Data collection script
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ collected_images/            # Collected training images
Ã¢â€â€š       Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ angry/
Ã¢â€â€š       Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ disgust/
Ã¢â€â€š       Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ fear/
Ã¢â€â€š       Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ happy/
Ã¢â€â€š       Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ neutral/
Ã¢â€â€š       Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ sad/
Ã¢â€â€š       Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ surprised/
Ã¢â€â€š
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ training/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ __init__.py
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ train_model.py               # Model training script
Ã¢â€â€š
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ models/
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ emotion_model.h5             # Trained emotion model
Ã¢â€â€š
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ utils/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ __init__.py
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ fps.py                       # FPS counter
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ logger.py                    # Prediction logging
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ draw.py                      # Visualization utilities
Ã¢â€â€š
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ output/
    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ emotion_log.csv              # Emotion prediction log
    Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ training_history.png         # Training plots
```

## Configuration

Edit `config/config.yaml` to customize the system:

```yaml
# Camera Configuration
camera:
  index: 0                           # Webcam index
  resolution:
    width: 640
    height: 480

# Detection Configuration
detection:
  confidence_threshold: 0.6          # Minimum confidence for prediction
  min_face_size: 20                  # Minimum face size in pixels
  face_detector: "haarcascade"       # "haarcascade" or "dnn"

# Logging Configuration
logging:
  enabled: true
  output_file: "output/emotion_log.csv"
  log_interval: 1                    # Log every N frames

# Advanced Features
features:
  emotion_history_enabled: true
  history_window: 30                 # Number of frames to track
  confidence_smoothing_enabled: true
  smoothing_window: 5
  show_statistics: true
  gpu_acceleration: true
```

## Output Files

### emotion_log.csv
Records all emotion predictions with timestamps:
```
timestamp,emotion,confidence,num_faces,frame_number
2024-01-15 14:30:45.123,happy,0.9234,1,156
2024-01-15 14:30:45.156,happy,0.8912,1,157
```

### training_history.png
Visualizes training accuracy and loss curves.

## Model Architecture

The CNN model consists of:
- 4 convolutional blocks with batch normalization
- Max pooling after each block
- Dropout for regularization
- 3 dense layers with dropout
- Softmax output for 7 emotion classes

Total parameters: ~3.2M

## Supported Emotions

1. **Happy** Ã°Å¸ËœÅ  - Green bounding box
2. **Sad** Ã°Å¸ËœÂ¢ - Blue bounding box
3. **Angry** Ã°Å¸ËœÂ  - Red bounding box
4. **Surprised** Ã°Å¸ËœÂ² - Orange bounding box
5. **Neutral** Ã°Å¸ËœÂ - Gray bounding box
6. **Fear** Ã°Å¸ËœÂ¨ - Purple bounding box
7. **Disgust** Ã°Å¸Â¤Â¢ - Dark green bounding box

## Performance Optimization

### Enable GPU Acceleration
```yaml
features:
  gpu_acceleration: true
```

### Reduce Frame Processing
Increase `log_interval` to process fewer frames and improve performance.

### Adjust Confidence Threshold
Lower the threshold to detect more emotions, higher to be more selective.

## Troubleshooting

### Webcam Not Found
- Check camera index in `config.yaml` (usually 0 for built-in webcam)
- Ensure no other application is using the webcam
- Try different indices if you have multiple cameras

### Model Not Loading
- Run `python training/train_model.py` to create a trained model first
- Verify model path in `config.yaml`
- Check that `models/` directory exists

### Low Prediction Accuracy
- Collect more diverse training data (different lighting, angles, expressions)
- Increase training epochs in `config.yaml`
- Ensure proper face preprocessing

### Memory Issues
- Reduce batch size in `config.yaml`
- Disable GPU acceleration if causing issues
- Process smaller video resolution

### Slow Performance
- Switch to Haar Cascade detector (faster than DNN)
- Reduce frame resolution
- Disable visualization overlays temporarily

## Advanced Usage

### Custom Training with Transfer Learning

```python
from emotion_recognition.emotion_classifier import EmotionClassifier
from tensorflow.keras.applications import MobileNetV2

# Load pre-trained model
classifier = EmotionClassifier()
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False)
# Add custom layers...
```

### Batch Processing on Video Files

```python
import cv2
from face_detection.face_detector import FaceDetector
from emotion_recognition.emotion_classifier import EmotionClassifier

detector = FaceDetector()
classifier = EmotionClassifier()

cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret: break
    
    faces = detector.detect(frame)
    for x, y, w, h in faces:
        face_roi = frame[y:y+h, x:x+w]
        emotion, confidence = classifier.predict_emotion(face_roi)
        print(f"Emotion: {emotion}, Confidence: {confidence:.2%}")
```

### Export Statistics to Report

```python
from utils.logger import EmotionLogger

logger = EmotionLogger()
stats = logger.get_statistics()

# Generate report
report = f"Total Predictions: {stats['total']}\n"
for emotion, data in stats.items():
    if emotion != 'total':
        report += f"{emotion}: {data['percentage']:.1f}%\n"
```

## Expected Outputs

### Console Output
```
============================================================
Emotion Detection System
============================================================

Controls:
  'q' - Quit
  's' - Show statistics
  'c' - Clear statistics
  'f' - Toggle face detector method
  'h' - Toggle history tracking
============================================================

Starting real-time emotion detection...

Detected 1 face - Emotion: Happy, Confidence: 0.92
FPS: 28.5
```

### Window Display
- Real-time video with:
  - Face bounding boxes
  - Emotion labels
  - Confidence scores
  - FPS counter
  - Emotion statistics overlay

## References

- OpenCV: https://opencv.org/
- TensorFlow/Keras: https://tensorflow.org/
- Haar Cascades: https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html

## License

This project is provided as-is for educational and commercial use.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the configuration in `config/config.yaml`
3. Check output logs in `output/` directory
4. Verify model is trained before running inference

## Future Enhancements

- [ ] Multi-language support for UI
- [ ] Real-time model quantization for edge devices
- [ ] REST API for remote inference
- [ ] Cloud integration for model updates
- [ ] Mobile app support
- [ ] Advanced emotion sub-categories
- [ ] Emotion intensity measurement
- [ ] Multi-person emotion aggregate analysis

---

**Built with Ã¢ÂÂ¤Ã¯Â¸Â using Python, OpenCV, and TensorFlow**
