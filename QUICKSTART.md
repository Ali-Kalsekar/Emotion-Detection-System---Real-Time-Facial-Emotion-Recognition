# Quick Start Guide

## Installation & Setup

### 1. Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Webcam (optional, can use video files)

### 2. Install Dependencies
```bash
cd emotion_detection_system
pip install -r requirements.txt
```

Wait for installation to complete. This may take a few minutes.

### 3. Verify Installation
```bash
python -c "import cv2, tensorflow, keras; print('All dependencies installed successfully!')"
```

## Usage Guide

### First Time Setup

#### Step 1: Collect Training Data
```bash
python dataset/collect_data.py
```

**What to do:**
- Position your face in the webcam
- Press number keys 1-7 to collect samples for each emotion:
  - 1 = Angry
  - 2 = Disgust
  - 3 = Fear
  - 4 = Happy
  - 5 = Neutral
  - 6 = Sad
  - 7 = Surprised
- Collect at least 50-100 samples per emotion
- Press 's' to see progress
- Press 'q' when done

**Tip:** Vary your facial expressions and angles for better model performance.

#### Step 2: Train the Model
```bash
python training/train_model.py
```

**Expected output:**
- Loading dataset...
- Building model...
- Training progress with epochs
- Training history plot saved to `output/training_history.png`

This process may take 10-30 minutes depending on your hardware.

#### Step 3: Run Real-time Emotion Detection
```bash
python main.py
```

**What to expect:**
- Webcam window opens with your face
- Green bounding box around detected faces
- Emotion label and confidence score displayed
- FPS counter in top-left
- Statistics panel on the left side

**Controls:**
- Press 'q' to quit
- Press 's' to show emotion statistics
- Press 'c' to clear statistics

### Running Subsequent Times

Once the model is trained, simply run:
```bash
python main.py
```

## File Structure After Setup

```
emotion_detection_system/
├── main.py
├── requirements.txt
├── config/
│   └── config.yaml
├── face_detection/
├── emotion_recognition/
├── dataset/
│   └── collected_images/          # Your training images
│       ├── angry/
│       ├── disgust/
│       ├── fear/
│       ├── happy/
│       ├── neutral/
│       ├── sad/
│       └── surprised/
├── training/
├── models/
│   └── emotion_model.h5           # Trained model (created after training)
├── utils/
└── output/
    ├── emotion_log.csv            # Prediction logs
    └── training_history.png       # Training plots
```

## Common Issues & Solutions

### Issue: ImportError for opencv, tensorflow, etc.
**Solution:** Make sure you installed requirements:
```bash
pip install -r requirements.txt
```

### Issue: Webcam not detected
**Solution:** 
- Check `config/config.yaml` and change `camera.index` (try 0, 1, 2)
- Or run from terminal with `-device` flag
- Make sure no other app is using webcam

### Issue: Model not found error
**Solution:** Train the model first:
```bash
python training/train_model.py
```

### Issue: Low accuracy
**Solution:**
- Collect more training data (200+ samples per emotion)
- Vary lighting conditions and angles
- Retrain the model with more epochs (change in `config.yaml`)

### Issue: Slow performance
**Solution:**
- Use Haar Cascade detector: Change `detection.face_detector` to "haarcascade" in `config.yaml`
- Lower camera resolution: Set lower `camera.resolution` values
- Disable statistics: Set `features.show_statistics` to false

## Configuration Customization

Edit `config/config.yaml` to customize:

```yaml
# Webcam settings
camera:
  index: 0                    # Which camera to use
  resolution:
    width: 640
    height: 480

# Detection sensitivity
detection:
  confidence_threshold: 0.6  # Higher = more confident predictions required
  face_detector: "haarcascade"  # Or "dnn" for more accuracy

# Display options
display:
  show_fps: true
  show_confidence: true
```

## Next Steps

After successfully running the system:

1. **Collect More Data** - The more diverse your training data, the better the model
2. **Fine-tune** - Adjust confidence threshold and detection parameters in `config.yaml`
3. **Export Results** - Check `output/emotion_log.csv` for prediction records
4. **Experiment** - Try different face detectors and model configurations

## Performance Tips

- GPU support: If you have NVIDIA GPU, install GPU version of TensorFlow for 5-10x speedup
- Model caching: The model loads once and runs inference very quickly
- Batch processing: Process multiple faces in parallel (see advanced usage in README.md)

## Getting Help

1. Check the main `README.md` for detailed documentation
2. Review `config.yaml` comments for configuration options
3. Check the troubleshooting section in `README.md`

Enjoy emotion detection! 🎉
