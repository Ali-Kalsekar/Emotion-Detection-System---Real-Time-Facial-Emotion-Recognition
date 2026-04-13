# Deployment Guide

Guide for deploying the Emotion Detection System in production environments.

## Pre-Deployment Checklist

- [ ] Test system locally: `python test_system.py`
- [ ] Train model: `python training/train_model.py`
- [ ] Test inference: `python main.py`
- [ ] Review configuration in `config/config.yaml`
- [ ] Verify logs directory permissions
- [ ] Set up monitoring/alerting systems

## Local Deployment

### 1. System Requirements
- Python 3.8+
- 2GB+ RAM (4GB+ recommended)
- GPU support optional but recommended

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python test_system.py
```

### 4. Training
```bash
python training/train_model.py
```

### 5. Production Run
```bash
python main.py
```

## Docker Deployment

### 1. Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
```

### 2. Build Image
```bash
docker build -t emotion-detection:latest .
```

### 3. Run Container
```bash
docker run -v /dev/video0:/dev/video0 \
           -v $(pwd)/output:/app/output \
           emotion-detection:latest
```

### Docker Compose for Multiple Services

```yaml
version: '3'

services:
  emotion_detector:
    build: .
    volumes:
      - /dev/video0:/dev/video0
      - ./output:/app/output
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
    restart: unless-stopped
```

## Kubernetes Deployment

### 1. Create Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: emotion-detector
spec:
  replicas: 1
  selector:
    matchLabels:
      app: emotion-detector
  template:
    metadata:
      labels:
        app: emotion-detector
    spec:
      containers:
      - name: detector
        image: emotion-detection:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: output
          mountPath: /app/output
        - name: models
          mountPath: /app/models
      volumes:
      - name: output
        emptyDir: {}
      - name: models
        configMap:
          name: emotion-models
```

### 2. Deploy to Kubernetes
```bash
kubectl apply -f deployment.yaml
```

## Edge Device Deployment (Raspberry Pi/Jetson)

### Raspberry Pi

```bash
# Install dependencies
pip install --no-cache-dir -r requirements_rpi.txt

# Run with reduced resources
python main.py --config config/config_rpi.yaml
```

**config_rpi.yaml:**
```yaml
camera:
  resolution:
    width: 320
    height: 240

detection:
  face_detector: "haarcascade"  # Faster

model:
  input_size: 128
```

### NVIDIA Jetson

```bash
# Enable GPU
pip install torch torchvision torchaudio
pip install --no-cache-dir -r requirements.txt

# Run with GPU
python main.py --gpu
```

## Cloud Deployment

### AWS Deployment

#### 1. Create EC2 Instance
```bash
# g3.4xlarge (GPU instance recommended)
# Ubuntu 20.04 LTS
```

#### 2. Install CUDA/cuDNN
```bash
chmod +x install_cuda.sh
./install_cuda.sh
```

#### 3. Deploy Application
```bash
git clone <repo>
cd emotion_detection_system
pip install -r requirements.txt
python training/train_model.py  # If first time
python main.py
```

#### 4. Create AMI for Scaling
```bash
# Create custom AMI from configured instance
# Use Auto Scaling Group for multiple instances
```

### GCP Deployment

#### 1. Create Compute Instance
```bash
gcloud compute instances create emotion-detector \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --accelerator=type=nvidia-tesla-k80,count=1 \
  --create-disk=auto-delete=yes,boot=yes,size=50GB
```

#### 2. SSH and Install
```bash
gcloud compute ssh emotion-detector
# Then follow standard installation steps
```

### Azure Deployment

```bash
# Create resource group
az group create --name emotion-rg --location eastus

# Create container instance
az container create \
  --resource-group emotion-rg \
  --name emotion-detector \
  --image emotion-detection:latest \
  --gpu 1
```

## REST API Deployment

### Using Flask

```python
# api.py
from flask import Flask, request, jsonify
from main import EmotionDetectionSystem
import cv2
import numpy as np
import base64

app = Flask(__name__)
system = EmotionDetectionSystem()

@app.route('/predict', methods=['POST'])
def predict():
    """Predict emotion from image"""
    data = request.json
    image_data = base64.b64decode(data['image'])
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    
    faces = system.face_detector.detect(image)
    results = []
    
    for x, y, w, h in faces:
        roi = image[y:y+h, x:x+w]
        emotion, confidence = system.emotion_classifier.predict_emotion(roi)
        results.append({
            'emotion': emotion,
            'confidence': float(confidence),
            'bbox': [x, y, w, h]
        })
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Deploy with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api:app
```

## Monitoring & Logging

### Application Monitoring

```python
# monitoring.py
import psutil
import time
from datetime import datetime

def log_system_metrics():
    """Log system metrics"""
    timestamp = datetime.now().isoformat()
    
    metrics = {
        'timestamp': timestamp,
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'gpu_utilization': get_gpu_util(),  # Custom function
    }
    
    # Send to monitoring system
    send_to_monitoring(metrics)

def send_to_monitoring(metrics):
    """Send metrics to monitoring backend"""
    # Integrate with your monitoring system
    # (Prometheus, DataDog, CloudWatch, etc.)
    pass
```

### Health Checks

```python
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': system.emotion_classifier.model is not None,
        'detector_ready': system.face_detector is not None
    })
```

## Performance Optimization

### Model Quantization (TensorFlow Lite)

```python
# quantize_model.py
import tensorflow as tf

def quantize_model(model_path, output_path):
    """Quantize model for faster inference"""
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Quantized model saved to {output_path}")

# Usage
quantize_model('models/emotion_model.h5', 'models/emotion_model.tflite')
```

### Batch Processing

```python
def process_batch_video(video_path, batch_size=32):
    """Process video in batches for efficiency"""
    cap = cv2.VideoCapture(video_path)
    batch = []
    results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        batch.append(frame)
        
        if len(batch) == batch_size:
            # Process batch
            predictions = system.emotion_classifier.predict_batch(batch)
            results.extend(predictions)
            batch = []
    
    # Process remaining frames
    if batch:
        predictions = system.emotion_classifier.predict_batch(batch)
        results.extend(predictions)
    
    cap.release()
    return results
```

## Security Considerations

### Authentication

```python
from functools import wraps
from flask import request

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not validate_api_key(api_key):
            return {'error': 'Invalid API key'}, 401
        return f(*args, **kwargs)
    return decorated

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # Your code
    pass
```

### Data Privacy

- Encrypt stored predictions
- Use HTTPS for API endpoints
- Implement access controls
- Log audit trails
- Follow GDPR/HIPAA compliance

## Backup & Recovery

### Model Backup Strategy

```bash
# Backup model
tar -czf emotion_model_backup_$(date +%Y%m%d_%H%M%S).tar.gz models/

# Backup configuration
cp config/config.yaml config/config_backup_$(date +%Y%m%d_%H%M%S).yaml

# Backup logs
cp output/emotion_log.csv output/emotion_log_backup_$(date +%Y%m%d_%H%M%S).csv
```

### Disaster Recovery Plan

1. **Identify Critical Components**
   - Trained model
   - Configuration files
   - Log data

2. **Implement Redundancy**
   - Multi-region replication
   - Database backups
   - Model versioning

3. **Test Recovery Process**
   - Regular backups
   - Restore validation
   - Failover testing

## Scaling Strategies

### Horizontal Scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: emotion-detector-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: emotion-detector
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Vertical Scaling

Update resources in deployment YAML:
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

## Troubleshooting Deployment

### Issue: Memory Exhaustion
```bash
# Check memory usage
ps aux | grep python

# Reduce model size
python quantize_model.py
```

### Issue: High Latency
```bash
# Profile performance
python -m cProfile -s cumtime main.py

# Enable batch processing
batch_size = 32
```

### Issue: GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi

# Verify TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Maintenance Schedule

### Daily
- Monitor error logs
- Check system health
- Verify model predictions

### Weekly
- Review performance metrics
- Backup data
- Update dependencies

### Monthly
- Retrain model with new data
- Performance optimization
- Security updates

### Quarterly
- Disaster recovery testing
- Model evaluation
- Infrastructure review

---

**For more information, see README.md and ARCHITECTURE.md**
