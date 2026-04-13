"""
Advanced Features & Examples
Demonstrations of advanced features and customizations
"""

# ============================================================
# Example 1: Custom Training with Save/Load
# ============================================================
def example_train_custom_model():
    """Train model with custom parameters."""
    from training.train_model import EmotionModelTrainer
    import yaml
    
    trainer = EmotionModelTrainer(config_path="config/config.yaml")
    
    # Load your dataset
    X_data, y_data = trainer.load_dataset(dataset_path="dataset/collected_images")
    
    # Build model
    trainer.build_model()
    
    # Train with custom epochs
    trainer.train(X_data, y_data)
    
    # Evaluate
    trainer.evaluate(X_data, y_data)
    
    # Save model with custom name
    trainer.save_model(save_path="models/custom_emotion_model.h5")
    
    # Plot and save training history
    trainer.plot_training_history(save_path="output/custom_training_history.png")


# ============================================================
# Example 2: Batch Video Processing
# ============================================================
def example_batch_video_processing():
    """Process multiple videos and generate reports."""
    import cv2
    import json
    from face_detection.face_detector import FaceDetector
    from emotion_recognition.emotion_classifier import EmotionClassifier
    from utils.logger import EmotionLogger
    
    detector = FaceDetector(method="haarcascade")
    classifier = EmotionClassifier()
    logger = EmotionLogger(log_file="output/batch_processing_log.csv")
    
    video_files = [
        "path/to/video1.mp4",
        "path/to/video2.mp4",
    ]
    
    results = {}
    
    for video_file in video_files:
        print(f"\nProcessing {video_file}...")
        cap = cv2.VideoCapture(video_file)
        frame_count = 0
        emotion_counts = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = detector.detect(frame)
            
            # Process each face
            for x, y, w, h in faces:
                face_roi = frame[y:y+h, x:x+w]
                emotion, confidence = classifier.predict_emotion(face_roi)
                
                # Update statistics
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                logger.log_prediction(emotion, confidence, len(faces), frame_count)
            
            frame_count += 1
        
        cap.release()
        results[video_file] = {
            'frames': frame_count,
            'emotions': emotion_counts
        }
        
        print(f"  Processed {frame_count} frames")
        print(f"  Emotions detected: {emotion_counts}")
    
    # Save batch results
    with open("output/batch_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nBatch processing complete! Results saved to output/batch_results.json")


# ============================================================
# Example 3: Real-time with Custom Callbacks
# ============================================================
def example_realtime_with_callbacks():
    """Run inference with custom event callbacks."""
    import cv2
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from face_detection.face_detector import FaceDetector
    from emotion_recognition.emotion_classifier import EmotionClassifier
    from utils.fps import FPSCounter
    from utils.draw import Visualizer
    
    class CustomCallbacks:
        def on_emotion_detected(self, emotion, confidence):
            print(f"[CALLBACK] Detected: {emotion} (confidence: {confidence:.2%})")
        
        def on_multiple_faces(self, count):
            if count > 1:
                print(f"[CALLBACK] {count} faces detected!")
        
        def on_low_confidence(self, emotion, confidence):
            if confidence < 0.5:
                print(f"[CALLBACK] Low confidence for {emotion}: {confidence:.2%}")
    
    detector = FaceDetector()
    classifier = EmotionClassifier()
    visualizer = Visualizer()
    fps_counter = FPSCounter()
    callbacks = CustomCallbacks()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        faces = detector.detect(frame)
        
        callbacks.on_multiple_faces(len(faces))
        
        for x, y, w, h in faces:
            face_roi = frame[y:y+h, x:x+w]
            emotion, confidence = classifier.predict_emotion(face_roi)
            
            callbacks.on_emotion_detected(emotion, confidence)
            callbacks.on_low_confidence(emotion, confidence)
            
            frame = visualizer.draw_face_box(frame, x, y, w, h, emotion, confidence)
        
        fps_counter.update()
        fps = fps_counter.get_fps()
        frame = visualizer.draw_fps(frame, fps)
        
        cv2.imshow("Custom Callbacks", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Example 4: Model Comparison
# ============================================================
def example_compare_detectors():
    """Compare face detection methods."""
    import cv2
    import time
    import numpy as np
    from face_detection.face_detector import FaceDetector
    
    # Load test image
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not capture frame")
        return
    
    methods = ["haarcascade", "dnn"]
    
    print("\n" + "="*60)
    print("DETECTOR COMPARISON")
    print("="*60 + "\n")
    
    for method in methods:
        try:
            detector = FaceDetector(method=method)
            
            # Warm up
            detector.detect(frame)
            
            # Benchmark
            start = time.time()
            for _ in range(10):
                faces = detector.detect(frame)
            elapsed = time.time() - start
            
            avg_time = elapsed / 10 * 1000  # ms
            fps = 1000 / avg_time
            
            print(f"{method.upper()}:")
            print(f"  Faces detected: {len(faces)}")
            print(f"  Average time: {avg_time:.2f} ms")
            print(f"  FPS: {fps:.1f}\n")
        
        except Exception as e:
            print(f"{method.upper()}: Error - {e}\n")


# ============================================================
# Example 5: Export Statistics Report
# ============================================================
def example_generate_report():
    """Generate comprehensive statistics report."""
    from utils.logger import EmotionLogger
    import json
    from datetime import datetime
    
    logger = EmotionLogger()
    # Assuming predictions have been made and logged
    stats = logger.get_statistics()
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_predictions': stats['total'],
        'emotion_breakdown': {}
    }
    
    for emotion in logger.emotion_counts:
        report['emotion_breakdown'][emotion] = {
            'count': stats[emotion]['count'],
            'percentage': stats[emotion]['percentage']
        }
    
    # Save report
    with open("output/emotion_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print("Report saved to output/emotion_report.json")
    print(json.dumps(report, indent=2))


# ============================================================
# Example 6: Real-time Emotion History Analysis
# ============================================================
def example_emotion_trends():
    """Analyze emotion trends over time."""
    import pandas as pd
    from utils.logger import EmotionLogger
    
    logger = EmotionLogger()
    
    # Read log file
    df = pd.read_csv(logger.log_file)
    
    # Group by emotion
    emotion_stats = df.groupby('emotion').size()
    
    # Get most common emotion
    most_common = emotion_stats.idxmax()
    
    print(f"\nMost common emotion: {most_common}")
    print(f"\nEmotion distribution:")
    print(emotion_stats)
    
    # Time-based analysis
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"\nFirst detection: {df['timestamp'].min()}")
    print(f"Last detection: {df['timestamp'].max()}")
    print(f"Duration: {df['timestamp'].max() - df['timestamp'].min()}")


# ============================================================
# Main Demo
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADVANCED FEATURES EXAMPLES")
    print("="*60)
    
    print("\nAvailable examples:")
    print("1. Train custom model")
    print("2. Batch video processing")
    print("3. Real-time with callbacks")
    print("4. Compare detectors")
    print("5. Generate report")
    print("6. Emotion trends analysis")
    
    choice = input("\nEnter example number (1-6): ").strip()
    
    try:
        if choice == "1":
            example_train_custom_model()
        elif choice == "2":
            example_batch_video_processing()
        elif choice == "3":
            example_realtime_with_callbacks()
        elif choice == "4":
            example_compare_detectors()
        elif choice == "5":
            example_generate_report()
        elif choice == "6":
            example_emotion_trends()
        else:
            print("Invalid choice")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60 + "\n")
