"""
Test & Verification Script
Tests all components of the emotion detection system
"""
import sys
import os


def test_imports():
    """Test if all required libraries can be imported."""
    print("\n" + "="*60)
    print("TESTING IMPORTS")
    print("="*60)
    
    libraries = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'tensorflow': 'TensorFlow',
        'keras': 'Keras',
        'yaml': 'PyYAML',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'PIL': 'Pillow'
    }
    
    all_passed = True
    
    for lib, name in libraries.items():
        try:
            __import__(lib)
            print(f"✓ {name:20} - OK")
        except ImportError as e:
            print(f"✗ {name:20} - FAILED: {e}")
            all_passed = False
    
    return all_passed


def test_project_structure():
    """Test if project structure is correct."""
    print("\n" + "="*60)
    print("TESTING PROJECT STRUCTURE")
    print("="*60)
    
    required_dirs = [
        'config',
        'dataset',
        'emotion_recognition',
        'face_detection',
        'models',
        'output',
        'training',
        'utils'
    ]
    
    required_files = [
        'config/config.yaml',
        'main.py',
        'requirements.txt'
    ]
    
    all_passed = True
    
    print("\nDirectories:")
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"✓ {dir_name:30} - OK")
        else:
            print(f"✗ {dir_name:30} - NOT FOUND")
            all_passed = False
    
    print("\nFiles:")
    for file_name in required_files:
        if os.path.isfile(file_name):
            print(f"✓ {file_name:30} - OK")
        else:
            print(f"✗ {file_name:30} - NOT FOUND")
            all_passed = False
    
    return all_passed


def test_config():
    """Test configuration file."""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION")
    print("="*60)
    
    try:
        import yaml
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['camera', 'model', 'detection', 'emotions', 'logging', 'display']
        
        for key in required_keys:
            if key in config:
                print(f"✓ {key:20} - OK")
            else:
                print(f"✗ {key:20} - MISSING")
                return False
        
        print(f"\nCamera index: {config['camera']['index']}")
        print(f"Model path: {config['model']['path']}")
        print(f"Face detector: {config['detection']['face_detector']}")
        print(f"Emotions: {len(config['emotions'])} supported")
        
        return True
    
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_face_detector():
    """Test face detector initialization."""
    print("\n" + "="*60)
    print("TESTING FACE DETECTOR")
    print("="*60)
    
    try:
        from face_detection.face_detector import FaceDetector
        
        # Test Haar Cascade
        detector_haar = FaceDetector(method="haarcascade")
        print("✓ Haar Cascade detector - OK")
        
        # Test DNN
        try:
            detector_dnn = FaceDetector(method="dnn")
            print("✓ DNN detector - OK")
        except:
            print("✗ DNN detector - NOT AVAILABLE (This is OK)")
        
        return True
    
    except Exception as e:
        print(f"✗ Face detector test failed: {e}")
        return False


def test_emotion_classifier():
    """Test emotion classifier initialization."""
    print("\n" + "="*60)
    print("TESTING EMOTION CLASSIFIER")
    print("="*60)
    
    try:
        from emotion_recognition.emotion_classifier import EmotionClassifier
        
        classifier = EmotionClassifier()
        print("✓ Emotion classifier - OK")
        
        emotions = classifier.get_emotions()
        print(f"✓ Supported emotions: {len(emotions)}")
        print(f"  {', '.join(emotions)}")
        
        return True
    
    except Exception as e:
        print(f"✗ Emotion classifier test failed: {e}")
        return False


def test_utilities():
    """Test utility modules."""
    print("\n" + "="*60)
    print("TESTING UTILITIES")
    print("="*60)
    
    try:
        from utils.fps import FPSCounter
        from utils.logger import EmotionLogger
        from utils.draw import Visualizer
        
        # Test FPS counter
        fps_counter = FPSCounter()
        fps_counter.update()
        fps = fps_counter.get_fps()
        print("✓ FPS Counter - OK")
        
        # Test logger
        logger = EmotionLogger(log_file="output/test_log.csv")
        print("✓ Logger - OK")
        
        # Test visualizer
        visualizer = Visualizer()
        print("✓ Visualizer - OK")
        
        return True
    
    except Exception as e:
        print(f"✗ Utilities test failed: {e}")
        return False


def test_models_directory():
    """Test if models directory is accessible."""
    print("\n" + "="*60)
    print("TESTING MODELS DIRECTORY")
    print("="*60)
    
    try:
        import os
        
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        print(f"✓ Models directory - OK")
        
        model_path = "models/emotion_model.h5"
        if os.path.exists(model_path):
            print(f"✓ Trained model found at {model_path}")
            return True
        else:
            print(f"⚠ No trained model found (this is normal on first run)")
            print(f"  Run: python training/train_model.py")
            return True
    
    except Exception as e:
        print(f"✗ Models directory test failed: {e}")
        return False


def test_system_requirements():
    """Test system requirements."""
    print("\n" + "="*60)
    print("TESTING SYSTEM REQUIREMENTS")
    print("="*60)
    
    try:
        import cv2
        import sys
        
        # Python version
        if sys.version_info >= (3, 8):
            print(f"✓ Python version {sys.version.split()[0]} - OK")
        else:
            print(f"✗ Python version {sys.version.split()[0]} - REQUIREMENT: 3.8+")
            return False
        
        # OpenCV version
        cv_version = cv2.__version__
        print(f"✓ OpenCV version {cv_version} - OK")
        
        # Test webcam access
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print(f"✓ Webcam access - OK")
            cap.release()
        else:
            print(f"⚠ Webcam not detected (may not have one)")
        
        return True
    
    except Exception as e:
        print(f"✗ System requirements test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  EMOTION DETECTION SYSTEM - VERIFICATION TEST".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Project Structure", test_project_structure()))
    results.append(("Configuration", test_config()))
    results.append(("System Requirements", test_system_requirements()))
    results.append(("Face Detector", test_face_detector()))
    results.append(("Emotion Classifier", test_emotion_classifier()))
    results.append(("Utilities", test_utilities()))
    results.append(("Models Directory", test_models_directory()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:30} {status}")
    
    print("="*60)
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Collect training data: python dataset/collect_data.py")
        print("2. Train the model: python training/train_model.py")
        print("3. Run inference: python main.py")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please fix the issues above.")
        print("\nInstall missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
