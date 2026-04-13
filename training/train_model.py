"""
Model Training Script
Trains deep learning model for emotion recognition
"""
import os
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


class EmotionModelTrainer:
    """Trains emotion recognition CNN model."""
    
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to configuration file
        """
        # Make config path absolute if it's relative
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "..", config_path)
            config_path = os.path.normpath(config_path)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.input_size = self.config['model']['input_size']
        self.batch_size = self.config['training']['batch_size']
        self.epochs = self.config['training']['epochs']
        self.learning_rate = self.config['training']['learning_rate']
        self.model_path = self.config['model']['path']
        
        # Make model path absolute if it's relative
        if not os.path.isabs(self.model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(script_dir, "..", self.model_path)
            self.model_path = os.path.normpath(self.model_path)
        
        self.model = None
        self.history = None
    
    def load_dataset(self, dataset_path="dataset/collected_images"):
        """
        Load dataset from directory.
        
        Args:
            dataset_path: Path to dataset directory
        
        Returns:
            tuple: (X_data, y_data)
        """
        # Make dataset path absolute if it's relative
        if not os.path.isabs(dataset_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_path = os.path.join(script_dir, "..", dataset_path)
            dataset_path = os.path.normpath(dataset_path)
        
        print(f"\nLoading dataset from {dataset_path}...")
        
        X_data = []
        y_data = []
        
        for emotion_idx, emotion in enumerate(self.EMOTIONS):
            emotion_dir = os.path.join(dataset_path, emotion)
            
            if not os.path.exists(emotion_dir):
                print(f"Warning: {emotion_dir} not found")
                continue
            
            image_count = 0
            for image_file in os.listdir(emotion_dir):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        img_path = os.path.join(emotion_dir, image_file)
                        
                        # Read image
                        image = cv2.imread(img_path)
                        if image is None:
                            continue
                        
                        # Convert BGR to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Resize to input size
                        image = cv2.resize(image, (self.input_size, self.input_size))
                        
                        # Normalize
                        image = image.astype('float32') / 255.0
                        
                        X_data.append(image)
                        y_data.append(emotion_idx)
                        image_count += 1
                    
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
            
            print(f"  {emotion}: {image_count} images")
        
        if len(X_data) == 0:
            raise ValueError("No images found in dataset. Please collect data first.")
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        print(f"\nTotal images loaded: {len(X_data)}")
        print(f"Dataset shape: {X_data.shape}")
        
        return X_data, y_data
    
    def build_model(self):
        """
        Build CNN model architecture.
        
        Returns:
            Sequential: Compiled Keras model
        """
        print("\nBuilding CNN model...")
        
        model = Sequential([
            # Block 1
            Conv2D(32, (3, 3), activation='relu', padding='same',
                   input_shape=(self.input_size, self.input_size, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 2
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 3
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 4
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Flatten and Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(len(self.EMOTIONS), activation='softmax')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        model.summary()
        
        self.model = model
        return model
    
    def train(self, X_data, y_data):
        """
        Train the model.
        
        Args:
            X_data: Training images
            y_data: Training labels
        
        Returns:
            History: Training history
        """
        if self.model is None:
            self.build_model()
        
        # Split data
        print("\nSplitting dataset...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_data, y_data,
            test_size=self.config['training']['validation_split'],
            random_state=42,
            stratify=y_data
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        
        # Data augmentation
        print("\nSetting up data augmentation...")
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['training']['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        print("\nStarting training...")
        self.history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=self.batch_size),
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test images
            y_test: Test labels
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        print("\nEvaluating model...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Predictions
        y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.EMOTIONS))
        
        return test_loss, test_accuracy
    
    def save_model(self):
        """Save trained model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else ".", exist_ok=True)
        self.model.save(self.model_path)
        print(f"\nModel saved to {self.model_path}")
    
    def plot_training_history(self, save_path="output/training_history.png"):
        """
        Plot and save training history.
        
        Args:
            save_path: Path to save plot
        """
        if self.history is None:
            print("No training history to plot")
            return
        
        # Make save path absolute if it's relative
        if not os.path.isabs(save_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(script_dir, "..", save_path)
            save_path = os.path.normpath(save_path)
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"\nTraining history plot saved to {save_path}")
        plt.close()


def main():
    """Main entry point."""
    print("="*60)
    print("EMOTION RECOGNITION MODEL TRAINER")
    print("="*60)
    
    try:
        # Initialize trainer
        trainer = EmotionModelTrainer()
        
        # Load dataset
        X_data, y_data = trainer.load_dataset()
        
        # Build model
        trainer.build_model()
        
        # Train model
        trainer.train(X_data, y_data)
        
        # Evaluate model
        trainer.evaluate(X_data, y_data)
        
        # Save model
        trainer.save_model()
        
        # Plot training history
        trainer.plot_training_history()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")
    
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
