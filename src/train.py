"""
Training script for chest X-ray pneumonia classification.
Includes training loops for CNN, Vision Transformer, and traditional ML models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_preprocessing import ChestXRayPreprocessor
from model import CNNModel, TraditionalMLModels, VisionTransformer


class ModelTrainer:
    """Main training class for all model types."""
    
    def __init__(self, base_dir='chest_xray/chest_xray', results_dir='results'):
        """
        Initialize the trainer.
        
        Args:
            base_dir (str): Base directory containing the dataset
            results_dir (str): Directory to save results
        """
        self.base_dir = base_dir
        self.results_dir = results_dir
        self.preprocessor = ChestXRayPreprocessor(base_dir)
        
        # Setup logging
        self.setup_logging()
        
        # Create results directories
        os.makedirs(f'{results_dir}/logs', exist_ok=True)
        os.makedirs(f'{results_dir}/figures', exist_ok=True)
        os.makedirs(f'{results_dir}/models', exist_ok=True)
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_filename = f'{self.results_dir}/logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def prepare_data_generators(self, batch_size=32, image_size=(224, 224)):
        """
        Prepare data generators for CNN/ViT training.
        
        Args:
            batch_size (int): Batch size for training
            image_size (tuple): Target image size
            
        Returns:
            tuple: (train_generator, val_generator, test_generator)
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # No augmentation for validation and test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            self.preprocessor.train_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='training',
            color_mode='grayscale'
        )
        
        # Validation generator
        val_generator = train_datagen.flow_from_directory(
            self.preprocessor.train_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
            color_mode='grayscale'
        )
        
        # Test generator
        test_generator = val_test_datagen.flow_from_directory(
            self.preprocessor.test_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='binary',
            color_mode='grayscale',
            shuffle=False
        )
        
        self.logger.info(f"Training samples: {train_generator.samples}")
        self.logger.info(f"Validation samples: {val_generator.samples}")
        self.logger.info(f"Test samples: {test_generator.samples}")
        
        return train_generator, val_generator, test_generator
    
    def train_cnn_model(self, model_type='custom', epochs=50, batch_size=32, learning_rate=0.001):
        """
        Train CNN model.
        
        Args:
            model_type (str): Type of CNN model ('custom', 'vgg16', 'resnet50', 'densenet121')
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            
        Returns:
            dict: Training results
        """
        self.logger.info(f"Starting CNN training with {model_type} architecture")
        
        # Prepare data generators
        train_gen, val_gen, test_gen = self.prepare_data_generators(batch_size)
        
        # Initialize and compile model
        cnn_model = CNNModel(model_type=model_type)
        cnn_model.compile_model(learning_rate)
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint(
                f'{self.results_dir}/models/cnn_{model_type}_best.h5',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            )
        ]
        
        # Train model
        self.logger.info("Starting training...")
        history = cnn_model.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        test_results = cnn_model.evaluate(test_gen)
        self.logger.info(f"Test results: {test_results}")
        
        # Plot training history
        cnn_model.plot_training_history()
        
        # Save results
        results = {
            'model_type': model_type,
            'test_accuracy': test_results['accuracy'],
            'test_loss': test_results['loss'],
            'training_history': history.history,
            'epochs_trained': len(history.history['loss'])
        }
        
        # Save to JSON
        with open(f'{self.results_dir}/logs/cnn_{model_type}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def train_vit_model(self, epochs=50, batch_size=32, learning_rate=0.001):
        """
        Train Vision Transformer model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            
        Returns:
            dict: Training results
        """
        self.logger.info("Starting Vision Transformer training")
        
        # Prepare data generators
        train_gen, val_gen, test_gen = self.prepare_data_generators(batch_size)
        
        # Initialize and compile model
        vit_model = VisionTransformer()
        vit_model.compile_model(learning_rate)
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-7, verbose=1),
            ModelCheckpoint(
                f'{self.results_dir}/models/vit_best.h5',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            )
        ]
        
        # Train model
        self.logger.info("Starting Vision Transformer training...")
        history = vit_model.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        test_results = vit_model.evaluate(test_gen)
        self.logger.info(f"Vision Transformer test results: {test_results}")
        
        # Plot training history
        self.plot_training_history(history, 'Vision Transformer')
        
        # Save results
        results = {
            'model_type': 'vision_transformer',
            'test_accuracy': test_results['accuracy'],
            'test_loss': test_results['loss'],
            'training_history': history.history,
            'epochs_trained': len(history.history['loss'])
        }
        
        # Save to JSON
        with open(f'{self.results_dir}/logs/vit_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def train_traditional_ml_models(self, n_samples=100):
        """
        Train traditional ML models using extracted features.
        
        Args:
            n_samples (int): Number of samples per class for training
            
        Returns:
            dict: Training results
        """
        self.logger.info("Starting traditional ML models training")
        
        # Prepare dataset
        X, y, feature_names = self.preprocessor.prepare_dataset(n_samples)
        
        # Initialize traditional ML models
        ml_models = TraditionalMLModels()
        
        # Train and evaluate models
        results, X_scaled = ml_models.train_evaluate_models(X, y, feature_names)
        
        # Get best model
        best_model_name, best_model = ml_models.get_best_model()
        self.logger.info(f"Best traditional ML model: {best_model_name}")
        
        # Prepare test data
        test_X, test_y, test_paths = self.preprocessor.prepare_test_data()
        
        # Evaluate on test set
        ml_models.evaluate_final_model(best_model, test_X, test_y, test_paths)
        
        # Save results
        ml_results = {
            'model_type': 'traditional_ml',
            'best_model': best_model_name,
            'cv_results': results,
            'feature_names': feature_names
        }
        
        # Save to JSON
        with open(f'{self.results_dir}/logs/traditional_ml_results.json', 'w') as f:
            json.dump(ml_results, f, indent=2)
        
        return ml_results
    
    def plot_training_history(self, history, model_name):
        """Plot training history for any model."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_name} - Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/figures/{model_name.lower().replace(" ", "_")}_training_history.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_all_models(self):
        """Train and compare all model types."""
        self.logger.info("Starting comprehensive model comparison")
        
        all_results = {}
        
        # Train CNN models
        cnn_types = ['custom', 'vgg16', 'resnet50']
        for cnn_type in cnn_types:
            try:
                self.logger.info(f"Training CNN with {cnn_type} architecture")
                results = self.train_cnn_model(cnn_type, epochs=30)
                all_results[f'cnn_{cnn_type}'] = results
            except Exception as e:
                self.logger.error(f"Error training CNN {cnn_type}: {str(e)}")
        
        # Train Vision Transformer
        try:
            self.logger.info("Training Vision Transformer")
            vit_results = self.train_vit_model(epochs=30)
            all_results['vision_transformer'] = vit_results
        except Exception as e:
            self.logger.error(f"Error training Vision Transformer: {str(e)}")
        
        # Train traditional ML models
        try:
            self.logger.info("Training traditional ML models")
            ml_results = self.train_traditional_ml_models()
            all_results['traditional_ml'] = ml_results
        except Exception as e:
            self.logger.error(f"Error training traditional ML models: {str(e)}")
        
        # Create comparison plot
        self.plot_model_comparison(all_results)
        
        # Save all results
        with open(f'{self.results_dir}/logs/all_model_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return all_results
    
    def plot_model_comparison(self, all_results):
        """Plot comparison of all models."""
        model_names = []
        test_accuracies = []
        
        for model_name, results in all_results.items():
            if 'test_accuracy' in results:
                model_names.append(model_name.replace('_', ' ').title())
                test_accuracies.append(results['test_accuracy'])
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, test_accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
        plt.title('Model Performance Comparison')
        plt.xlabel('Model Type')
        plt.ylabel('Test Accuracy')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, test_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/figures/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        self.logger.info("\nModel Performance Summary:")
        for name, acc in zip(model_names, test_accuracies):
            self.logger.info(f"{name}: {acc:.3f}")


def main():
    """Main function to run training."""
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Print dataset statistics
    trainer.preprocessor.print_dataset_stats()
    
    # Run comprehensive model comparison
    results = trainer.compare_all_models()
    
    print("\nTraining completed! Check the results directory for outputs.")
    print(f"Results saved in: {trainer.results_dir}")


if __name__ == "__main__":
    main()
