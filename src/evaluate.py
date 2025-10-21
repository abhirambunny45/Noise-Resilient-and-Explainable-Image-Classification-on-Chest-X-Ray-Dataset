"""
Evaluation script for chest X-ray pneumonia classification.
Includes evaluation metrics and explainability tools (Grad-CAM, LIME).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
import cv2
import lime
from lime import lime_image, lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_preprocessing import ChestXRayPreprocessor
from model import CNNModel, TraditionalMLModels, VisionTransformer


class ModelEvaluator:
    """Main evaluation class for all model types."""
    
    def __init__(self, base_dir='chest_xray/chest_xray', results_dir='results'):
        """
        Initialize the evaluator.
        
        Args:
            base_dir (str): Base directory containing the dataset
            results_dir (str): Directory containing results
        """
        self.base_dir = base_dir
        self.results_dir = results_dir
        self.preprocessor = ChestXRayPreprocessor(base_dir)
        
        # Create evaluation directories
        os.makedirs(f'{results_dir}/figures', exist_ok=True)
        os.makedirs(f'{results_dir}/explanations', exist_ok=True)
        
    def evaluate_cnn_model(self, model_path, model_type='custom'):
        """
        Evaluate CNN model and generate explanations.
        
        Args:
            model_path (str): Path to saved model
            model_type (str): Type of CNN model
            
        Returns:
            dict: Evaluation results
        """
        print(f"Evaluating CNN model: {model_type}")
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Prepare test data
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            self.preprocessor.test_dir,
            target_size=(224, 224),
            batch_size=1,
            class_mode='binary',
            color_mode='grayscale',
            shuffle=False
        )
        
        # Get predictions
        predictions = model.predict(test_generator, verbose=0)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        
        # Get true labels
        true_labels = test_generator.classes
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_classes)
        
        # Classification report
        print(f"\nCNN {model_type} Test Results:")
        print(f"Accuracy: {accuracy:.3f}")
        print(classification_report(true_labels, predicted_classes, 
                                  target_names=['NORMAL', 'PNEUMONIA']))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(true_labels, predicted_classes, f'CNN {model_type}')
        
        # Generate Grad-CAM explanations
        self.generate_gradcam_explanations(model, test_generator, model_type)
        
        # Generate LIME explanations
        self.generate_lime_explanations(model, test_generator, model_type)
        
        return {
            'model_type': f'cnn_{model_type}',
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def evaluate_traditional_ml_model(self, model, scaler, test_X, test_y, test_paths):
        """
        Evaluate traditional ML model and generate explanations.
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            test_X (numpy.ndarray): Test features
            test_y (numpy.ndarray): Test labels
            test_paths (list): Test image paths
            
        Returns:
            dict: Evaluation results
        """
        print("Evaluating Traditional ML Model")
        
        # Scale test data
        test_X_scaled = scaler.transform(test_X)
        
        # Get predictions
        predictions = model.predict(test_X_scaled)
        probabilities = model.predict_proba(test_X_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(test_y, predictions)
        
        # Classification report
        print(f"\nTraditional ML Test Results:")
        print(f"Accuracy: {accuracy:.3f}")
        print(classification_report(test_y, predictions, 
                                  target_names=['NORMAL', 'PNEUMONIA']))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(test_y, predictions, 'Traditional ML')
        
        # Generate LIME explanations for traditional ML
        self.generate_lime_tabular_explanations(model, test_X_scaled, test_y, test_paths)
        
        return {
            'model_type': 'traditional_ml',
            'accuracy': accuracy,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': test_y
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['NORMAL', 'PNEUMONIA'],
                    yticklabels=['NORMAL', 'PNEUMONIA'])
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/figures/{model_name.lower().replace(" ", "_")}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_gradcam_explanations(self, model, test_generator, model_name, num_samples=5):
        """
        Generate Grad-CAM explanations for CNN models.
        
        Args:
            model: Trained CNN model
            test_generator: Test data generator
            model_name (str): Name of the model
            num_samples (int): Number of samples to explain
        """
        print(f"Generating Grad-CAM explanations for {model_name}")
        
        # Get sample images
        sample_images = []
        sample_labels = []
        
        for i in range(num_samples):
            img, label = test_generator.next()
            sample_images.append(img[0])
            sample_labels.append(label[0])
        
        # Generate Grad-CAM for each sample
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
        
        for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
            # Original image
            axes[0, i].imshow(img.squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original\nLabel: {"PNEUMONIA" if label == 1 else "NORMAL"}')
            axes[0, i].axis('off')
            
            # Generate Grad-CAM
            gradcam = self.compute_gradcam(model, img)
            
            # Overlay Grad-CAM on original image
            axes[1, i].imshow(img.squeeze(), cmap='gray')
            axes[1, i].imshow(gradcam, cmap='jet', alpha=0.5)
            axes[1, i].set_title('Grad-CAM Overlay')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/explanations/{model_name.lower().replace(" ", "_")}_gradcam.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def compute_gradcam(self, model, img_array, pred_index=None):
        """
        Compute Grad-CAM for a given image.
        
        Args:
            model: Trained model
            img_array: Input image array
            pred_index: Prediction index for multi-class models
            
        Returns:
            numpy.ndarray: Grad-CAM heatmap
        """
        # Get the last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            print("No convolutional layer found for Grad-CAM")
            return np.zeros(img_array.shape[:2])
        
        # Create a model that maps the input image to the activations of the last conv layer
        grad_model = tf.keras.models.Model(
            [model.inputs], [last_conv_layer.output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature map array by its importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        # Resize heatmap to match input image size
        heatmap = tf.image.resize(heatmap, (img_array.shape[1], img_array.shape[2]))
        
        return heatmap.numpy()
    
    def generate_lime_explanations(self, model, test_generator, model_name, num_samples=3):
        """
        Generate LIME explanations for CNN models.
        
        Args:
            model: Trained CNN model
            test_generator: Test data generator
            model_name (str): Name of the model
            num_samples (int): Number of samples to explain
        """
        print(f"Generating LIME explanations for {model_name}")
        
        # Get sample images
        sample_images = []
        sample_labels = []
        
        for i in range(num_samples):
            img, label = test_generator.next()
            sample_images.append(img[0])
            sample_labels.append(label[0])
        
        # Initialize LIME explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Generate explanations
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        
        for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
            # Original image
            axes[i, 0].imshow(img.squeeze(), cmap='gray')
            axes[i, 0].set_title(f'Original\nLabel: {"PNEUMONIA" if label == 1 else "NORMAL"}')
            axes[i, 0].axis('off')
            
            # Generate LIME explanation
            explanation = explainer.explain_instance(
                img.squeeze(),
                lambda x: model.predict(x.reshape(-1, 224, 224, 1)),
                top_labels=2,
                hide_color=0,
                num_samples=1000
            )
            
            # Get explanation for pneumonia class
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=5,
                hide_rest=True
            )
            
            axes[i, 1].imshow(temp, cmap='gray')
            axes[i, 1].set_title('LIME Explanation\n(Positive Features)')
            axes[i, 1].axis('off')
            
            # Get explanation for both classes
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=False,
                num_features=10,
                hide_rest=False
            )
            
            axes[i, 2].imshow(temp, cmap='gray')
            axes[i, 2].set_title('LIME Explanation\n(All Features)')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/explanations/{model_name.lower().replace(" ", "_")}_lime.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_lime_tabular_explanations(self, model, test_X_scaled, test_y, test_paths, num_samples=3):
        """
        Generate LIME explanations for traditional ML models.
        
        Args:
            model: Trained model
            test_X_scaled (numpy.ndarray): Scaled test features
            test_y (numpy.ndarray): Test labels
            test_paths (list): Test image paths
            num_samples (int): Number of samples to explain
        """
        print("Generating LIME tabular explanations for Traditional ML")
        
        # Get feature names (assuming 12 features from preprocessing)
        feature_names = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM',
                        'mean', 'std', 'variance', 'skewness', 'kurtosis', 'entropy']
        
        # Initialize LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            test_X_scaled,
            feature_names=feature_names,
            class_names=['NORMAL', 'PNEUMONIA'],
            mode='classification'
        )
        
        # Generate explanations for random samples
        np.random.seed(42)
        sample_indices = np.random.choice(len(test_X_scaled), num_samples, replace=False)
        
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i, idx in enumerate(sample_indices):
            # Generate explanation
            explanation = explainer.explain_instance(
                test_X_scaled[idx],
                model.predict_proba,
                num_features=len(feature_names)
            )
            
            # Plot explanation
            explanation.as_pyplot_figure()
            plt.title(f'Sample {i+1} - Image: {os.path.basename(test_paths[idx])}\n'
                     f'Actual: {"PNEUMONIA" if test_y[idx] == 1 else "NORMAL"}')
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/explanations/traditional_ml_lime_sample_{i+1}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def analyze_misclassifications(self, y_true, y_pred, test_paths, model_name, top_n=5):
        """
        Analyze misclassified cases.
        
        Args:
            y_true (numpy.ndarray): True labels
            y_pred (numpy.ndarray): Predicted labels
            test_paths (list): Test image paths
            model_name (str): Name of the model
            top_n (int): Number of misclassifications to analyze
        """
        print(f"\nAnalyzing misclassifications for {model_name}")
        
        # Find misclassified indices
        misclassified_idx = np.where(y_true != y_pred)[0]
        
        if len(misclassified_idx) == 0:
            print("No misclassifications found!")
            return
        
        print(f"Total misclassifications: {len(misclassified_idx)}")
        print(f"Misclassification rate: {len(misclassified_idx)/len(y_true):.3f}")
        
        # Analyze top misclassifications
        print(f"\nTop {min(top_n, len(misclassified_idx))} misclassifications:")
        for i, idx in enumerate(misclassified_idx[:top_n]):
            print(f"\n{i+1}. Image: {os.path.basename(test_paths[idx])}")
            print(f"   Actual: {'PNEUMONIA' if y_true[idx] == 1 else 'NORMAL'}")
            print(f"   Predicted: {'PNEUMONIA' if y_pred[idx] == 1 else 'NORMAL'}")
    
    def generate_comprehensive_evaluation(self):
        """Generate comprehensive evaluation of all models."""
        print("Starting comprehensive model evaluation")
        
        all_results = {}
        
        # Evaluate CNN models if they exist
        cnn_models = ['custom', 'vgg16', 'resnet50']
        for cnn_type in cnn_models:
            model_path = f'{self.results_dir}/models/cnn_{cnn_type}_best.h5'
            if os.path.exists(model_path):
                try:
                    results = self.evaluate_cnn_model(model_path, cnn_type)
                    all_results[f'cnn_{cnn_type}'] = results
                except Exception as e:
                    print(f"Error evaluating CNN {cnn_type}: {str(e)}")
        
        # Evaluate Vision Transformer if it exists
        vit_path = f'{self.results_dir}/models/vit_best.h5'
        if os.path.exists(vit_path):
            try:
                results = self.evaluate_cnn_model(vit_path, 'vision_transformer')
                all_results['vision_transformer'] = results
            except Exception as e:
                print(f"Error evaluating Vision Transformer: {str(e)}")
        
        # Evaluate traditional ML models
        try:
            # Prepare test data
            test_X, test_y, test_paths = self.preprocessor.prepare_test_data()
            
            # Load scaler and model (this would need to be saved during training)
            # For now, we'll create a simple example
            print("Note: Traditional ML evaluation requires saved model and scaler")
            print("This would typically be loaded from training results")
            
        except Exception as e:
            print(f"Error evaluating traditional ML models: {str(e)}")
        
        # Create overall comparison
        if all_results:
            self.plot_model_comparison(all_results)
        
        return all_results
    
    def plot_model_comparison(self, all_results):
        """Plot comparison of all evaluated models."""
        model_names = []
        accuracies = []
        
        for model_name, results in all_results.items():
            if 'accuracy' in results:
                model_names.append(model_name.replace('_', ' ').title())
                accuracies.append(results['accuracy'])
        
        if not model_names:
            print("No results to compare")
            return
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
        plt.title('Model Performance Comparison')
        plt.xlabel('Model Type')
        plt.ylabel('Test Accuracy')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/figures/evaluation_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print("\nModel Performance Summary:")
        for name, acc in zip(model_names, accuracies):
            print(f"{name}: {acc:.3f}")


def main():
    """Main function to run evaluation."""
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Run comprehensive evaluation
    results = evaluator.generate_comprehensive_evaluation()
    
    print("\nEvaluation completed! Check the results directory for outputs.")
    print(f"Results saved in: {evaluator.results_dir}")


if __name__ == "__main__":
    main()
