"""
Chest X-ray Pneumonia Classification Package

This package provides comprehensive tools for chest X-ray pneumonia classification
using both deep learning and traditional machine learning approaches.

Modules:
- data_preprocessing: Data loading, preprocessing, and feature extraction
- model: Model definitions (CNN, Vision Transformer, Traditional ML)
- train: Training scripts for all model types
- evaluate: Evaluation metrics and explainability tools

Author: Generated for CSCE 5222 Feature Engineering Project
"""

# Import main classes for easy access
from .data_preprocessing import ChestXRayPreprocessor
from .model import CNNModel, TraditionalMLModels, VisionTransformer
from .train import ModelTrainer
from .evaluate import ModelEvaluator

# Version information
__version__ = "1.0.0"
__author__ = "CSCE 5222 Feature Engineering Project"

# Package metadata
__all__ = [
    'ChestXRayPreprocessor',
    'CNNModel', 
    'TraditionalMLModels',
    'VisionTransformer',
    'ModelTrainer',
    'ModelEvaluator'
]

# Package description
__description__ = """
Noise-Resilient and Explainable Image Classification on Chest X-Ray Dataset

This package implements a comprehensive solution for chest X-ray pneumonia 
classification that includes:

1. Data Preprocessing:
   - Noise analysis and reduction techniques
   - Feature extraction (texture and statistical features)
   - Data augmentation and normalization

2. Model Architectures:
   - Custom CNN with batch normalization and dropout
   - Transfer learning with VGG16, ResNet50, DenseNet121
   - Vision Transformer for attention-based classification
   - Traditional ML models (Random Forest, SVM, Neural Networks)

3. Training Framework:
   - Comprehensive training pipeline for all model types
   - Cross-validation and hyperparameter optimization
   - Model comparison and performance analysis

4. Evaluation and Explainability:
   - Comprehensive evaluation metrics
   - Grad-CAM for CNN model interpretability
   - LIME explanations for both CNN and traditional ML models
   - Misclassification analysis

Key Features:
- Noise-resilient preprocessing techniques
- Multiple model architectures for comparison
- Comprehensive explainability tools
- Robust evaluation framework
- Production-ready code structure
"""

def get_package_info():
    """Get package information."""
    return {
        'name': 'chest_xray_classification',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'modules': __all__
    }

def print_package_info():
    """Print package information."""
    info = get_package_info()
    print(f"Package: {info['name']}")
    print(f"Version: {info['version']}")
    print(f"Author: {info['author']}")
    print(f"Modules: {', '.join(info['modules'])}")
    print(f"\nDescription:\n{info['description']}")
