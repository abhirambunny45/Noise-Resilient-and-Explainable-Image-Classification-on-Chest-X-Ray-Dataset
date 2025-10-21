"""
Model definitions for chest X-ray pneumonia classification.
Includes CNN, Transformer, and traditional ML models.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')


class CNNModel:
    """CNN-based model for chest X-ray classification."""
    
    def __init__(self, input_shape=(224, 224, 1), num_classes=2, model_type='custom'):
        """
        Initialize CNN model.
        
        Args:
            input_shape (tuple): Input image shape
            num_classes (int): Number of output classes
            model_type (str): Type of CNN model ('custom', 'vgg16', 'resnet50', 'densenet121')
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = None
        self.history = None
        
    def build_custom_cnn(self):
        """Build custom CNN architecture."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Convert grayscale to RGB for transfer learning
        if self.input_shape[-1] == 1:
            x = layers.Conv2D(3, (1, 1), padding='same')(inputs)
        else:
            x = inputs
            
        # First block
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Second block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Third block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Fourth block
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        return self.model
    
    def build_transfer_learning_model(self, base_model_name='vgg16'):
        """Build transfer learning model using pre-trained networks."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Convert grayscale to RGB
        if self.input_shape[-1] == 1:
            x = layers.Conv2D(3, (1, 1), padding='same')(inputs)
        else:
            x = inputs
        
        # Load pre-trained model
        if base_model_name == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_tensor=x)
        elif base_model_name == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x)
        elif base_model_name == 'densenet121':
            base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=x)
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        return self.model
    
    def build_model(self):
        """Build the model based on specified type."""
        if self.model_type == 'custom':
            return self.build_custom_cnn()
        elif self.model_type in ['vgg16', 'resnet50', 'densenet121']:
            return self.build_transfer_learning_model(self.model_type)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function."""
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, train_generator, val_generator, epochs=50, batch_size=32):
        """Train the model."""
        if self.model is None:
            self.compile_model()
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, test_generator):
        """Evaluate the model on test data."""
        if self.model is None:
            raise ValueError("Model not built yet")
        
        results = self.model.evaluate(test_generator, verbose=0)
        return dict(zip(self.model.metrics_names, results))
    
    def predict(self, test_generator):
        """Make predictions on test data."""
        if self.model is None:
            raise ValueError("Model not built yet")
        
        predictions = self.model.predict(test_generator, verbose=0)
        return predictions
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            raise ValueError("Model not trained yet")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/figures/cnn_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


class TraditionalMLModels:
    """Traditional machine learning models for feature-based classification."""
    
    def __init__(self):
        """Initialize traditional ML models."""
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        self.scaler = None
        self.results = {}
        
    def train_evaluate_models(self, X, y, feature_names):
        """
        Train and evaluate traditional ML models.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target labels
            feature_names (list): List of feature names
            
        Returns:
            dict: Results dictionary containing model performance metrics
        """
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Results storage
        self.results = {}
        
        # Plotting setup
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Train and evaluate each model
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}:")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            self.results[model_name] = {
                'cv_scores': cv_scores,
                'mean_cv': cv_scores.mean(),
                'std_cv': cv_scores.std()
            }
            
            # ROC and PR curves using cross-validation
            tprs = []
            aucs = []
            precisions = []
            recalls = []
            mean_fpr = np.linspace(0, 1, 100)
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
                model.fit(X_scaled[train_idx], y[train_idx])
                probas = model.predict_proba(X_scaled[val_idx])
                
                # ROC
                fpr, tpr, _ = roc_curve(y[val_idx], probas[:, 1])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                aucs.append(auc(fpr, tpr))
                
                # Precision-Recall
                precision, recall, _ = precision_recall_curve(y[val_idx], probas[:, 1])
                precisions.append(np.interp(mean_fpr, recall[::-1], precision[::-1]))
            
            # Plot ROC curves
            mean_tpr = np.mean(tprs, axis=0)
            mean_auc = np.mean(aucs)
            ax1.plot(mean_fpr, mean_tpr, label=f'{model_name} (AUC = {mean_auc:.2f})')
            
            # Plot PR curves
            mean_precision = np.mean(precisions, axis=0)
            ax2.plot(mean_fpr, mean_precision, label=f'{model_name}')
            
            print(f"Mean CV Accuracy: {self.results[model_name]['mean_cv']:.3f} "
                  f"(±{self.results[model_name]['std_cv']:.3f})")
            
            # Feature importance for Random Forest
            if model_name == 'Random Forest':
                model.fit(X_scaled, y)
                importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                sns.barplot(data=importance.head(10), x='importance', y='feature', ax=ax3)
                ax3.set_title('Top 10 Feature Importance (Random Forest)')
                ax3.set_xlabel('Importance Score')
        
        # Compare model performances
        cv_means = [self.results[model]['mean_cv'] for model in self.models]
        cv_stds = [self.results[model]['std_cv'] for model in self.models]
        
        ax4.bar(self.models.keys(), cv_means, yerr=cv_stds, capsize=5)
        ax4.set_title('Model Comparison')
        ax4.set_ylabel('Cross-validation Accuracy')
        ax4.set_ylim(0.5, 1.0)
        
        # Finalize plots
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('results/figures/traditional_ml_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.results, X_scaled
    
    def get_best_model(self):
        """Get the best performing model based on cross-validation scores."""
        if not self.results:
            raise ValueError("Models not trained yet")
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['mean_cv'])
        best_model = self.models[best_model_name]
        
        # Retrain on full dataset
        best_model.fit(self.scaler.transform(X), y)
        
        return best_model_name, best_model
    
    def evaluate_final_model(self, model, test_X, test_y, test_paths):
        """
        Evaluate the final model on test data.
        
        Args:
            model: Trained model
            test_X (numpy.ndarray): Test features
            test_y (numpy.ndarray): Test labels
            test_paths (list): Test image paths
        """
        # Scale test data
        test_X_scaled = self.scaler.transform(test_X)
        
        # Make predictions
        predictions = model.predict(test_X_scaled)
        probas = model.predict_proba(test_X_scaled)
        
        # Calculate metrics
        print("\nTest Set Performance:")
        print(classification_report(test_y, predictions))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(test_y, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('results/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze misclassifications
        misclassified_idx = np.where(predictions != test_y)[0]
        
        if len(misclassified_idx) > 0:
            print("\nAnalyzing misclassified cases:")
            for idx in misclassified_idx[:5]:  # Show first 5 misclassifications
                print(f"\nImage: {test_paths[idx]}")
                print(f"Actual: {'PNEUMONIA' if test_y[idx] == 1 else 'NORMAL'}")
                print(f"Predicted: {'PNEUMONIA' if predictions[idx] == 1 else 'NORMAL'}")
                print(f"Probability: {probas[idx][1]:.3f}")


class VisionTransformer:
    """Vision Transformer model for chest X-ray classification."""
    
    def __init__(self, input_shape=(224, 224, 1), num_classes=2, patch_size=16, num_patches=196, 
                 projection_dim=64, num_heads=4, transformer_layers=8, mlp_head_units=[2048, 1024]):
        """
        Initialize Vision Transformer.
        
        Args:
            input_shape (tuple): Input image shape
            num_classes (int): Number of output classes
            patch_size (int): Size of image patches
            num_patches (int): Number of patches
            projection_dim (int): Projection dimension
            num_heads (int): Number of attention heads
            transformer_layers (int): Number of transformer layers
            mlp_head_units (list): MLP head units
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.mlp_head_units = mlp_head_units
        self.model = None
        
    def build_model(self):
        """Build Vision Transformer model."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Convert grayscale to RGB
        if self.input_shape[-1] == 1:
            x = layers.Conv2D(3, (1, 1), padding='same')(inputs)
        else:
            x = inputs
        
        # Create patches
        patches = layers.Conv2D(
            self.projection_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid"
        )(x)
        patch_dims = patches.shape[-1]
        patches = layers.Reshape((self.num_patches, patch_dims))(patches)
        
        # Add position embeddings
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dim
        )(positions)
        encoded = patches + position_embedding
        
        # Transformer blocks
        for _ in range(self.transformer_layers):
            # Layer normalization 1
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded)
            
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)
            
            # Skip connection 1
            x2 = layers.Add()([attention_output, encoded])
            
            # Layer normalization 2
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            
            # MLP
            x3 = layers.Dense(self.projection_dim * 2, activation="gelu")(x3)
            x3 = layers.Dropout(0.1)(x3)
            x3 = layers.Dense(self.projection_dim, activation="gelu")(x3)
            
            # Skip connection 2
            encoded = layers.Add()([x3, x2])
        
        # Global average pooling
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded)
        attention_weights = tf.nn.softmax(layers.Dense(1)(representation), axis=1)
        weighted_representation = tf.reduce_sum(attention_weights * representation, axis=1)
        
        # MLP head
        for units in self.mlp_head_units:
            weighted_representation = layers.Dense(units, activation="gelu")(weighted_representation)
            weighted_representation = layers.Dropout(0.5)(weighted_representation)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation="softmax")(weighted_representation)
        
        self.model = Model(inputs, outputs)
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the Vision Transformer model."""
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )


def main():
    """Main function to demonstrate model usage."""
    # Example usage of CNN model
    cnn_model = CNNModel(model_type='custom')
    cnn_model.compile_model()
    print("CNN Model Summary:")
    cnn_model.model.summary()
    
    # Example usage of traditional ML models
    ml_models = TraditionalMLModels()
    print("\nTraditional ML Models initialized:")
    for name in ml_models.models.keys():
        print(f"- {name}")
    
    # Example usage of Vision Transformer
    vit_model = VisionTransformer()
    vit_model.compile_model()
    print("\nVision Transformer Model Summary:")
    vit_model.model.summary()


if __name__ == "__main__":
    main()
