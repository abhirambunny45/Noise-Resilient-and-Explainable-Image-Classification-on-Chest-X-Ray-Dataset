"""
Data preprocessing module for chest X-ray pneumonia classification.
Includes noise analysis, preprocessing functions, and feature extraction.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import graycomatrix, graycoprops
import warnings
warnings.filterwarnings('ignore')


class ChestXRayPreprocessor:
    """Main class for chest X-ray data preprocessing and feature extraction."""
    
    def __init__(self, base_dir='chest_xray/chest_xray'):
        """
        Initialize the preprocessor.
        
        Args:
            base_dir (str): Base directory containing the dataset
        """
        self.base_dir = base_dir
        self.train_dir = os.path.join(base_dir, 'train')
        self.test_dir = os.path.join(base_dir, 'test')
        self.val_dir = os.path.join(base_dir, 'val')
        self.scaler = StandardScaler()
        
    def print_dataset_stats(self):
        """Print and visualize dataset statistics."""
        stats = {}
        for dir_name, dir_path in [('Train', self.train_dir), ('Test', self.test_dir), ('Val', self.val_dir)]:
            normal = len(os.listdir(os.path.join(dir_path, 'NORMAL')))
            pneumonia = len(os.listdir(os.path.join(dir_path, 'PNEUMONIA')))
            stats[dir_name] = {'NORMAL': normal, 'PNEUMONIA': pneumonia, 'Total': normal + pneumonia}

        df_stats = pd.DataFrame(stats).T
        print("\nDataset Statistics:")
        print(df_stats)

        # Create subplots for better visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Bar plot
        df_stats[['NORMAL', 'PNEUMONIA']].plot(kind='bar', ax=ax1)
        ax1.set_title('Dataset Distribution')
        ax1.set_xlabel('Dataset Split')
        ax1.set_ylabel('Number of Images')
        ax1.legend(title='Class')

        # Pie chart for class distribution in training set
        train_dist = [stats['Train']['NORMAL'], stats['Train']['PNEUMONIA']]
        ax2.pie(train_dist, labels=['NORMAL', 'PNEUMONIA'], autopct='%1.1f%%')
        ax2.set_title('Training Set Class Distribution')

        plt.tight_layout()
        plt.savefig('results/figures/dataset_stats.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print detailed statistics
        total_images = sum(stats[split]['Total'] for split in stats)
        print(f"\nTotal Images in Dataset: {total_images}")
        for split in stats:
            total = stats[split]['Total']
            normal_pct = (stats[split]['NORMAL'] / total) * 100
            pneumonia_pct = (stats[split]['PNEUMONIA'] / total) * 100
            print(f"\n{split} Split:")
            print(f"Total Images: {total}")
            print(f"NORMAL: {stats[split]['NORMAL']} ({normal_pct:.1f}%)")
            print(f"PNEUMONIA: {stats[split]['PNEUMONIA']} ({pneumonia_pct:.1f}%)")

        return stats

    def analyze_image_noise(self, image_path):
        """
        Analyze noise metrics in an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Dictionary containing noise metrics
        """
        img = cv2.imread(image_path, 0)
        if img is None:
            return None
        # Calculate multiple noise metrics
        std_dev = np.std(img)
        entropy = -np.sum(np.histogram(img, bins=256, density=True)[0] *
                         np.log2(np.histogram(img, bins=256, density=True)[0] + 1e-10))
        return {
            'std_dev': std_dev,
            'entropy': entropy,
            'mean': np.mean(img),
            'variance': np.var(img)
        }

    def preprocess_image(self, image, apply_noise_reduction=True):
        """
        Apply preprocessing techniques to reduce noise.
        
        Args:
            image (numpy.ndarray): Input image
            apply_noise_reduction (bool): Whether to apply noise reduction
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        if apply_noise_reduction:
            # Apply multiple noise reduction techniques
            # 1. Gaussian blur
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            # 2. Median filtering
            median = cv2.medianBlur(blurred, 5)
            # 3. Bilateral filtering for edge preservation
            bilateral = cv2.bilateralFilter(median, 9, 75, 75)
            return bilateral
        return image

    def visualize_preprocessing_effects(self):
        """Visualize the effects of preprocessing techniques."""
        # Get sample images
        normal_path = os.path.join(self.train_dir, 'NORMAL', os.listdir(os.path.join(self.train_dir, 'NORMAL'))[0])
        pneumonia_path = os.path.join(self.train_dir, 'PNEUMONIA', os.listdir(os.path.join(self.train_dir, 'PNEUMONIA'))[0])

        images = {'NORMAL': normal_path, 'PNEUMONIA': pneumonia_path}

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        for idx, (label, img_path) in enumerate(images.items()):
            img = cv2.imread(img_path, 0)

            # Original
            axes[idx, 0].imshow(img, cmap='gray')
            axes[idx, 0].set_title(f'{label} - Original')

            # Gaussian Blur
            gaussian = cv2.GaussianBlur(img, (5, 5), 0)
            axes[idx, 1].imshow(gaussian, cmap='gray')
            axes[idx, 1].set_title('After Gaussian Blur')

            # Median Filter
            median = cv2.medianBlur(gaussian, 5)
            axes[idx, 2].imshow(median, cmap='gray')
            axes[idx, 2].set_title('After Median Filter')

            # Final Preprocessing
            final = self.preprocess_image(img)
            axes[idx, 3].imshow(final, cmap='gray')
            axes[idx, 3].set_title('Final Preprocessed')

            # Calculate and print noise metrics
            original_metrics = self.analyze_image_noise(img_path)
            processed_img_path = '/tmp/processed.png'
            cv2.imwrite(processed_img_path, final)
            processed_metrics = self.analyze_image_noise(processed_img_path)

            print(f"\nNoise Metrics for {label}:")
            print("Original Image:")
            for metric, value in original_metrics.items():
                print(f"{metric}: {value:.2f}")
            print("\nProcessed Image:")
            for metric, value in processed_metrics.items():
                print(f"{metric}: {value:.2f}")

        plt.tight_layout()
        plt.savefig('results/figures/preprocessing_effects.png', dpi=300, bbox_inches='tight')
        plt.show()

    def extract_texture_features(self, image):
        """
        Extract texture features using Gray-Level Co-occurrence Matrix (GLCM).
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            dict: Dictionary containing texture features
        """
        # Convert to 8-bit image if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Calculate GLCM features
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(image, distances=distances, angles=angles,
                           levels=256, symmetric=True, normed=True)

        # Extract Haralick features
        features = {
            'contrast': graycoprops(glcm, 'contrast').mean(),
            'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
            'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
            'energy': graycoprops(glcm, 'energy').mean(),
            'correlation': graycoprops(glcm, 'correlation').mean(),
            'ASM': graycoprops(glcm, 'ASM').mean()
        }

        return features

    def extract_statistical_features(self, image):
        """
        Extract statistical features from the image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            dict: Dictionary containing statistical features
        """
        # First-order statistics
        features = {
            'mean': np.mean(image),
            'std': np.std(image),
            'variance': np.var(image),
            'skewness': np.mean(((image - np.mean(image))/np.std(image))**3),
            'kurtosis': np.mean(((image - np.mean(image))/np.std(image))**4) - 3,
            'entropy': -np.sum(np.histogram(image, bins=256, density=True)[0] *
                              np.log2(np.histogram(image, bins=256, density=True)[0] + 1e-10))
        }

        return features

    def extract_all_features(self, image_path, apply_noise_reduction=True):
        """
        Extract all features from an image.
        
        Args:
            image_path (str): Path to the image file
            apply_noise_reduction (bool): Whether to apply noise reduction
            
        Returns:
            dict: Dictionary containing all extracted features
        """
        # Load and preprocess image
        img = cv2.imread(image_path, 0)
        img = cv2.resize(img, (224, 224))

        if apply_noise_reduction:
            img = self.preprocess_image(img)

        # Extract features
        texture_feats = self.extract_texture_features(img)
        stat_feats = self.extract_statistical_features(img)

        # Combine all features
        features = {**texture_feats, **stat_feats}
        return features

    def analyze_features(self, n_samples=50):
        """
        Analyze feature distributions with and without noise reduction.
        
        Args:
            n_samples (int): Number of samples to analyze per class
        """
        features_data = []

        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_path = os.path.join(self.train_dir, class_name)
            image_files = os.listdir(class_path)[:n_samples]

            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)

                # Extract features with and without noise reduction
                features_with_nr = self.extract_all_features(img_path, True)
                features_without_nr = self.extract_all_features(img_path, False)

                # Add to data
                features_data.append({
                    'class': class_name,
                    'preprocessing': 'With Noise Reduction',
                    **features_with_nr
                })
                features_data.append({
                    'class': class_name,
                    'preprocessing': 'Without Noise Reduction',
                    **features_without_nr
                })

        # Convert to DataFrame
        df_features = pd.DataFrame(features_data)

        # Visualize distributions
        feature_names = [col for col in df_features.columns
                        if col not in ['class', 'preprocessing']]

        n_features = len(feature_names)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        plt.figure(figsize=(20, 5*n_rows))

        for idx, feature in enumerate(feature_names, 1):
            plt.subplot(n_rows, n_cols, idx)

            sns.boxplot(data=df_features, x='class', y=feature,
                       hue='preprocessing')
            plt.title(f'{feature} Distribution')
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('results/figures/feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print statistical analysis
        print("\nFeature Analysis:")
        for feature in feature_names:
            print(f"\n{feature}:")
            for prep in ['With Noise Reduction', 'Without Noise Reduction']:
                print(f"\n{prep}:")
                for class_name in ['NORMAL', 'PNEUMONIA']:
                    data = df_features[(df_features['class'] == class_name) &
                                     (df_features['preprocessing'] == prep)][feature]
                    print(f"{class_name}:")
                    print(f"  Mean: {data.mean():.4f}")
                    print(f"  Std:  {data.std():.4f}")

        return df_features

    def prepare_dataset(self, n_samples=100):
        """
        Prepare dataset for training.
        
        Args:
            n_samples (int): Number of samples per class
            
        Returns:
            tuple: (X, y, feature_names) where X is features, y is labels, feature_names is list of feature names
        """
        X = []
        y = []
        feature_names = []

        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_path = os.path.join(self.train_dir, class_name)
            image_files = os.listdir(class_path)[:n_samples]

            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                features = self.extract_all_features(img_path, True)

                if not feature_names:
                    feature_names = list(features.keys())

                X.append(list(features.values()))
                y.append(1 if class_name == 'PNEUMONIA' else 0)

        X = np.array(X)
        y = np.array(y)

        return X, y, feature_names

    def prepare_test_data(self):
        """
        Prepare test dataset.
        
        Returns:
            tuple: (test_X, test_y, test_paths) where test_X is features, test_y is labels, test_paths is image paths
        """
        test_X = []
        test_y = []
        test_paths = []

        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_path = os.path.join(self.test_dir, class_name)
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                features = self.extract_all_features(img_path, True)

                test_X.append(list(features.values()))
                test_y.append(1 if class_name == 'PNEUMONIA' else 0)
                test_paths.append(img_path)

        return np.array(test_X), np.array(test_y), test_paths

    def analyze_feature_importance(self, X, y, feature_names):
        """
        Analyze feature importance using PCA and Random Forest.
        
        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target labels
            feature_names (list): List of feature names
            
        Returns:
            tuple: (pca, importance_df) PCA object and feature importance DataFrame
        """
        from sklearn.ensemble import RandomForestClassifier
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        # Plot explained variance ratio
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA Analysis')
        plt.grid(True)
        plt.savefig('results/figures/pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Feature importance using Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)

        # Plot feature importance
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('results/figures/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\nTop 5 Most Important Features:")
        print(importance.head())

        return pca, importance


def main():
    """Main function to demonstrate the preprocessor."""
    # Initialize preprocessor
    preprocessor = ChestXRayPreprocessor()
    
    # Print dataset statistics
    stats = preprocessor.print_dataset_stats()
    
    # Visualize preprocessing effects
    preprocessor.visualize_preprocessing_effects()
    
    # Analyze features
    df_features = preprocessor.analyze_features()
    
    # Prepare dataset
    X, y, feature_names = preprocessor.prepare_dataset()
    
    # Analyze feature importance
    pca, importance = preprocessor.analyze_feature_importance(X, y, feature_names)
    
    print(f"\nDataset prepared with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"Feature names: {feature_names}")


if __name__ == "__main__":
    main()
