"""
Arrhythmia detection model using machine learning.
Supports multiple model types: Random Forest, SVM, and Neural Network.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib


class ArrhythmiaPredictor:
    """
    Arrhythmia detection predictor using machine learning.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model to use ('random_forest', 'svm', or 'neural_network')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'mean_rr', 'std_rr', 'rmssd', 'pnn50', 'lf_power', 'hf_power',
            'lf_hf_ratio', 'rr_irregularity', 'rr_variance', 'cv',
            'mean', 'std', 'var', 'skewness', 'kurtosis', 'min', 'max',
            'range', 'num_beats', 'duration'
        ]
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == 'neural_network':
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X, y, test_size=0.2):
        """
        Train the model on provided data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=normal, 1=arrhythmia)
            test_size: Fraction of data to use for testing
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        print("\n=== Model Performance ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
        print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def predict(self, X):
        """
        Predict arrhythmia from features.
        
        Args:
            X: Feature vector or matrix
            
        Returns:
            predictions: Predicted class (0=normal, 1=arrhythmia)
            probabilities: Probability of arrhythmia
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]  # Probability of class 1 (arrhythmia)
        
        return predictions, probabilities
    
    def save(self, filepath):
        """Save the trained model."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load a trained model."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.model_type = data['model_type']
        print(f"Model loaded from {filepath}")
    
    def get_explainer(self):
        """
        Get explainable AI explainer for this model.
        
        Returns:
            ModelExplainer: XAI explainer instance
        """
        from explainability import ModelExplainer
        return ModelExplainer(self.model, self.scaler, self.feature_names)

