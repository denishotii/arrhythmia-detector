"""
Explainable AI (XAI) for Arrhythmia Detection Model
Provides interpretability and feature importance analysis using SHAP.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Install with: pip install shap")


class ModelExplainer:
    """
    Explainable AI wrapper for arrhythmia detection model.
    Provides feature importance and prediction explanations.
    """
    
    def __init__(self, model, scaler, feature_names=None):
        """
        Initialize explainer.
        
        Args:
            model: Trained ML model (Random Forest, SVM, or Neural Network)
            scaler: StandardScaler used for preprocessing
            feature_names: List of feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library not installed. Install with: pip install shap")
        
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names or self._get_default_feature_names()
        
        # Initialize SHAP explainer based on model type
        self.explainer = None
        self.explainer_type = None
        self._initialize_explainer()
    
    def _get_default_feature_names(self):
        """Get default feature names."""
        return [
            'mean_rr', 'std_rr', 'rmssd', 'pnn50', 'lf_power', 'hf_power',
            'lf_hf_ratio', 'rr_irregularity', 'rr_variance', 'cv',
            'mean', 'std', 'var', 'skewness', 'kurtosis', 'min', 'max',
            'range', 'num_beats', 'duration'
        ]
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer for model type."""
        model_type = type(self.model).__name__
        
        if 'RandomForest' in model_type:
            # TreeExplainer is fast and exact for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
            self.explainer_type = 'tree'
        elif 'SVC' in model_type or 'SVM' in model_type:
            # KernelExplainer for SVM (slower but works)
            # Use LinearExplainer if kernel is linear
            if hasattr(self.model, 'kernel') and self.model.kernel == 'linear':
                self.explainer = shap.LinearExplainer(self.model, self.scaler.transform([[0]*len(self.feature_names)]))
            else:
                # Use sample for background
                background = self.scaler.transform([[0]*len(self.feature_names)] * 100)
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            self.explainer_type = 'kernel'
        elif 'MLP' in model_type or 'NeuralNetwork' in model_type:
            # DeepExplainer or KernelExplainer for neural networks
            background = self.scaler.transform([[0]*len(self.feature_names)] * 100)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            self.explainer_type = 'kernel'
        else:
            # Default: KernelExplainer
            background = self.scaler.transform([[0]*len(self.feature_names)] * 100)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            self.explainer_type = 'kernel'
    
    def explain_prediction(self, X, prediction_idx=0):
        """
        Explain a single prediction.
        
        Args:
            X: Feature vector (raw, will be scaled)
            prediction_idx: Index of prediction to explain (if multiple)
            
        Returns:
            shap_values: SHAP values for the prediction
            explanation_dict: Dictionary with feature contributions
        """
        # Scale features
        X_scaled = self.scaler.transform(X.reshape(1, -1) if X.ndim == 1 else X)
        
        # Get SHAP values
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X_scaled)
            # For binary classification, get values for class 1 (arrhythmia)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (arrhythmia)
        else:
            shap_values = self.explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (arrhythmia)
        
        # Get explanation for single prediction
        if shap_values.ndim > 1:
            shap_vals = shap_values[prediction_idx]
        else:
            shap_vals = shap_values
        
        # Create explanation dictionary
        explanation_dict = {
            'feature_names': self.feature_names,
            'shap_values': shap_vals.tolist() if hasattr(shap_vals, 'tolist') else shap_vals,
            'feature_values': X_scaled[prediction_idx] if X_scaled.ndim > 1 else X_scaled,
            'contributions': dict(zip(self.feature_names, shap_vals)),
            'top_features': sorted(
                zip(self.feature_names, shap_vals),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]  # Top 10 most important features
        }
        
        return shap_values, explanation_dict
    
    def plot_feature_importance(self, shap_values, save_path=None, show=True):
        """
        Plot SHAP feature importance.
        
        Args:
            shap_values: SHAP values from explain_prediction
            save_path: Path to save plot (optional)
            show: Whether to display plot
        """
        if shap_values.ndim > 1:
            shap.summary_plot(shap_values, feature_names=self.feature_names, show=False)
        else:
            shap.waterfall_plot(shap.Explanation(
                values=shap_values,
                base_values=self.explainer.expected_value[1] if hasattr(self.explainer.expected_value, '__len__') else self.explainer.expected_value,
                data=self.feature_names,
                feature_names=self.feature_names
            ), show=False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_waterfall(self, shap_values, prediction_idx=0, save_path=None, show=True):
        """
        Plot waterfall plot for single prediction.
        
        Args:
            shap_values: SHAP values from explain_prediction
            prediction_idx: Index of prediction
            save_path: Path to save plot
            show: Whether to display plot
        """
        # Get values for single prediction
        if shap_values.ndim > 1:
            values = shap_values[prediction_idx]
        else:
            values = shap_values
        
        # Get base value
        base_value = self.explainer.expected_value
        if hasattr(base_value, '__len__'):
            base_value = base_value[1]  # Class 1 (arrhythmia)
        
        # Create explanation object
        exp = shap.Explanation(
            values=values,
            base_values=base_value,
            data=self.feature_names,
            feature_names=self.feature_names
        )
        
        shap.waterfall_plot(exp, show=False)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Waterfall plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_feature_importance_ranking(self, X_sample=None, top_n=10):
        """
        Get feature importance ranking using SHAP values.
        
        Args:
            X_sample: Sample of data to compute importance on (optional)
            top_n: Number of top features to return
            
        Returns:
            importance_df: DataFrame with feature importance
        """
        if X_sample is None:
            # Use dummy data for importance calculation
            X_sample = np.zeros((100, len(self.feature_names)))
        
        # Scale features
        X_scaled = self.scaler.transform(X_sample)
        
        # Get SHAP values
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (arrhythmia)
        else:
            shap_values = self.explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        
        # Compute mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0) if shap_values.ndim > 1 else np.abs(shap_values)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap,
            'abs_importance': np.abs(mean_shap)
        }).sort_values('abs_importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def explain_batch(self, X, top_features=5):
        """
        Explain predictions for a batch of samples.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            top_features: Number of top features to highlight
            
        Returns:
            explanations: List of explanation dictionaries
        """
        X_scaled = self.scaler.transform(X)
        
        # Get SHAP values
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            shap_values = self.explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        
        explanations = []
        for i in range(len(X_scaled)):
            shap_vals = shap_values[i] if shap_values.ndim > 1 else shap_values
            
            # Get top contributing features
            feature_contributions = list(zip(self.feature_names, shap_vals))
            top_pos = sorted([(f, v) for f, v in feature_contributions if v > 0], 
                           key=lambda x: x[1], reverse=True)[:top_features]
            top_neg = sorted([(f, v) for f, v in feature_contributions if v < 0], 
                           key=lambda x: x[1])[:top_features]
            
            explanations.append({
                'sample_idx': i,
                'top_positive_features': top_pos,  # Features pushing towards arrhythmia
                'top_negative_features': top_neg,  # Features pushing towards normal
                'total_contribution': np.sum(shap_vals)
            })
        
        return explanations


def explain_prediction_from_model(model_path, signal_data, sampling_rate=125):
    """
    Quick function to explain a prediction from a saved model.
    
    Args:
        model_path: Path to saved model
        signal_data: Signal array (will be preprocessed)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        explanation: Explanation dictionary
    """
    from model import ArrhythmiaPredictor
    from data_preprocessing import preprocess_signal
    
    # Load model
    predictor = ArrhythmiaPredictor()
    predictor.load(model_path)
    
    # Preprocess signal
    features = preprocess_signal(signal_data, sampling_rate)
    
    # Create explainer
    explainer = ModelExplainer(
        predictor.model,
        predictor.scaler,
        predictor.feature_names
    )
    
    # Explain prediction
    shap_values, explanation = explainer.explain_prediction(features)
    
    return explanation, shap_values


if __name__ == '__main__':
    print("Explainable AI Module for Arrhythmia Detection")
    print("=" * 60)
    
    if not SHAP_AVAILABLE:
        print("\n⚠️ SHAP library not installed.")
        print("Install with: pip install shap")
    else:
        print("\n✅ SHAP library available")
        print("\nUsage:")
        print("  from explainability import ModelExplainer, explain_prediction_from_model")
        print("  explanation, shap_vals = explain_prediction_from_model('model.pkl', signal, 360)")

