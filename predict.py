"""
Prediction script: Use trained model to predict arrhythmia from new data.
"""

import numpy as np
from data_preprocessing import preprocess_signal
from model import ArrhythmiaPredictor
import argparse
import sys


def predict_from_signal(model_path, signal_data, sampling_rate=125):
    """
    Predict arrhythmia from a signal array.
    
    Args:
        model_path: Path to trained model
        signal_data: 1D array of signal values
        sampling_rate: Sampling frequency in Hz
        
    Returns:
        prediction: 0 (normal) or 1 (arrhythmia)
        probability: Probability of arrhythmia (0-1)
    """
    # Load model
    predictor = ArrhythmiaPredictor()
    predictor.load(model_path)
    
    # Extract features
    feature_vector = preprocess_signal(signal_data, sampling_rate)
    feature_vector = feature_vector.reshape(1, -1)  # Reshape for single prediction
    
    # Predict
    prediction, probability = predictor.predict(feature_vector)
    
    return prediction[0], probability[0]


def main():
    parser = argparse.ArgumentParser(description='Predict arrhythmia from signal data')
    parser.add_argument('--model', type=str, default='model.pkl',
                       help='Path to trained model')
    parser.add_argument('--signal', type=str,
                       help='Path to signal data file (CSV or numpy array)')
    parser.add_argument('--rate', type=int, default=125,
                       help='Sampling rate in Hz')
    
    args = parser.parse_args()
    
    # Load signal data
    if args.signal:
        try:
            if args.signal.endswith('.csv'):
                signal_data = np.loadtxt(args.signal, delimiter=',')
            else:
                signal_data = np.load(args.signal)
        except Exception as e:
            print(f"Error loading signal data: {e}")
            sys.exit(1)
    else:
        print("No signal data provided. Use --signal to specify a file.")
        sys.exit(1)
    
    # Predict
    print("Running prediction...")
    prediction, probability = predict_from_signal(args.model, signal_data, args.rate)
    
    print("\n" + "=" * 50)
    print("Prediction Results:")
    print("=" * 50)
    print(f"Prediction: {'ARRHYTHMIA DETECTED' if prediction == 1 else 'NORMAL'}")
    print(f"Confidence: {probability:.2%}")
    print("=" * 50)


if __name__ == '__main__':
    main()

