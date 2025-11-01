"""
Load PPG dataset from the Healthcare Hackathon Challenge 3.
This script will be adapted once we receive the actual PPG dataset format.
"""

import numpy as np
import pandas as pd
from data_preprocessing import preprocess_signal
from model import ArrhythmiaPredictor
import os


def load_ppg_dataset(data_path, sampling_rate=125):
    """
    Load PPG dataset from challenge.
    
    This function will be adapted based on the actual dataset format provided.
    Possible formats: CSV, JSON, HDF5, etc.
    
    Args:
        data_path: Path to PPG dataset file or directory
        sampling_rate: Sampling frequency in Hz (default: 125 Hz for PPG)
        
    Returns:
        signals: List of signal arrays
        labels: List of labels (0=normal, 1=arrhythmia) if available
        sampling_rates: List of sampling rates
    """
    signals = []
    labels = []
    sampling_rates = []
    
    # Check if path is a file or directory
    if os.path.isfile(data_path):
        # Single file - try to load based on extension
        if data_path.endswith('.csv'):
            try:
                df = pd.read_csv(data_path)
                # Assume first column is signal, last column might be label
                if 'signal' in df.columns:
                    signal_data = df['signal'].values
                elif 'ppg' in df.columns:
                    signal_data = df['ppg'].values
                else:
                    # Assume first column is signal
                    signal_data = df.iloc[:, 0].values
                
                # Check if labels are provided
                if 'label' in df.columns:
                    label = int(df['label'].iloc[0])
                elif 'arrhythmia' in df.columns:
                    label = int(df['arrhythmia'].iloc[0])
                else:
                    label = None
                
                # Segment into 30-second windows if long enough
                window_size = int(30 * sampling_rate)
                if len(signal_data) >= window_size:
                    num_windows = len(signal_data) // window_size
                    for i in range(num_windows):
                        start = i * window_size
                        end = start + window_size
                        signals.append(signal_data[start:end])
                        labels.append(label if label is not None else 0)
                        sampling_rates.append(sampling_rate)
                else:
                    # Single segment
                    signals.append(signal_data)
                    labels.append(label if label is not None else 0)
                    sampling_rates.append(sampling_rate)
                    
            except Exception as e:
                print(f"Error loading CSV: {e}")
                
        elif data_path.endswith('.json'):
            try:
                import json
                with open(data_path, 'r') as f:
                    data = json.load(f)
                    # Adapt based on JSON structure
                    # This will be updated based on actual format
                    pass
            except Exception as e:
                print(f"Error loading JSON: {e}")
        
        elif data_path.endswith('.npy'):
            # NumPy array file
            signal_data = np.load(data_path)
            window_size = int(30 * sampling_rate)
            if len(signal_data) >= window_size:
                num_windows = len(signal_data) // window_size
                for i in range(num_windows):
                    start = i * window_size
                    end = start + window_size
                    signals.append(signal_data[start:end])
                    labels.append(0)  # Unknown label
                    sampling_rates.append(sampling_rate)
            else:
                signals.append(signal_data)
                labels.append(0)
                sampling_rates.append(sampling_rate)
    
    elif os.path.isdir(data_path):
        # Directory - load all files
        for filename in os.listdir(data_path):
            if filename.endswith(('.csv', '.npy', '.txt')):
                file_path = os.path.join(data_path, filename)
                sigs, lbls, srs = load_ppg_dataset(file_path, sampling_rate)
                signals.extend(sigs)
                labels.extend(lbls)
                sampling_rates.extend(srs)
    
    return signals, labels, sampling_rates


def predict_from_ppg_dataset(model_path, ppg_data_path, sampling_rate=125):
    """
    Load PPG dataset and make predictions.
    
    Args:
        model_path: Path to trained model
        ppg_data_path: Path to PPG dataset
        sampling_rate: Sampling rate in Hz
        
    Returns:
        predictions: List of predictions (0=normal, 1=arrhythmia)
        probabilities: List of probability scores
    """
    # Load model
    predictor = ArrhythmiaPredictor()
    predictor.load(model_path)
    
    # Load PPG data
    signals, labels, sampling_rates = load_ppg_dataset(ppg_data_path, sampling_rate)
    
    if not signals:
        print("No signals loaded!")
        return None, None
    
    print(f"Loaded {len(signals)} signal segments")
    
    # Extract features and predict
    predictions = []
    probabilities = []
    
    for i, signal in enumerate(signals):
        try:
            sr = sampling_rates[i] if i < len(sampling_rates) else sampling_rate
            features = preprocess_signal(signal, sampling_rate=sr)
            features = features.reshape(1, -1)
            
            pred, prob = predictor.predict(features)
            predictions.append(pred[0])
            probabilities.append(prob[0])
            
        except Exception as e:
            print(f"Error processing signal {i}: {e}")
            predictions.append(-1)  # Error marker
            probabilities.append(0.0)
    
    return predictions, probabilities


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Load PPG dataset and predict')
    parser.add_argument('--model', type=str, default='model.pkl',
                       help='Path to trained model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to PPG dataset')
    parser.add_argument('--rate', type=int, default=125,
                       help='Sampling rate in Hz')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PPG Dataset Prediction")
    print("=" * 60)
    
    predictions, probabilities = predict_from_ppg_dataset(
        args.model, args.data, args.rate
    )
    
    if predictions:
        print(f"\nPredictions: {len(predictions)} total")
        print(f"Arrhythmia detected: {sum(p == 1 for p in predictions)}")
        print(f"Normal: {sum(p == 0 for p in predictions)}")
        
        # Show first few predictions
        print("\nFirst 10 predictions:")
        for i, (pred, prob) in enumerate(zip(predictions[:10], probabilities[:10])):
            status = "ARRHYTHMIA" if pred == 1 else "NORMAL" if pred == 0 else "ERROR"
            print(f"  {i+1}. {status} (confidence: {prob:.2%})")

