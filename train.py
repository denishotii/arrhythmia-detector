"""
Training script for arrhythmia detection model.
Loads data, extracts features, and trains the model.
"""

import numpy as np
import pandas as pd
from data_preprocessing import preprocess_signal
from model import ArrhythmiaPredictor
import argparse


def load_mit_bih_dataset(record_ids, data_dir='data/physionet.org/files/mitdb/1.0.0/'):
    """
    Load MIT-BIH Arrhythmia Database records.
    Requires wfdb package and downloaded dataset.
    
    Args:
        record_ids: List of record IDs to load (e.g., [100, 101, 102])
        data_dir: Directory containing MIT-BIH dataset
        
    Returns:
        signals: List of signal arrays
        labels: List of labels (0=normal, 1=arrhythmia)
        sampling_rates: List of sampling rates for each signal
    """
    try:
        import wfdb
    except ImportError:
        print("wfdb package not installed. Install with: pip install wfdb")
        return None, None, None
    
    signals = []
    labels = []
    sampling_rates = []
    
    # MIT-BIH beat annotations: 
    # N = Normal, L = Left bundle branch block, R = Right bundle branch block
    # A = Atrial premature, V = Ventricular, F = Fusion, J = Nodal (junctional)
    # E = Ventricular escape, / = Paced, etc.
    arrhythmia_annotations = ['A', 'V', 'F', 'J', 'E', 'a', 'j', 'S']  # Arrhythmia types
    normal_annotations = ['N', 'L', 'R']  # Normal beats
    
    for record_id in record_ids:
        try:
            # Load record and annotations
            record = wfdb.rdrecord(f'{data_dir}/{record_id}', channels=[0])  # First channel only
            ann = wfdb.rdann(f'{data_dir}/{record_id}', 'atr')
            
            # Extract signal (first channel, typically MLII)
            signal_data = record.p_signal[:, 0]
            sampling_rate = record.fs
            
            # Segment signal into windows (30-second windows)
            window_size = int(30 * sampling_rate)  # 30 seconds
            num_windows = len(signal_data) // window_size
            
            print(f"  Processing record {record_id}: {num_windows} windows, {len(signal_data)/sampling_rate:.1f}s total")
            
            for i in range(num_windows):
                start = i * window_size
                end = start + window_size
                segment = signal_data[start:end]
                
                # Get annotations in this window
                window_ann_indices = [idx for idx in range(len(ann.sample))
                                     if start <= ann.sample[idx] < end]
                window_symbols = [ann.symbol[idx] for idx in window_ann_indices]
                
                # Count normal vs arrhythmia beats
                normal_count = sum(1 for sym in window_symbols if sym in normal_annotations)
                arrhythmia_count = sum(1 for sym in window_symbols if sym in arrhythmia_annotations)
                total_beats = len(window_symbols)
                
                # Label: 1 if majority of beats are arrhythmic, or if any arrhythmia present
                if total_beats > 0:
                    has_arrhythmia = (arrhythmia_count > 0) or (arrhythmia_count / total_beats > 0.2)
                else:
                    # No beats annotated, skip this window
                    continue
                
                signals.append(segment)
                labels.append(1 if has_arrhythmia else 0)
                sampling_rates.append(sampling_rate)
                
        except Exception as e:
            print(f"  Error loading record {record_id}: {e}")
            continue
    
    return signals, labels, sampling_rates


def generate_synthetic_data(n_samples=1000, sampling_rate=125):
    """
    Generate synthetic PPG/heart rate data for testing.
    Creates normal and arrhythmic patterns.
    
    Args:
        n_samples: Number of samples to generate
        sampling_rate: Sampling frequency in Hz
        
    Returns:
        signals: List of signal arrays
        labels: List of labels (0=normal, 1=arrhythmia)
    """
    signals = []
    labels = []
    
    for _ in range(n_samples):
        duration = 30  # 30 seconds
        t = np.linspace(0, duration, duration * sampling_rate)
        
        # Random choice: normal or arrhythmia
        is_arrhythmia = np.random.choice([0, 1], p=[0.7, 0.3])
        
        if is_arrhythmia:
            # Arrhythmic pattern: irregular heart rate
            base_rate = 70 + np.random.uniform(-10, 10)
            rate_variation = np.random.uniform(20, 50) * np.sin(2 * np.pi * np.random.uniform(0.1, 0.5) * t)
            heart_rate = base_rate + rate_variation
            
            # Add sudden spikes (ventricular beats)
            spikes = np.random.poisson(3, len(t)) * np.random.choice([-1, 0, 1], len(t))
            heart_rate += spikes * 30
            
            # Simulate PPG signal from heart rate
            signal_data = np.sin(2 * np.pi * (heart_rate / 60) * t) + 0.1 * np.random.randn(len(t))
            
            labels.append(1)
        else:
            # Normal pattern: regular heart rate with slight variation
            base_rate = 70 + np.random.uniform(-5, 5)
            rate_variation = np.random.uniform(2, 5) * np.sin(2 * np.pi * 0.1 * t)
            heart_rate = base_rate + rate_variation
            
            # Simulate PPG signal
            signal_data = np.sin(2 * np.pi * (heart_rate / 60) * t) + 0.05 * np.random.randn(len(t))
            
            labels.append(0)
        
        signals.append(signal_data)
    
    return signals, labels


def main():
    parser = argparse.ArgumentParser(description='Train arrhythmia detection model')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['random_forest', 'svm', 'neural_network'],
                       help='Model type to use')
    parser.add_argument('--data', type=str, default='synthetic',
                       choices=['synthetic', 'mit-bih'],
                       help='Data source to use')
    parser.add_argument('--save', type=str, default='model.pkl',
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Arrhythmia Detection Model Training")
    print("=" * 50)
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    sampling_rates = None
    if args.data == 'synthetic':
        signals, labels = generate_synthetic_data(n_samples=1000)
        sampling_rates = [125] * len(signals)  # Synthetic data uses 125 Hz
    elif args.data == 'mit-bih':
        # Load MIT-BIH records (use subset for faster training)
        # Records 100-124 and 200-234 are available
        record_ids = list(range(100, 125)) + list(range(200, 235))
        # For quick testing, use first 5 records from each group
        record_ids = list(range(100, 105)) + list(range(200, 205))
        
        print(f"Loading records: {record_ids}")
        signals, labels, sampling_rates = load_mit_bih_dataset(record_ids)
        if signals is None or len(signals) == 0:
            print("Failed to load MIT-BIH data. Falling back to synthetic data.")
            signals, labels = generate_synthetic_data(n_samples=1000)
            sampling_rates = [125] * len(signals)
    
    print(f"Loaded {len(signals)} samples ({sum(labels)} arrhythmia, {len(labels)-sum(labels)} normal)")
    
    # Extract features
    print("\nExtracting features...")
    features = []
    valid_labels = []
    
    # Default sampling rate (for synthetic data)
    default_sr = 125
    
    for i, signal in enumerate(signals):
        try:
            # Use appropriate sampling rate (MIT-BIH uses 360 Hz, synthetic uses 125 Hz)
            sr = sampling_rates[i] if sampling_rates and i < len(sampling_rates) else default_sr
            feature_vector = preprocess_signal(signal, sampling_rate=sr)
            features.append(feature_vector)
            valid_labels.append(labels[i])
        except Exception as e:
            print(f"Error processing signal {i}: {e}")
            continue
    
    X = np.array(features)
    y = np.array(valid_labels)
    
    print(f"Extracted features: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Train model
    predictor = ArrhythmiaPredictor(model_type=args.model)
    results = predictor.train(X, y)
    
    # Save model
    predictor.save(args.save)
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()

