"""
PPG and heart rate data preprocessing for arrhythmia detection.
Extracts features from heart rate signals.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import stats
import neurokit2 as nk


def extract_rr_intervals(signal_data, sampling_rate=125):
    """
    Extract R-R intervals (time between heartbeats) from PPG/ECG signal.
    
    Args:
        signal_data: 1D array of signal values
        sampling_rate: Sampling frequency in Hz
        
    Returns:
        rr_intervals: Array of R-R intervals in milliseconds
    """
    # Detect peaks (heartbeats) using NeuroKit2
    peaks, info = nk.ecg_peaks(signal_data, sampling_rate=sampling_rate)
    rr_intervals = np.diff(info['ECG_R_Peaks']) / sampling_rate * 1000  # Convert to ms
    
    return rr_intervals


def extract_hrv_features(rr_intervals):
    """
    Extract Heart Rate Variability (HRV) features from R-R intervals.
    These features are strong indicators of arrhythmia.
    
    Args:
        rr_intervals: Array of R-R intervals in milliseconds
        
    Returns:
        features_dict: Dictionary of HRV features
    """
    if len(rr_intervals) < 10:
        return None
    
    features = {}
    
    # Time-domain features
    features['mean_rr'] = np.mean(rr_intervals)
    features['std_rr'] = np.std(rr_intervals)
    features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2))  # Root Mean Square of Successive Differences
    features['nn50'] = np.sum(np.abs(np.diff(rr_intervals)) > 50)  # Number of intervals > 50ms
    features['pnn50'] = (features['nn50'] / len(rr_intervals)) * 100  # Percentage of NN50
    
    # Frequency-domain features (using simple FFT)
    if len(rr_intervals) >= 64:
        fft_values = np.fft.rfft(rr_intervals)
        power = np.abs(fft_values) ** 2
        freqs = np.fft.rfftfreq(len(rr_intervals))
        
        # Low frequency (0.04-0.15 Hz) and High frequency (0.15-0.4 Hz) power
        lf_indices = (freqs >= 0.04) & (freqs < 0.15)
        hf_indices = (freqs >= 0.15) & (freqs <= 0.4)
        
        features['lf_power'] = np.sum(power[lf_indices])
        features['hf_power'] = np.sum(power[hf_indices])
        features['lf_hf_ratio'] = features['lf_power'] / (features['hf_power'] + 1e-10)
    
    # Irregularity metrics (key for arrhythmia)
    features['rr_irregularity'] = np.std(np.diff(rr_intervals))
    features['rr_variance'] = np.var(rr_intervals)
    features['cv'] = features['std_rr'] / (features['mean_rr'] + 1e-10)  # Coefficient of variation
    
    return features


def extract_statistical_features(signal_data, window_size=1000):
    """
    Extract statistical features from raw signal.
    
    Args:
        signal_data: 1D array of signal values
        window_size: Window size for feature extraction
        
    Returns:
        features_dict: Dictionary of statistical features
    """
    features = {}
    
    features['mean'] = np.mean(signal_data)
    features['std'] = np.std(signal_data)
    features['var'] = np.var(signal_data)
    features['skewness'] = stats.skew(signal_data)
    features['kurtosis'] = stats.kurtosis(signal_data)
    features['min'] = np.min(signal_data)
    features['max'] = np.max(signal_data)
    features['range'] = features['max'] - features['min']
    
    return features


def preprocess_signal(signal_data, sampling_rate=125):
    """
    Main preprocessing function: extracts all features from signal.
    
    Args:
        signal_data: 1D array of signal values
        sampling_rate: Sampling frequency in Hz
        
    Returns:
        feature_vector: Combined feature vector for ML model
    """
    # Extract R-R intervals
    rr_intervals = extract_rr_intervals(signal_data, sampling_rate)
    
    if rr_intervals is None or len(rr_intervals) < 5:
        # Return zero features if signal is too short
        return np.zeros(20)  # Adjust size based on total features
    
    # Extract HRV features
    hrv_features = extract_hrv_features(rr_intervals)
    
    # Extract statistical features
    stat_features = extract_statistical_features(signal_data)
    
    # Combine all features
    feature_vector = np.array([
        hrv_features.get('mean_rr', 0),
        hrv_features.get('std_rr', 0),
        hrv_features.get('rmssd', 0),
        hrv_features.get('pnn50', 0),
        hrv_features.get('lf_power', 0),
        hrv_features.get('hf_power', 0),
        hrv_features.get('lf_hf_ratio', 0),
        hrv_features.get('rr_irregularity', 0),
        hrv_features.get('rr_variance', 0),
        hrv_features.get('cv', 0),
        stat_features.get('mean', 0),
        stat_features.get('std', 0),
        stat_features.get('var', 0),
        stat_features.get('skewness', 0),
        stat_features.get('kurtosis', 0),
        stat_features.get('min', 0),
        stat_features.get('max', 0),
        stat_features.get('range', 0),
        len(rr_intervals),  # Number of heartbeats
        len(signal_data) / sampling_rate  # Duration in seconds
    ])
    
    return feature_vector

