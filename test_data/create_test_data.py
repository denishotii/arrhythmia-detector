"""
Create test CSV files from MIT-BIH dataset for testing arrhythmia detection.
"""

import wfdb
import numpy as np
import os


def create_test_csv(record_id, output_name, duration_seconds=30, data_dir='data/physionet.org/files/mitdb/1.0.0/'):
    """
    Extract ECG signal from MIT-BIH record and save as CSV.
    
    Args:
        record_id: MIT-BIH record ID (e.g., 100, 101, etc.)
        output_name: Output CSV filename
        duration_seconds: Duration of signal to extract (seconds)
        data_dir: Directory containing MIT-BIH data
    """
    try:
        # Load record
        record = wfdb.rdrecord(f'{data_dir}/{record_id}', channels=[0])
        ann = wfdb.rdann(f'{data_dir}/{record_id}', 'atr')
        
        # Extract signal
        signal_data = record.p_signal[:, 0]
        sampling_rate = record.fs
        
        # Extract duration
        num_samples = int(duration_seconds * sampling_rate)
        max_samples = len(signal_data)
        
        if num_samples > max_samples:
            # If requested duration is longer than available, use what we have
            num_samples = max_samples
            actual_duration = num_samples / sampling_rate
            print(f"   ⚠️  Warning: Only {actual_duration:.1f}s available (requested {duration_seconds}s)")
        
        # Extract segment (first N seconds)
        signal_segment = signal_data[:num_samples]
        
        # Get annotations in this segment
        segment_annotations = [
            ann.symbol[idx] for idx in range(len(ann.sample))
            if ann.sample[idx] < num_samples
        ]
        
        # Check for arrhythmia
        arrhythmia_types = ['A', 'V', 'F', 'J', 'E', 'a', 'j', 'S']
        has_arrhythmia = any(ann_symbol in arrhythmia_types for ann_symbol in segment_annotations)
        
        # Save to CSV
        np.savetxt(output_name, signal_segment, delimiter=',', fmt='%.6f')
        
        print(f"✅ Created {output_name}")
        print(f"   Record: {record_id}")
        print(f"   Samples: {len(signal_segment)} ({duration_seconds}s at {sampling_rate} Hz)")
        print(f"   Arrhythmia detected: {'Yes' if has_arrhythmia else 'No'}")
        print(f"   Annotations: {len(segment_annotations)} beats")
        if segment_annotations:
            unique_annotations = list(set(segment_annotations))
            print(f"   Beat types: {', '.join(unique_annotations)}")
        
        return True, has_arrhythmia
        
    except Exception as e:
        print(f"❌ Error processing record {record_id}: {e}")
        return False, None


def main():
    """Create test CSV files from MIT-BIH dataset."""
    print("=" * 60)
    print("Creating Test CSV Files from MIT-BIH Dataset")
    print("=" * 60)
    
    data_dir = 'data/physionet.org/files/mitdb/1.0.0/'
    
    # Test cases
    test_cases = [
        # (record_id, filename, expected_label, description)
        (100, 'test_data_normal.csv', 'normal', 'Normal rhythm (first 30s)'),
        (201, 'test_data_arrhythmia.csv', 'arrhythmia', 'Arrhythmia present (first 30s)'),
        (100, 'test_data_normal_10s.csv', 'normal', 'Normal rhythm (first 10s)'),
        (203, 'test_data_arrhythmia_10s.csv', 'arrhythmia', 'Arrhythmia present (first 10s)'),
    ]
    
    print("\nExtracting test data...\n")
    
    results = []
    for record_id, filename, expected_label, description in test_cases:
        duration = 30 if '30s' in filename else 10
        
        print(f"📊 Processing: {description}")
        success, has_arrhythmia = create_test_csv(record_id, filename, duration, data_dir)
        
        if success:
            actual_label = 'arrhythmia' if has_arrhythmia else 'normal'
            match = "✅" if actual_label == expected_label else "⚠️"
            print(f"   {match} Expected: {expected_label}, Actual: {actual_label}")
            results.append((filename, actual_label, match == "✅"))
        print()
    
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    for filename, label, correct in results:
        status = "✅" if correct else "⚠️"
        print(f"{status} {filename:30s} - {label}")
    
    print("\n✅ Test files created! Use them with:")
    print("   python predict.py --model model.pkl --signal test_data_normal.csv --rate 360")
    print("   python explain_predictions.py --model model.pkl --signal test_data_normal.csv --rate 360")


if __name__ == '__main__':
    main()

