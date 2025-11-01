"""
Quick test script to verify MIT-BIH data loading works.
"""

import sys
try:
    import wfdb
    print("✓ wfdb installed")
except ImportError:
    print("✗ wfdb not installed. Run: pip install wfdb")
    sys.exit(1)

from train import load_mit_bih_dataset

# Test loading a single record
print("\nTesting MIT-BIH data loading...")
print("=" * 50)

record_ids = [100]  # Test with record 100
data_dir = 'data/physionet.org/files/mitdb/1.0.0/'

signals, labels, sampling_rates = load_mit_bih_dataset(record_ids, data_dir=data_dir)

if signals:
    print(f"\n✓ Successfully loaded {len(signals)} windows")
    print(f"  Signals: {len(signals)}")
    print(f"  Labels: {sum(labels)} arrhythmia, {len(labels)-sum(labels)} normal")
    print(f"  Sampling rates: {set(sampling_rates)} Hz")
    print(f"  Signal shape (first): {signals[0].shape}")
    print(f"  Signal duration: {signals[0].shape[0] / sampling_rates[0]:.1f} seconds")
    print("\n✓ Ready to train!")
else:
    print("\n✗ Failed to load data")

