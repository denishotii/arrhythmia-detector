# Test Data Files

This directory contains test CSV files extracted from the MIT-BIH Arrhythmia Database for testing the arrhythmia detection model.

## Available Test Files

### 30-Second Segments (Recommended for Testing)

1. **test_data_normal_30s.csv**
   - **Source**: MIT-BIH Record 100 (segment at 60 seconds)
   - **Duration**: 30 seconds
   - **Sampling Rate**: 360 Hz
   - **Samples**: 10,800
   - **Expected**: Normal rhythm (mostly N beats)
   - **Size**: ~105 KB

2. **test_data_arrhythmia_30s.csv**
   - **Source**: MIT-BIH Record 203 (first 30 seconds)
   - **Duration**: 30 seconds
   - **Sampling Rate**: 360 Hz
   - **Samples**: 10,800
   - **Expected**: Arrhythmia (V - Ventricular beats present)
   - **Size**: ~103 KB

### 10-Second Segments (Shorter Test Cases)

3. **test_data_normal_10s.csv**
   - **Source**: MIT-BIH Record 100 (first 10 seconds)
   - **Duration**: 10 seconds
   - **Sampling Rate**: 360 Hz
   - **Samples**: 3,600
   - **Note**: May contain some arrhythmia (A beats)

4. **test_data_arrhythmia_10s.csv**
   - **Source**: MIT-BIH Record 203 (first 10 seconds)
   - **Duration**: 10 seconds
   - **Sampling Rate**: 360 Hz
   - **Samples**: 3,600
   - **Note**: Contains ventricular beats (V)

## Usage

### Basic Prediction

```bash
# Test with normal data
python predict.py --model model.pkl --signal test_data_normal_30s.csv --rate 360

# Test with arrhythmia data
python predict.py --model model.pkl --signal test_data_arrhythmia_30s.csv --rate 360
```

### Explainable AI Analysis

```bash
# Explain normal prediction
python explain_predictions.py --model model.pkl --signal test_data_normal_30s.csv --rate 360 --top 10

# Explain arrhythmia prediction
python explain_predictions.py --model model.pkl --signal test_data_arrhythmia_30s.csv --rate 360 --top 10 --save explanation.png
```

### From Python

```python
from predict import predict_from_signal
import numpy as np

# Load test data
signal = np.loadtxt('test_data_normal_30s.csv', delimiter=',')

# Predict
prediction, probability = predict_from_signal('model.pkl', signal, sampling_rate=360)
print(f"Prediction: {'ARRHYTHMIA' if prediction == 1 else 'NORMAL'}")
print(f"Confidence: {probability:.2%}")
```

## File Format

All test files are **CSV format** with:
- Single column of ECG signal values
- Values separated by commas
- One value per line
- Floating-point format (precision: 6 decimals)

Example:
```csv
0.123456
0.124567
0.125678
...
```

## Creating More Test Files

To create additional test files from MIT-BIH data:

```bash
python create_test_data.py
```

Or manually:

```python
import wfdb
import numpy as np

# Load record
record = wfdb.rdrecord('data/physionet.org/files/mitdb/1.0.0/100', channels=[0])

# Extract signal (first 30 seconds = 10800 samples at 360 Hz)
signal = record.p_signal[:, 0][:10800]

# Save to CSV
np.savetxt('my_test_data.csv', signal, delimiter=',', fmt='%.6f')
```

## Notes

- **Sampling Rate**: All files use 360 Hz (MIT-BIH standard)
- **Duration**: 30-second files recommended for best accuracy (minimum 30s for HRV features)
- **Source Data**: Real clinical ECG data from MIT-BIH Arrhythmia Database
- **Labels**: Based on MIT-BIH annotations, but actual model predictions may vary

## Expected Results

When testing with these files, you should see:

- **test_data_normal_30s.csv**: Likely predicts "NORMAL" with confidence > 50%
- **test_data_arrhythmia_30s.csv**: Likely predicts "ARRHYTHMIA" with confidence > 70%

*Note: Actual predictions depend on model training and specific signal segments.*

