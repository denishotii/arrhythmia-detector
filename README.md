# Arrhythmia Detection Model

A machine learning model for predicting arrhythmias using ECG/PPG signals, designed for real-time driver health assessment in healthcare applications.

## 🎯 Project Overview

**Objective**: Develop a **camera-based, non-invasive** arrhythmia detection system that uses only video monitoring of the driver's face. The system extracts heart rate signals from video (remote photoplethysmography/rPPG) and predicts arrhythmia in real-time.

**Challenge Context**: This project is designed for the Healthcare Hackathon's Challenge 3: "Camera-based Driver Health Assessment" - **NO PHYSICAL SENSORS OR CABLES**. Only a video camera monitors the driver's face, from which we extract heartbeat data and detect arrhythmias to enhance road safety.

## 📊 Dataset: MIT-BIH Arrhythmia Database

### Dataset Details

- **Source**: MIT-BIH Arrhythmia Database from PhysioNet
- **Records**: 48 half-hour excerpts of two-channel ambulatory ECG recordings from 47 subjects
- **Sampling Rate**: 360 samples per second per channel
- **Resolution**: 11-bit resolution over a 10 mV range
- **Annotations**: Each record includes expert annotations for every heartbeat, classified by cardiologists
- **Download**: https://www.physionet.org/content/mitdb/1.0.0/

### Training Data

**Current Model Training:**
- **Records Used**: 9 records (100-104, 200-203)
- **Total Samples**: 540 thirty-second windows
- **Class Distribution**:
  - Normal: 319 samples (59%)
  - Arrhythmia: 221 samples (41%)
- **Window Size**: 30-second segments (10,800 samples at 360 Hz)
- **Total Duration**: ~4.5 hours of ECG data

**Available Records**:
- Group 1: Records 100-124 (23 records, random selection)
- Group 2: Records 200-234 (25 records, rare arrhythmias)

### Annotation Types

The model distinguishes between:
- **Normal Beats**: N (Normal), L (Left bundle branch block), R (Right bundle branch block)
- **Arrhythmic Beats**: A (Atrial premature), V (Ventricular), F (Fusion), J (Nodal/junctional), E (Ventricular escape), S (Supraventricular)

## 🧠 Model Architecture

### Available Models

1. **Random Forest** (Current Best Performance)
   - 100 estimators
   - Max depth: 15
   - Balanced class weights
   - **Accuracy**: 94.4%

2. **Support Vector Machine (SVM)**
   - RBF kernel
   - Probability estimation enabled
   - Balanced class weights

3. **Neural Network (MLP)**
   - Hidden layers: (64, 32)
   - ReLU activation
   - Adam optimizer

### Feature Extraction

The model extracts **20 features** from each signal segment:

**Heart Rate Variability (HRV) Features:**
- Mean R-R interval
- Standard deviation of R-R intervals
- RMSSD (Root Mean Square of Successive Differences)
- pNN50 (Percentage of intervals > 50ms)
- Low Frequency (LF) power
- High Frequency (HF) power
- LF/HF ratio
- R-R irregularity metrics
- Coefficient of variation

**Statistical Features:**
- Mean, std, variance
- Skewness, kurtosis
- Min, max, range
- Number of heartbeats
- Signal duration

## 📈 Model Performance

### Current Results (Random Forest)

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.4% |
| **Precision** | 93.2% |
| **Recall** | 93.2% |
| **F1-Score** | 93.2% |

**Confusion Matrix:**
```
Normal:       61 correct,  3 false positives
Arrhythmia:    3 false negatives, 41 correct
```

## 🔧 Setup & Installation

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required Packages:**
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `scikit-learn>=1.3.0`
- `scipy>=1.10.0`
- `matplotlib>=3.7.0`
- `wfdb>=4.1.0` (for MIT-BIH data loading)
- `neurokit2>=0.2.0` (for ECG/PPG processing)
- `opencv-python>=4.8.0` (for video processing and rPPG extraction)
- `shap>=0.42.0` (for explainable AI and feature importance)

### 3. Download MIT-BIH Dataset (Optional)

The dataset should be placed in:
```
data/physionet.org/files/mitdb/1.0.0/
```

## 🚀 Usage

### Training the Model

#### Train on MIT-BIH Data (Recommended)
```bash
python train.py --model random_forest --data mit-bih
```

#### Train on Synthetic Data (Testing)
```bash
python train.py --model random_forest --data synthetic
```

**Model Options:**
- `--model`: `random_forest`, `svm`, or `neural_network`
- `--data`: `mit-bih` or `synthetic`
- `--save`: Path to save trained model (default: `model.pkl`)

### Real-time Detection from Video (Primary Method)

```python
from real_time_detector import RealTimeArrhythmiaDetector
from video_rppg_extractor import extract_rppg_from_api_frames

# Initialize detector
detector = RealTimeArrhythmiaDetector('model.pkl', buffer_duration=30, fps=30)

# Add frames from Cloud API
while True:
    frame = api_client.get_next_frame()  # From challenge Cloud API
    detector.add_frame(frame)
    
    # Process every 2 seconds
    result = detector.process_and_predict()
    if result:
        print(f"Status: {result['status']}, Confidence: {result['confidence']:.2%}")
        
        # Check for alerts
        alert, details = detector.check_alert()
        if alert:
            print(f"⚠️ ALERT: {details['message']}")
```

### Making Predictions from Signal Files

#### From Command Line

```bash
# From signal file
python predict.py --model model.pkl --signal your_signal.csv --rate 360

# From video file (extract rPPG first)
python -c "
from video_rppg_extractor import extract_rppg_from_video_file
from predict import predict_from_signal
rppg, fps, _ = extract_rppg_from_video_file('video.mp4')
pred, prob = predict_from_signal('model.pkl', rppg, fps)
print(f'Prediction: {pred}, Confidence: {prob:.2%}')
"
```

**Parameters:**
- `--model`: Path to trained model file
- `--signal`: Path to signal data (CSV or numpy array)
- `--rate`: Sampling rate in Hz (360 for MIT-BIH, 30 for video rPPG, 125 for synthetic/PPG)

#### From Python Code

```python
from data_preprocessing import preprocess_signal
from model import ArrhythmiaPredictor
import numpy as np

# Load model
predictor = ArrhythmiaPredictor(model_type='random_forest')
predictor.load('model.pkl')

# Process new signal
signal = np.array([...])  # Your ECG/PPG signal data
features = preprocess_signal(signal, sampling_rate=360)
prediction, probability = predictor.predict(features.reshape(1, -1))

# Output
print(f"Prediction: {'ARRHYTHMIA DETECTED' if prediction == 1 else 'NORMAL'}")
print(f"Confidence: {probability:.2%}")
```

## 📥 Input Requirements for Prediction

### Expected Input Format

**Signal Data:**
- **Type**: 1D numpy array or CSV file
- **Duration**: Minimum ~30 seconds recommended (for reliable HRV features)
- **Sampling Rate**: 
  - **ECG (MIT-BIH)**: 360 Hz
  - **PPG/Wearable**: 125 Hz (or as specified)
- **Values**: Raw signal amplitude values

### Input Examples

**From NumPy Array:**
```python
import numpy as np
signal = np.loadtxt('ecg_signal.csv')  # Load from file
# or
signal = np.array([...])  # Your signal data
```

**From CSV File:**
```bash
python predict.py --model model.pkl --signal ecg_data.csv --rate 360
```

### What the Model Predicts

- **Output**: Binary classification
  - `0` = Normal heartbeat pattern
  - `1` = Arrhythmia detected
- **Confidence**: Probability score (0-1) indicating likelihood of arrhythmia

## 📁 Project Structure

```
arrhythmia-detector/
├── data_preprocessing.py       # Feature extraction pipeline
├── model.py                    # ML models implementation
├── train.py                    # Training script
├── predict.py                  # Prediction script
├── video_rppg_extractor.py     # Extract rPPG from video frames ⭐
├── real_time_detector.py       # Real-time video processing ⭐
├── explainability.py            # Explainable AI (SHAP) ⭐
├── explain_predictions.py      # XAI command-line tool ⭐
├── load_ppg_dataset.py         # PPG dataset loader (if needed)
├── test_mitbih.py              # Test script for MIT-BIH loading
├── requirements.txt            # Python dependencies
├── model.pkl                   # Trained model (after training)
├── README.md                   # This file
├── DATA_REQUIREMENTS.md        # Data compatibility analysis
├── venv/                       # Virtual environment (gitignored)
└── data/
    └── physionet.org/
        └── files/
            └── mitdb/
                └── 1.0.0/      # MIT-BIH dataset (gitignored)
```

⭐ **New modules for camera-based detection**

## 🔬 Data Preprocessing Pipeline

1. **Signal Loading**: Read ECG/PPG/rPPG signal from file, array, or video
2. **R-R Interval Extraction**: Detect heartbeats and calculate intervals
3. **HRV Feature Calculation**: Compute heart rate variability metrics
4. **Statistical Feature Extraction**: Calculate signal statistics
5. **Feature Normalization**: Standard scaling for ML input
6. **Classification**: Binary prediction (normal vs arrhythmia)

## 🧠 Explainable AI (XAI)

The project includes **SHAP-based explainability** to understand model decisions:

### Features

- **Feature Importance**: Visualize which features drive predictions
- **Prediction Explanations**: Understand why a specific prediction was made
- **Feature Contributions**: See how each feature pushes towards normal or arrhythmia
- **Visualizations**: Waterfall plots and feature importance charts

### Usage

```bash
# Explain a prediction
python explain_predictions.py --model model.pkl --signal data.csv --rate 360 --save explanation.png
```

```python
# From Python code
from explainability import ModelExplainer
from model import ArrhythmiaPredictor

# Load model and get explainer
predictor = ArrhythmiaPredictor()
predictor.load('model.pkl')
explainer = predictor.get_explainer()

# Explain a prediction
features = preprocess_signal(signal_data, sampling_rate=360)
shap_values, explanation = explainer.explain_prediction(features)

# Visualize
explainer.plot_waterfall(shap_values)
explainer.plot_feature_importance(shap_values)
```

### What You Can Learn

- **Top Contributing Features**: Which HRV metrics are most important
- **Prediction Direction**: Features pushing towards arrhythmia vs normal
- **Clinical Interpretability**: Understand model decisions in medical context
- **Model Transparency**: Build trust with explainable predictions

## 🎯 Future Enhancements

### Model Improvements
- **Enhanced Training**: Expand to all 48 MIT-BIH records
- **Advanced Architectures**: LSTM, Transformer-based models
- **Ensemble Methods**: Combine multiple model predictions
- **Real-time Processing**: Sliding window for continuous monitoring

### Integration for Hackathon
- **✅ Remote PPG Extraction**: Extract heart rate from facial video (rPPG) - **IMPLEMENTED**
- **✅ Real-time Video Processing**: Process video frames in real-time - **IMPLEMENTED**
- **✅ Cloud API Integration**: Template for video API integration - **READY**
- **✅ Alert System**: Real-time alerts for arrhythmia detection - **IMPLEMENTED**
- **Video Analysis**: Combine rPPG predictions with facial cues (skin tone, pupil dilation)

### Performance Optimization
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Cross-validation**: More robust model evaluation
- **Class Imbalance Handling**: Advanced techniques for skewed datasets

## 📚 References

- **MIT-BIH Arrhythmia Database**: https://www.physionet.org/content/mitdb/1.0.0/
- **PhysioNet**: https://physionet.org/
- **wfdb Package**: https://github.com/MIT-LCP/wfdb-python
- **NeuroKit2**: https://neurokit2.readthedocs.io/

## ⚠️ Important Notes

**Medical Disclaimer**: This project is intended for **educational and research purposes only**. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

**Hackathon Context**: This model is designed for the Healthcare Hackathon challenge focusing on driver health assessment. It should be integrated with other safety systems and not used as the sole determinant for medical decisions.

## 🤝 Contributing

This project is developed for a healthcare hackathon. For improvements:
1. Expand training dataset to include more records
2. Experiment with different model architectures
3. Integrate video analysis features
4. Optimize for real-time performance

## 📄 License

See LICENSE file for details.

---

**Status**: ✅ Model trained and ready for predictions
**Last Updated**: Training completed with 94.4% accuracy on MIT-BIH dataset
