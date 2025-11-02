"""
Explain predictions from arrhythmia detection model.
Command-line tool for XAI analysis.
"""

import numpy as np
import argparse
from model import ArrhythmiaPredictor
from data_preprocessing import preprocess_signal
from explainability import ModelExplainer, explain_prediction_from_model
import sys


def main():
    parser = argparse.ArgumentParser(description='Explain arrhythmia detection predictions')
    parser.add_argument('--model', type=str, default='model.pkl',
                       help='Path to trained model')
    parser.add_argument('--signal', type=str, required=True,
                       help='Path to signal data (CSV, numpy array, or video)')
    parser.add_argument('--rate', type=int, default=125,
                       help='Sampling rate in Hz')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save explanation plots (optional)')
    parser.add_argument('--top', type=int, default=10,
                       help='Number of top features to display')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Explainable AI: Arrhythmia Detection Prediction")
    print("=" * 60)
    
    # Load signal data
    print(f"\nLoading signal from {args.signal}...")
    try:
        if args.signal.endswith('.csv'):
            signal_data = np.loadtxt(args.signal, delimiter=',')
        elif args.signal.endswith('.npy'):
            signal_data = np.load(args.signal)
        elif args.signal.endswith(('.mp4', '.avi', '.mov')):
            # Extract rPPG from video
            print("Extracting rPPG from video...")
            from video_rppg_extractor import extract_rppg_from_video_file
            signal_data, _, _ = extract_rppg_from_video_file(args.signal)
            if signal_data is None:
                print("Error: Failed to extract rPPG from video")
                sys.exit(1)
            args.rate = 30  # Video typically 30 fps
        else:
            signal_data = np.loadtxt(args.signal)
    except Exception as e:
        print(f"Error loading signal: {e}")
        sys.exit(1)
    
    print(f"Signal loaded: {len(signal_data)} samples, {len(signal_data)/args.rate:.1f} seconds")
    
    # Get explanation
    print("\nGenerating explanation...")
    try:
        explanation, shap_values = explain_prediction_from_model(
            args.model, signal_data, args.rate
        )
    except Exception as e:
        print(f"Error generating explanation: {e}")
        print("\nMake sure SHAP is installed: pip install shap")
        sys.exit(1)
    
    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION EXPLANATION")
    print("=" * 60)
    
    # Show top contributing features
    print(f"\n📊 Top {args.top} Most Important Features:")
    print("-" * 60)
    
    for i, (feature, contribution) in enumerate(explanation['top_features'][:args.top], 1):
        direction = "📈 PUSHES TOWARDS ARRHYTHMIA" if contribution > 0 else "📉 PUSHES TOWARDS NORMAL"
        print(f"{i:2d}. {feature:20s}: {contribution:+.4f}  {direction}")
    
    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY:")
    print(f"  Total features analyzed: {len(explanation['feature_names'])}")
    print(f"  Total contribution: {sum(explanation['shap_values']):+.4f}")
    
    positive_contrib = sum(v for v in explanation['shap_values'] if v > 0)
    negative_contrib = sum(v for v in explanation['shap_values'] if v < 0)
    
    print(f"  Positive contributions (→ arrhythmia): {positive_contrib:+.4f}")
    print(f"  Negative contributions (→ normal): {negative_contrib:+.4f}")
    
    net_prediction = positive_contrib + negative_contrib
    if net_prediction > 0:
        print(f"\n  ✅ Net prediction: ARRHYTHMIA (confidence: {abs(net_prediction):.2%})")
    else:
        print(f"\n  ✅ Net prediction: NORMAL (confidence: {abs(net_prediction):.2%})")
    
    # Also show actual prediction
    print("\n" + "-" * 60)
    print("ACTUAL PREDICTION:")
    
    predictor = ArrhythmiaPredictor()
    predictor.load(args.model)
    features = preprocess_signal(signal_data, args.rate)
    pred, prob = predictor.predict(features.reshape(1, -1))
    
    status = "ARRHYTHMIA DETECTED" if pred[0] == 1 else "NORMAL"
    print(f"  Status: {status}")
    print(f"  Confidence: {prob[0]:.2%}")
    
    # Feature categories
    print("\n" + "-" * 60)
    print("FEATURE CATEGORIES:")
    
    hrv_features = [f for f in explanation['feature_names'] if any(x in f for x in ['rr', 'hrv', 'rmssd', 'pnn'])]
    freq_features = [f for f in explanation['feature_names'] if any(x in f for x in ['lf', 'hf', 'power', 'ratio'])]
    stat_features = [f for f in explanation['feature_names'] if f not in hrv_features and f not in freq_features]
    
    print(f"  HRV Features: {len(hrv_features)} (heart rate variability)")
    print(f"  Frequency Features: {len(freq_features)} (spectral analysis)")
    print(f"  Statistical Features: {len(stat_features)} (signal statistics)")
    
    # Save plots if requested
    if args.save:
        print(f"\nSaving plots to {args.save}...")
        explainer = predictor.get_explainer()
        
        # Waterfall plot
        explainer.plot_waterfall(shap_values, save_path=args.save.replace('.png', '_waterfall.png'), show=False)
        
        # Feature importance plot
        if shap_values.ndim > 1:
            explainer.plot_feature_importance(shap_values, save_path=args.save.replace('.png', '_importance.png'), show=False)
        
        print(f"  ✅ Plots saved")
    
    print("\n" + "=" * 60)
    print("Explanation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

