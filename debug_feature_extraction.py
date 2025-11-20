"""
Script untuk membandingkan feature extraction antara CSV dan real-time extraction
Ini akan membantu menemukan perbedaan yang menyebabkan prediksi salah
"""

import librosa
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict

def extract_features_from_csv(csv_path, sample_index=0):
    """Get features from CSV (ground truth)"""
    print("="*70)
    print(f"LOADING FEATURES FROM CSV (Sample #{sample_index})")
    print("="*70)
    
    data = pd.read_csv(csv_path)
    
    # Get one sample
    sample = data.iloc[sample_index]
    filename = sample['filename'] if 'filename' in sample else f"Sample {sample_index}"
    label = sample['label']
    
    print(f"Filename: {filename}")
    print(f"True Label: {label}")
    
    # Get features (exclude filename and label)
    features = sample.drop(['filename', 'label'], errors='ignore')
    
    print(f"Total features: {len(features)}")
    print(f"\nFirst 10 features:")
    for i, (name, value) in enumerate(list(features.items())[:10], 1):
        print(f"  {i:2d}. {name:25s} = {value:.6f}")
    
    return features.to_dict(), filename, label

def extract_features_from_audio(audio_path):
    """Extract features from audio file (real-time)"""
    print("\n" + "="*70)
    print(f"EXTRACTING FEATURES FROM AUDIO")
    print("="*70)
    
    # Load audio (SAME as training script)
    y, sr = librosa.load(audio_path, duration=30)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    
    print(f"Audio loaded:")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(y)/sr:.2f} seconds")
    print(f"  Trimmed duration: {len(y_trimmed)/sr:.2f} seconds")
    
    features = OrderedDict()
    
    # 1. Length
    features['length'] = len(y_trimmed) / sr
    
    # 2. Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y_trimmed, sr=sr)
    features['chroma_stft_mean'] = np.mean(chroma_stft)
    features['chroma_stft_var'] = np.var(chroma_stft)
    
    # 3. RMS
    rms = librosa.feature.rms(y=y_trimmed)
    features['rms_mean'] = np.mean(rms)
    features['rms_var'] = np.var(rms)
    
    # 4. Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_var'] = np.var(spectral_centroids)
    
    # 5. Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_var'] = np.var(spectral_bandwidth)
    
    # 6. Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)
    features['rolloff_mean'] = np.mean(spectral_rolloff)
    features['rolloff_var'] = np.var(spectral_rolloff)
    
    # 7. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y_trimmed)
    features['zero_crossing_rate_mean'] = np.mean(zcr)
    features['zero_crossing_rate_var'] = np.var(zcr)
    
    # 8. Harmony and Percussive
    y_harmonic, y_percussive = librosa.effects.hpss(y_trimmed)
    features['harmony_mean'] = np.mean(y_harmonic)
    features['harmony_var'] = np.var(y_harmonic)
    features['perceptr_mean'] = np.mean(y_percussive)
    features['perceptr_var'] = np.var(y_percussive)
    
    # 9. Tempo
    tempo, _ = librosa.beat.beat_track(y=y_trimmed, sr=sr)
    features['tempo'] = tempo
    
    # 10. MFCCs
    mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = np.mean(mfccs[i-1])
        features[f'mfcc{i}_var'] = np.var(mfccs[i-1])
    
    print(f"\nTotal features extracted: {len(features)}")
    print(f"\nFirst 10 features:")
    for i, (name, value) in enumerate(list(features.items())[:10], 1):
        print(f"  {i:2d}. {name:25s} = {value:.6f}")
    
    return dict(features)

def compare_features(csv_features, audio_features):
    """Compare features from CSV and audio"""
    print("\n" + "="*70)
    print("COMPARING FEATURES")
    print("="*70)
    
    all_keys = set(csv_features.keys()) | set(audio_features.keys())
    
    print(f"\nTotal unique features: {len(all_keys)}")
    
    # Check for missing features
    missing_in_audio = set(csv_features.keys()) - set(audio_features.keys())
    missing_in_csv = set(audio_features.keys()) - set(csv_features.keys())
    
    if missing_in_audio:
        print(f"\n⚠️ Features in CSV but NOT in audio extraction ({len(missing_in_audio)}):")
        for feat in list(missing_in_audio)[:5]:
            print(f"  • {feat}")
    
    if missing_in_csv:
        print(f"\n⚠️ Features in audio but NOT in CSV ({len(missing_in_csv)}):")
        for feat in list(missing_in_csv)[:5]:
            print(f"  • {feat}")
    
    # Compare values
    print(f"\n" + "="*70)
    print("VALUE COMPARISON")
    print("="*70)
    
    comparison_data = []
    
    def to_scalar(v):
        """Convert possible array-like feature to a single float (mean). Return np.nan if not convertible."""
        try:
            # pandas Series handled here too
            if isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v, dtype=float)
                if arr.size == 0:
                    return np.nan
                return float(np.mean(arr))
            # pandas may pass a Series or other; try to convert via numpy
            import pandas as _pd
            if isinstance(v, _pd.Series):
                arr = v.astype(float).values
                if arr.size == 0:
                    return np.nan
                return float(np.mean(arr))
            # scalar convertible
            return float(v)
        except Exception:
            return np.nan
    
    for key in sorted(all_keys):
        csv_val_raw = csv_features.get(key, np.nan)
        audio_val_raw = audio_features.get(key, np.nan)
        
        csv_val = to_scalar(csv_val_raw)
        audio_val = to_scalar(audio_val_raw)
        
        # Only compare when both are finite scalars
        if np.isfinite(csv_val) and np.isfinite(audio_val):
            diff = abs(csv_val - audio_val)
            diff_pct = (diff / (abs(csv_val) + 1e-10)) * 100
            
            comparison_data.append({
                'Feature': key,
                'CSV': csv_val,
                'Audio': audio_val,
                'Diff': diff,
                'Diff%': diff_pct
            })
    
    # Sort by difference percentage
    comparison_data.sort(key=lambda x: x['Diff%'], reverse=True)
    
    # Show top 20 most different
    print(f"\nTop 20 features with LARGEST differences:")
    print(f"{'Feature':<25} {'CSV Value':>12} {'Audio Value':>12} {'Diff%':>10}")
    print("-" * 70)
    
    for item in comparison_data[:20]:
        print(f"{item['Feature']:<25} {item['CSV']:>12.6f} {item['Audio']:>12.6f} {item['Diff%']:>9.2f}%")
    
    # Calculate overall similarity
    if len(comparison_data) > 0:
        valid_diffs = [x['Diff%'] for x in comparison_data if np.isfinite(x['Diff%'])]
        if len(valid_diffs) > 0:
            avg_diff_pct = np.mean(valid_diffs)
            print(f"\nAverage difference: {avg_diff_pct:.2f}%")
            
            if avg_diff_pct > 10:
                print("⚠️ WARNING: Large differences detected!")
                print("This could be due to:")
                print("  1. Different audio preprocessing")
                print("  2. Different librosa parameters")
                print("  3. Different audio file used")
            elif avg_diff_pct > 5:
                print("⚠️ Moderate differences detected")
            else:
                print("✅ Features are very similar!")
        else:
            print("⚠️ No finite Diff% values to compute average")
    else:
        print("⚠️ No comparable features found")
    
    return comparison_data

def test_prediction(features_csv, features_audio):
    """Test prediction with both feature sets"""
    print("\n" + "="*70)
    print("TESTING PREDICTIONS")
    print("="*70)
    
    # Load models
    with open('./saved_models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('./saved_models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('./saved_models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Get feature names
    if hasattr(scaler, 'feature_names_in_'):
        feature_names = scaler.feature_names_in_.tolist()
    else:
        with open('./saved_models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
    
    print(f"Model expects {len(feature_names)} features")
    
    # Prepare CSV features
    df_csv = pd.DataFrame([features_csv])
    df_csv = df_csv[feature_names]
    X_csv_scaled = scaler.transform(df_csv)
    pred_csv = model.predict(X_csv_scaled)[0]
    pred_csv_label = label_encoder.inverse_transform([pred_csv])[0]
    proba_csv = model.predict_proba(X_csv_scaled)[0]
    
    print(f"\n1️⃣ Prediction using CSV features:")
    print(f"   Predicted: {pred_csv_label}")
    print(f"   Confidence: {np.max(proba_csv):.1%}")
    print(f"   Top 3 genres:")
    top3_idx = np.argsort(proba_csv)[::-1][:3]
    for idx in top3_idx:
        print(f"     • {label_encoder.classes_[idx]}: {proba_csv[idx]:.1%}")
    
    # Prepare audio features
    df_audio = pd.DataFrame([features_audio])
    # Add missing features with 0
    for feat in feature_names:
        if feat not in df_audio.columns:
            df_audio[feat] = 0
    df_audio = df_audio[feature_names]
    X_audio_scaled = scaler.transform(df_audio)
    pred_audio = model.predict(X_audio_scaled)[0]
    pred_audio_label = label_encoder.inverse_transform([pred_audio])[0]
    proba_audio = model.predict_proba(X_audio_scaled)[0]
    
    print(f"\n2️⃣ Prediction using AUDIO extraction features:")
    print(f"   Predicted: {pred_audio_label}")
    print(f"   Confidence: {np.max(proba_audio):.1%}")
    print(f"   Top 3 genres:")
    top3_idx = np.argsort(proba_audio)[::-1][:3]
    for idx in top3_idx:
        print(f"     • {label_encoder.classes_[idx]}: {proba_audio[idx]:.1%}")
    
    # Compare
    if pred_csv_label == pred_audio_label:
        print(f"\n✅ Both predictions match: {pred_csv_label}")
    else:
        print(f"\n❌ PREDICTIONS DIFFER!")
        print(f"   CSV:   {pred_csv_label}")
        print(f"   Audio: {pred_audio_label}")
        print(f"\n   This explains why dashboard predictions are wrong!")

if __name__ == "__main__":
    import sys
    import os

    # Path to CSV
    csv_path = './saved_models/features_30_sec.csv'

    # Get sample index from command line or use default
    sample_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # --- Validate CSV exists and index is in range ---
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        print("Please ensure './saved_models/features_30_sec.csv' exists")
        sys.exit(1)

    data = pd.read_csv(csv_path)
    n_samples = len(data)
    if sample_idx < 0 or sample_idx >= n_samples:
        print(f"❌ Sample index {sample_idx} out of range (0 .. {n_samples-1})")
        print("Available samples (first 10):")
        for i, row in data.head(10).iterrows():
            fn = row.get('filename', '<no filename>')
            lbl = row.get('label', '<no label>')
            print(f"  {i}: {fn} [{lbl}]")
        print(f"\nUsage: python debug_feature_extraction.py <index> <path_to_audio?>")
        sys.exit(1)
    # --- end validation ---

    # Get audio path from command line
    if len(sys.argv) > 2:
        audio_path = sys.argv[2]
    else:
        # Try to find the audio file based on CSV
        sample = data.iloc[sample_idx]
        if 'filename' in sample:
            filename = sample['filename']
            # Try common paths
            possible_paths = [
                f'./Data/genres_original/{sample["label"]}/{filename}',
                f'../Data/genres_original/{sample["label"]}/{filename}',
                f'/content/gtzan-dataset/Data/genres_original/{sample["label"]}/{filename}'
            ]
            
            audio_path = None
            for path in possible_paths:
                import os
                if os.path.exists(path):
                    audio_path = path
                    break
            
            if audio_path is None:
                print(f"❌ Could not find audio file: {filename}")
                print(f"Please provide path as: python debug_feature_extraction.py {sample_idx} <path_to_audio>")
                sys.exit(1)
        else:
            print("❌ Please provide audio file path")
            print(f"Usage: python debug_feature_extraction.py {sample_idx} <path_to_audio>")
            sys.exit(1)
    
    print("="*70)
    print("FEATURE EXTRACTION COMPARISON")
    print("="*70)
    print(f"CSV Sample: {sample_idx}")
    print(f"Audio File: {audio_path}")
    print("="*70)
    
    # Extract features from both sources
    csv_features, filename, true_label = extract_features_from_csv(csv_path, sample_idx)
    audio_features = extract_features_from_audio(audio_path)
    
    # Compare
    comparison = compare_features(csv_features, audio_features)
    
    # Test predictions
    test_prediction(csv_features, audio_features)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nTrue label: {true_label}")
    print("\nIf predictions differ, the issue is in feature extraction.")
    print("Check the 'Top 20 features with LARGEST differences' above.")