"""
Script untuk test model prediction dengan data dari training set
Untuk memastikan model dan feature extraction bekerja dengan benar
"""

import pickle
import pandas as pd
import numpy as np
import os

def load_models():
    """Load saved models"""
    print("="*70)
    print("LOADING MODELS")
    print("="*70)
    
    with open('./saved_models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded")
    
    with open('./saved_models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Scaler loaded")
    
    with open('./saved_models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("✅ Label encoder loaded")
    
    # Get feature names
    if hasattr(scaler, 'feature_names_in_'):
        feature_names = scaler.feature_names_in_.tolist()
        print(f"✅ Feature names from scaler: {len(feature_names)} features")
    else:
        try:
            with open('./saved_models/feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
            print(f"✅ Feature names from file: {len(feature_names)} features")
        except:
            print("❌ No feature names found!")
            feature_names = None
    
    return model, scaler, label_encoder, feature_names

def load_test_data():
    """Load test data from CSV"""
    print("\n" + "="*70)
    print("LOADING TEST DATA")
    print("="*70)
    
    # Try to find CSV
    possible_paths = [
        './saved_models/features_3_sec.csv'
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        print("❌ Could not find features_3_sec.csv")
        return None, None
    
    print(f"✅ Found CSV: {csv_path}")
    
    data = pd.read_csv(csv_path)
    print(f"📊 Total rows: {len(data)}")
    
    # Drop filename if exists
    if 'filename' in data.columns:
        data = data.drop('filename', axis=1)
    
    # Separate features and labels
    X = data.drop('label', axis=1)
    y = data['label']
    
    print(f"📊 Features shape: {X.shape}")
    print(f"📊 Labels shape: {y.shape}")
    
    return X, y

def test_predictions(model, scaler, label_encoder, feature_names, X, y, num_samples=10):
    """Test model predictions"""
    print("\n" + "="*70)
    print(f"TESTING PREDICTIONS (First {num_samples} samples)")
    print("="*70)
    
    # Ensure feature order matches
    if feature_names is not None:
        print(f"\n📋 Reordering features to match training...")
        X_ordered = X[feature_names]
    else:
        print("⚠️ No feature names - using data as-is")
        X_ordered = X
    
    # Take first N samples
    X_test = X_ordered.head(num_samples)
    y_test = y.head(num_samples)
    
    # Scale
    X_scaled = scaler.transform(X_test)
    
    # Predict
    predictions = model.predict(X_scaled)
    predicted_labels = label_encoder.inverse_transform(predictions)
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_scaled)
        max_probs = np.max(probabilities, axis=1)
    else:
        max_probs = [1.0] * len(predictions)
    
    # Display results
    print(f"\n{'No':<4} {'True Label':<12} {'Predicted':<12} {'Confidence':<12} {'Status'}")
    print("-" * 60)
    
    correct = 0
    for i in range(len(X_test)):
        true_label = y_test.iloc[i]
        pred_label = predicted_labels[i]
        confidence = max_probs[i]
        status = "✓" if true_label == pred_label else "✗"
        
        if true_label == pred_label:
            correct += 1
        
        print(f"{i+1:<4} {true_label:<12} {pred_label:<12} {confidence:>6.1%}       {status}")
    
    accuracy = correct / len(X_test)
    print("-" * 60)
    print(f"Accuracy: {correct}/{len(X_test)} = {accuracy:.1%}")
    
    if accuracy < 0.5:
        print("\n⚠️ WARNING: Accuracy is very low!")
        print("This suggests feature order mismatch or model issues.")
    elif accuracy > 0.8:
        print("\n✅ Good accuracy! Model is working correctly.")
    else:
        print("\n⚠️ Moderate accuracy. May need investigation.")
    
    return accuracy

def test_specific_genres(model, scaler, label_encoder, feature_names, X, y):
    """Test predictions for each genre"""
    print("\n" + "="*70)
    print("TESTING BY GENRE")
    print("="*70)
    
    genres = y.unique()
    
    for genre in sorted(genres):
        # Get samples for this genre
        genre_mask = y == genre
        X_genre = X[genre_mask]
        y_genre = y[genre_mask]
        
        # Take first 5 samples
        X_test = X_genre.head(5)
        y_test = y_genre.head(5)
        
        if feature_names is not None:
            X_test = X_test[feature_names]
        
        # Scale and predict
        X_scaled = scaler.transform(X_test)
        predictions = model.predict(X_scaled)
        predicted_labels = label_encoder.inverse_transform(predictions)
        
        # Calculate accuracy
        correct = sum(y_test.values == predicted_labels)
        accuracy = correct / len(y_test)
        
        print(f"\n{genre.upper():<12} - Accuracy: {correct}/5 = {accuracy:.0%}")
        
        # Show predictions
        for i, (true_label, pred_label) in enumerate(zip(y_test.values, predicted_labels), 1):
            status = "✓" if true_label == pred_label else f"✗ (predicted as {pred_label})"
            print(f"  Sample {i}: {status}")

if __name__ == "__main__":
    # Load everything
    model, scaler, label_encoder, feature_names = load_models()
    
    if feature_names is None:
        print("\n❌ Cannot proceed without feature names!")
        print("Please run the training script with feature name saving enabled.")
        exit(1)
    
    X, y = load_test_data()
    
    if X is None:
        print("\n❌ Cannot proceed without test data!")
        exit(1)
    
    # Test predictions
    accuracy = test_predictions(model, scaler, label_encoder, feature_names, X, y, num_samples=20)
    
    # Test by genre
    test_specific_genres(model, scaler, label_encoder, feature_names, X, y)
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    
    if accuracy > 0.8:
        print("\n✅ Model is working correctly!")
        print("You can now use the dashboard with confidence.")
    else:
        print("\n⚠️ Model accuracy is low. Possible issues:")
        print("  1. Feature order mismatch")
        print("  2. Model not trained properly")
        print("  3. Scaler not fitted correctly")
        print("\nRecommendation: Retrain the model and ensure feature names are saved.")