import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import pickle
import os
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from scipy import signal
from scipy.fft import fft, fftfreq

st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        with open('./saved_models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('./saved_models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('./saved_models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Get feature names from scaler
        if hasattr(scaler, 'feature_names_in_'):
            feature_names = scaler.feature_names_in_.tolist()
            st.sidebar.success(f"✓ Loaded {len(feature_names)} features from scaler")
        else:
            try:
                with open('./saved_models/feature_names.pkl', 'rb') as f:
                    feature_names = pickle.load(f)
                st.sidebar.success(f"✓ Loaded {len(feature_names)} features from file")
            except:
                st.sidebar.error("⚠️ Could not load feature names!")
                feature_names = None
        
        return model, scaler, label_encoder, feature_names, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, False

@st.cache_data
def load_comparison_data():
    try:
        comparison_df = pd.read_csv('./saved_models/model_comparison.csv')
        importance_df = pd.read_csv('./saved_models/feature_importance.csv')
        return comparison_df, importance_df
    except Exception as e:
        st.warning(f"Could not load comparison data: {e}")
        return None, None

def extract_features_gtzan_format(audio_file, expected_feature_order=None):
    """
    Extract features in GTZAN CSV format - EXACT ORDER MATTERS!
    CRITICAL: Must use EXACT same calculation as training script!
    """
    try:
        # Load and preprocess audio - EXACT same as training
        y, sr = librosa.load(audio_file, duration=30)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Use OrderedDict to maintain insertion order
        from collections import OrderedDict
        features = OrderedDict()
        
        # CRITICAL: Length must be in SAMPLES, not seconds!
        # Training uses: len(y_trimmed) directly as integer
        features['length'] = len(y_trimmed)  # ← FIXED: Not divided by sr!
        
        # 2. Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=y_trimmed, sr=sr)
        features['chroma_stft_mean'] = np.mean(chroma_stft)
        features['chroma_stft_var'] = np.var(chroma_stft)
        
        # 3. RMS Energy
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
        
        # 6. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)
        features['rolloff_mean'] = np.mean(spectral_rolloff)
        features['rolloff_var'] = np.var(spectral_rolloff)
        
        # 7. Zero Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y_trimmed)
        features['zero_crossing_rate_mean'] = np.mean(zero_crossing_rate)
        features['zero_crossing_rate_var'] = np.var(zero_crossing_rate)
        
        # 8. Harmonic and Percussive
        y_harmonic, y_percussive = librosa.effects.hpss(y_trimmed)
        features['harmony_mean'] = np.mean(y_harmonic)
        features['harmony_var'] = np.var(y_harmonic)
        features['perceptr_mean'] = np.mean(y_percussive)
        features['perceptr_var'] = np.var(y_percussive)
        
        # 9. Tempo
        tempo, _ = librosa.beat.beat_track(y=y_trimmed, sr=sr)
        features['tempo'] = tempo
        
        # 10. MFCCs (20 coefficients) - MUST BE IN ORDER
        mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=20)
        for i in range(1, 21):
            features[f'mfcc{i}_mean'] = np.mean(mfccs[i-1])
            features[f'mfcc{i}_var'] = np.var(mfccs[i-1])
        
        # If expected order is provided, reorder to match
        if expected_feature_order is not None:
            ordered_features = OrderedDict()
            for feat_name in expected_feature_order:
                if feat_name in features:
                    ordered_features[feat_name] = features[feat_name]
                else:
                    # Missing feature - set to 0
                    ordered_features[feat_name] = 0.0
                    st.warning(f"⚠️ Missing feature: {feat_name} - setting to 0")
            
            return ordered_features, y_trimmed, sr
        
        return features, y_trimmed, sr
    
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None

@st.cache_data
def load_recommendation_data():
    try:
        # Try different possible paths
        possible_paths = [
            './saved_models/features_30_sec.csv'
        ]
        
        rec_data = None
        for path in possible_paths:
            if os.path.exists(path):
                rec_data = pd.read_csv(path, index_col='filename')
                break
        
        if rec_data is None:
            return None, None, None, None, False
        
        rec_labels = rec_data[['label']]
        rec_features = rec_data.drop(columns=['length', 'label'], errors='ignore')
        
        from sklearn.preprocessing import StandardScaler
        scaler_rec = StandardScaler()
        rec_scaled = scaler_rec.fit_transform(rec_features)
        
        return rec_features, rec_labels, rec_scaled, scaler_rec, True
    except Exception as e:
        st.warning(f"Recommendation data not available: {e}")
        return None, None, None, None, False

def get_song_recommendations(features_dict, rec_features, rec_labels, rec_scaled, scaler_rec, top_n=5):
    try:
        new_features_df = pd.DataFrame([features_dict])
        
        # Match columns with dataset
        for col in rec_features.columns:
            if col not in new_features_df.columns:
                new_features_df[col] = 0
        
        new_features_df = new_features_df[rec_features.columns]
        
        new_features_scaled = scaler_rec.transform(new_features_df)
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(new_features_scaled, rec_scaled)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        recommendations = []
        for idx in top_indices:
            song_name = rec_labels.index[idx]
            similarity_score = similarities[idx]
            genre = rec_labels.iloc[idx]['label']
            
            recommendations.append({
                'Song': song_name,
                'Similarity': f"{similarity_score:.2%}",
                'Genre': genre.title()
            })
        
        rec_df = pd.DataFrame(recommendations)
        return rec_df
    
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return None

def create_waveform_plot(y, sr):
    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#1f77b4')
    ax.set_title('Waveform', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    plt.tight_layout()
    return fig

def create_spectrogram_plot(y, sr):
    fig, ax = plt.subplots(figsize=(12, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='viridis')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Spectrogram', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def create_mel_spectrogram_plot(y, sr):
    fig, ax = plt.subplots(figsize=(12, 4))
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='coolwarm')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel Spectrogram', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def create_mfcc_plot(y, sr):
    fig, ax = plt.subplots(figsize=(12, 4))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax, cmap='cool')
    fig.colorbar(img, ax=ax)
    ax.set_title('MFCCs (20 coefficients)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def main():
    st.markdown('<h1 class="main-header">🎵 Music Genre Classification Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">Using Digital Signal Processing and Machine Learning</p>', unsafe_allow_html=True)
    
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["🏠 Home", "🎵 Audio Classifier", "📊 Model Performance", "📈 Data Analysis", "ℹ️ About"]
    )
    
    model, scaler, label_encoder, feature_names, models_loaded = load_models()
    
    if not models_loaded:
        st.error("⚠️ Models not found! Please train the model first using the main script.")
        st.stop()
    
    if page == "🏠 Home":
        st.markdown('<h2 class="sub-header">Welcome to Music Genre Classification System</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Genres Supported", "10")
            st.caption("blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if feature_names:
                st.metric("DSP Features", len(feature_names))
            else:
                st.metric("DSP Features", "57")
            st.caption("MFCCs, Spectral, Temporal, Tonal, Harmonic")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            comparison_df, _ = load_comparison_data()
            if comparison_df is not None:
                best_acc = comparison_df['Accuracy'].max()
                st.metric("Best Accuracy", f"{best_acc:.2%}")
                best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
                st.caption(f"Model: {best_model_name}")
            else:
                st.metric("Best Accuracy", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown('<h3 class="sub-header">🔬 Key Features</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Digital Signal Processing
            - **Time Domain**: Waveform, ZCR, RMS Energy
            - **Frequency Domain**: FFT, Spectral Analysis
            - **Time-Frequency**: STFT, Mel Spectrogram
            - **Advanced**: MFCCs (20 coefficients)
            - **Harmonic/Percussive**: HPSS Separation
            """)
        
        with col2:
            st.markdown("""
            ### Machine Learning
            - **Algorithms**: Random Forest, SVM, Neural Network, XGBoost
            - **Validation**: 5-Fold Cross-Validation
            - **Metrics**: Accuracy, F1-Score, Confusion Matrix
            - **Feature Analysis**: Permutation Importance
            - **Recommendation**: Cosine Similarity
            """)
        
        st.markdown("---")
        
        st.markdown('<h3 class="sub-header">🚀 Quick Start Guide</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        1. **🎵 Audio Classifier**: Upload your music file and get instant genre prediction
        2. **📊 Model Performance**: View detailed model comparison and evaluation metrics
        3. **📈 Data Analysis**: Explore dataset statistics and visualizations
        4. **ℹ️ About**: Learn more about the methodology and techniques used
        """)
        
        st.info("👈 Use the sidebar to navigate between different pages")
    
    elif page == "🎵 Audio Classifier":
        st.markdown('<h2 class="sub-header">Upload and Classify Your Music</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <b>📌 Instructions:</b>
            <ol>
                <li>Upload an audio file (WAV, MP3, FLAC, OGG)</li>
                <li>Wait for feature extraction (may take a few seconds)</li>
                <li>View the predicted genre and confidence scores</li>
                <li>Explore audio visualizations</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'ogg'],
            help="Supported formats: WAV, MP3, FLAC, OGG"
        )
        
        if uploaded_file is not None:
            st.markdown("### 📁 File Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Filename:** {uploaded_file.name}")
            with col2:
                st.info(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
            with col3:
                st.info(f"**Type:** {uploaded_file.type}")
            
            st.markdown("### 🎧 Audio Player")
            st.audio(uploaded_file)
            
            with st.spinner("🔄 Extracting features... This may take a moment..."):
                features, y, sr = extract_features_gtzan_format(uploaded_file, expected_feature_order=feature_names)
            
            if features is not None and feature_names is not None:
                st.success("✅ Features extracted successfully!")
                
                # Convert OrderedDict to DataFrame
                features_df = pd.DataFrame([dict(features)])
                
                # CRITICAL: Ensure exact column order
                features_df = features_df[feature_names]
                
                # DEBUG: Show feature comparison
                with st.expander("🔍 Debug Info - Feature Matching"):
                    st.write(f"📊 Extracted features: {len(features_df.columns)}")
                    st.write(f"📊 Expected by model: {len(feature_names)}")
                    
                    # Verify order
                    order_match = list(features_df.columns) == list(feature_names)
                    if order_match:
                        st.success("✅ Feature order matches perfectly!")
                    else:
                        st.error("❌ Feature order mismatch!")
                    
                    # Show first and last features
                    comparison_df = pd.DataFrame({
                        'Position': range(1, 11),
                        'Extracted': list(features.keys())[:10],
                        'Expected': feature_names[:10],
                        'Match': ['✓' if list(features.keys())[i] == feature_names[i] else '✗' for i in range(10)]
                    })
                    st.write("**First 10 Features Comparison:**")
                    st.dataframe(comparison_df, hide_index=True)
                    
                    # Show sample values
                    st.write("**Sample Feature Values:**")
                    sample_values = pd.DataFrame({
                        'Feature': list(features.keys())[:10],
                        'Value': [f"{features[k]:.6f}" for k in list(features.keys())[:10]]
                    })
                    st.dataframe(sample_values, hide_index=True)
                
                try:
                    # Scale features
                    features_scaled = scaler.transform(features_df)
                    
                    # Predict
                    prediction = model.predict(features_scaled)[0]
                    predicted_genre = label_encoder.inverse_transform([prediction])[0]
                    
                    # Get probabilities
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(features_scaled)[0]
                    else:
                        probabilities = None
                    
                    st.markdown("---")
                    st.markdown("### 🎯 Prediction Results")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 30px; border-radius: 15px; text-align: center; color: white;">
                            <h1 style="margin: 0; font-size: 3rem;">🎵</h1>
                            <h2 style="margin: 10px 0;">{predicted_genre.upper()}</h2>
                            <p style="margin: 0; font-size: 1.2rem;">Predicted Genre</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if probabilities is not None:
                            st.markdown("#### Confidence Scores")
                            
                            prob_df = pd.DataFrame({
                                'Genre': label_encoder.classes_,
                                'Probability': probabilities
                            }).sort_values('Probability', ascending=False)
                            
                            fig = px.bar(
                                prob_df,
                                x='Probability',
                                y='Genre',
                                orientation='h',
                                color='Probability',
                                color_continuous_scale='blues',
                                text='Probability'
                            )
                            fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                            fig.update_layout(
                                height=400,
                                yaxis={'categoryorder': 'total ascending'},
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("### 📊 Audio Visualizations")
                    
                    viz_tabs = st.tabs(["🌊 Waveform", "📈 Spectrogram", "🎨 Mel Spectrogram", "🎼 MFCCs"])
                    
                    with viz_tabs[0]:
                        with st.spinner("Generating waveform..."):
                            fig = create_waveform_plot(y, sr)
                            st.pyplot(fig)
                    
                    with viz_tabs[1]:
                        with st.spinner("Generating spectrogram..."):
                            fig = create_spectrogram_plot(y, sr)
                            st.pyplot(fig)
                    
                    with viz_tabs[2]:
                        with st.spinner("Generating mel spectrogram..."):
                            fig = create_mel_spectrogram_plot(y, sr)
                            st.pyplot(fig)
                    
                    with viz_tabs[3]:
                        with st.spinner("Generating MFCCs..."):
                            fig = create_mfcc_plot(y, sr)
                            st.pyplot(fig)
                    
                    st.markdown("---")
                    st.markdown("### 🎯 Song Recommendations")
                    st.info("🎵 Based on audio features similarity with dataset songs")

                    rec_features, rec_labels, rec_scaled, scaler_rec, rec_loaded = load_recommendation_data()

                    if rec_loaded:
                        with st.spinner("🔍 Finding similar songs..."):
                            recommendations = get_song_recommendations(
                                features, 
                                rec_features, 
                                rec_labels, 
                                rec_scaled, 
                                scaler_rec, 
                                top_n=5
                            )
                        
                        if recommendations is not None:
                            st.markdown("#### Top 5 Similar Songs from Dataset:")
                            st.dataframe(
                                recommendations,
                                use_container_width=True,
                                hide_index=True
                            )
                            fig = px.bar(
                                recommendations,
                                x='Song',
                                y='Similarity',
                                color='Genre',
                                title='Similarity Scores',
                                text='Similarity'
                            )
                            fig.update_layout(height=400, xaxis_tickangle=-45)
                            fig.update_traces(textposition='outside')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.success("✅ These songs have similar audio characteristics to your uploaded file!")
                    else:
                        st.warning("⚠️ Recommendation feature requires features_30_sec.csv")
                    
                    st.markdown("---")
                    st.markdown("### 📋 Extracted Features")
                    
                    with st.expander("View detailed feature values"):
                        feature_display = pd.DataFrame({
                            'Feature': list(features.keys()),
                            'Value': list(features.values())
                        })
                        st.dataframe(feature_display, use_container_width=True, height=400)
                
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
                    import traceback
                    st.error(traceback.format_exc())
            
            elif features is None:
                st.error("❌ Failed to extract features from audio file")
            else:
                st.error("❌ Feature names not loaded. Please retrain the model.")
        
        else:
            st.markdown("""
            <div class="warning-box">
                <b>⚠️ No file uploaded</b><br>
                Please upload an audio file to start classification.
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "📊 Model Performance":
        st.markdown('<h2 class="sub-header">Model Performance Analysis</h2>', unsafe_allow_html=True)
        
        comparison_df, importance_df = load_comparison_data()
        
        if comparison_df is not None:
            st.markdown("### 📈 Model Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    comparison_df,
                    x='Model',
                    y='Accuracy',
                    title='Model Accuracy Comparison',
                    color='Accuracy',
                    color_continuous_scale='Blues',
                    text='Accuracy'
                )
                fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    comparison_df,
                    x='Model',
                    y='F1-Score',
                    title='Model F1-Score Comparison',
                    color='F1-Score',
                    color_continuous_scale='Reds',
                    text='F1-Score'
                )
                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'F1-Score']),
                use_container_width=True
            )
            
            best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
            st.success(f"""
            🏆 **Best Model**: {best_model['Model']}  
            ✅ **Accuracy**: {best_model['Accuracy']:.2%}  
            ✅ **F1-Score**: {best_model['F1-Score']:.4f}
            """)
        
        if importance_df is not None:
            st.markdown("---")
            st.markdown("### 🔍 Feature Importance Analysis")
            
            top_20 = importance_df.head(20)
            
            fig = px.bar(
                top_20,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 20 Most Important Features',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("View all feature importance values"):
                st.dataframe(importance_df, use_container_width=True, height=400)
    
    elif page == "📈 Data Analysis":
        st.markdown('<h2 class="sub-header">Dataset Analysis and Insights</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 📊 GTZAN Dataset Overview
        
        The GTZAN dataset is a collection of 1,000 audio tracks, each 30 seconds long:
        - **Total tracks**: 1,000 (100 per genre)
        - **Genres**: 10 (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
        - **Format**: WAV, 22050 Hz, mono
        - **Duration**: 30 seconds per track
        """)
        
        st.markdown("---")
        
        st.markdown("### 🎵 Genre Distribution")
        
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                 'jazz', 'metal', 'pop', 'reggae', 'rock']
        counts = [100] * 10
        
        fig = px.bar(
            x=genres,
            y=counts,
            labels={'x': 'Genre', 'y': 'Number of Tracks'},
            title='Number of Tracks per Genre',
            color=counts,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 🔬 Feature Extraction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Time Domain Features
            - Zero Crossing Rate (ZCR)
            - Root Mean Square (RMS) Energy
            - Duration (Length)
            
            #### Frequency Domain Features
            - Spectral Centroid
            - Spectral Rolloff
            - Spectral Bandwidth
            - Chroma STFT
            """)
        
        with col2:
            st.markdown("""
            #### Time-Frequency Features
            - Mel-Frequency Cepstral Coefficients (MFCCs)
            - 20 MFCC coefficients with mean and variance
            
            #### Other Features
            - Tempo (BPM)
            - Harmonic Components (HPSS)
            - Percussive Components (HPSS)
            """)
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Digital Signal Processing Techniques")
        
        st.markdown("""
        #### Signal Processing Pipeline
        1. **Audio Loading**
           - Sample rate: 22050 Hz
           - Duration: 30 seconds
           - Format: Mono channel
        
        2. **Preprocessing**
           - Silence trimming (top_db=20)
           - Normalization
        
        3. **Feature Extraction**
           - Time-domain analysis
           - Frequency-domain analysis
           - Time-frequency analysis
        
        4. **Feature Aggregation**
           - Statistical measures (mean, variance)
           - Per-frame and global features
        """)
    
    elif page == "ℹ️ About":
        st.markdown('<h2 class="sub-header">About This Project</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ## 🎵 Music Genre Classification Using Digital Signal Processing and Machine Learning
        
        ### 📖 Project Overview
        
        This project implements an intelligent music genre classification system that combines:
        - **Digital Signal Processing (DSP)** techniques for feature extraction
        - **Machine Learning** algorithms for classification
        - **Content-based Recommendation** using cosine similarity
        
        ### 🎯 Objectives
        
        1. Extract meaningful features from audio signals using DSP techniques
        2. Train and compare multiple machine learning algorithms
        3. Achieve high accuracy in genre classification
        4. Provide interactive visualization and prediction interface
        5. Implement content-based music recommendation
        
        ### 🔬 Methodology
        
        #### 1. Data Preprocessing
        - Audio loading and normalization
        - Silence trimming (top_db=20)
        - Duration standardization (30 seconds)
        
        #### 2. Feature Extraction (57 features)
        - **Length**: 1 feature
        - **Chroma STFT**: mean & variance = 2 features
        - **RMS**: mean & variance = 2 features
        - **Spectral Centroid**: mean & variance = 2 features
        - **Spectral Bandwidth**: mean & variance = 2 features
        - **Spectral Rolloff**: mean & variance = 2 features
        - **Zero Crossing Rate**: mean & variance = 2 features
        - **Harmony**: mean & variance = 2 features
        - **Percussive**: mean & variance = 2 features
        - **Tempo**: 1 feature
        - **MFCCs**: 20 coefficients × 2 (mean & var) = 40 features
        
        #### 3. Machine Learning Models
        - Random Forest Classifier
        - Support Vector Machine (SVM)
        - Neural Network (MLP)
        - XGBoost Classifier
        
        #### 4. Evaluation
        - 5-Fold Cross-Validation
        - Accuracy, F1-Score, Confusion Matrix
        - Permutation-based Feature Importance
        
        ### 📊 Results Summary
        """)
        
        comparison_df, importance_df = load_comparison_data()
        
        if comparison_df is not None:
            st.dataframe(
                comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'F1-Score']),
                use_container_width=True
            )
            
            best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
            st.success(f"""
            🏆 **Best Model**: {best_model['Model']}  
            ✅ **Accuracy**: {best_model['Accuracy']:.2%}  
            ✅ **F1-Score**: {best_model['F1-Score']:.4f}
            """)
        
        st.markdown("""
        ### 🛠️ Technologies Used
        
        - **Python 3.8+**
        - **Libraries**:
          - `librosa` - Audio analysis
          - `scikit-learn` - Machine learning
          - `xgboost` - Gradient boosting
          - `streamlit` - Web dashboard
          - `plotly` - Interactive visualizations
          - `scipy` - Signal processing
        
        ### 📚 Dataset
        
        **GTZAN Dataset** (Tzanetakis & Cook, 2002)
        - 1,000 audio tracks (100 per genre)
        - 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
        - 30 seconds per track
        - 22050 Hz sampling rate, mono
        
        ### 🎓 References
        
        1. Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. IEEE Transactions on speech and audio processing.
        2. McFee, B., et al. (2015). librosa: Audio and music signal analysis in python.
        3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
        4. Oppenheim, A. V., & Schafer, R. W. (2009). Discrete-time signal processing.
        
        ### 👨‍💻 Developer Information
        
        **Project Type**: Academic Research / Final Project  
        **Domain**: Music Information Retrieval (MIR)  
        **Techniques**: Digital Signal Processing, Machine Learning, Audio Analysis
        
        ### 🔧 Contact & Support
        
        For questions, feedback, or collaboration:
        - GitHub: [Your GitHub URL]
        - Email: [Your Email]
        - LinkedIn: [Your LinkedIn]
        
        ### 📄 License
        
        This project uses:
        - **GTZAN Dataset**: Available for research purposes
        - **Code**: MIT License
        - **Free Music Archive**: Creative Commons licenses
        
        ---
        
        ### 🙏 Acknowledgments
        
        - GTZAN Dataset creators
        - Librosa development team
        - Scikit-learn contributors
        - Streamlit community
        - Open-source community
        
        ---
        
        <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-top: 30px;">
            <h3>🎵 Thank you for using this system! 🎵</h3>
            <p>Built with ❤️ for music lovers and data scientists</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()