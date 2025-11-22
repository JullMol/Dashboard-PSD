import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pickle
import os
import sys
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from collections import OrderedDict
from scipy import signal
from scipy.fft import fft, fftfreq

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + current_dir

st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: inherit;
        text-align: center;
        margin-bottom: 0;
        background: -webkit-linear-gradient(120deg, #00F260, #0575E6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        text-align: center;
        font-size: 1.5rem;
        font-weight: 500;
        color: inherit;
        opacity: 0.8;
        margin-top: -5px;
        margin-bottom: 0;
    }
    .author-text {
        text-align: center;
        font-size: 0.9rem;
        font-weight: 600;
        color: inherit;
        opacity: 0.6;
        margin-top: 5px;
        margin-bottom: 40px;
        font-style: italic;
        letter-spacing: 1px;
    }
    .workflow-step {
        background: linear-gradient(135deg, #0575E6 0%, #021B79 100%);
        border-left: 5px solid #00F260;
        padding: 25px;
        border-radius: 15px;
        color: #ffffff;
        text-align: center;
        height: 100%;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    .workflow-step:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 242, 96, 0.2);
        border-left: 5px solid #ffffff;
    }
    .workflow-icon {
        font-size: 2.5rem;
        margin-bottom: 10px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .workflow-title {
        font-weight: 800;
        font-size: 1.2rem;
        margin-bottom: 5px;
        color: #ffffff;
        letter-spacing: 0.5px;
    }
    .workflow-desc {
        font-size: 0.9rem;
        color: #e0e0e0;
        font-weight: 400;
    }
    .metric-container {
        text-align: center;
        padding: 10px;
    }
    .metric-value {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1;
        filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.1));
    }
    .metric-label {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 10px;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .cta-box {
        background: linear-gradient(90deg, rgba(5, 117, 230, 0.1) 0%, rgba(0, 242, 96, 0.1) 100%);
        border: 1px solid rgba(5, 117, 230, 0.2);
        color: inherit;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 25px 0;
        font-weight: 500;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        base_path = './saved_models'
        if not os.path.exists(base_path):
            return None, None, None, None, False

        model = None
        scaler = None
        label_encoder = None
        feature_names = None

        if os.path.exists(f'{base_path}/genre_classifier.pkl'):
            with open(f'{base_path}/genre_classifier.pkl', 'rb') as f:
                package = pickle.load(f)
                model = package.get('model')
                scaler = package.get('scaler')
                label_encoder = package.get('label_encoder')
                feature_names = package.get('feature_names')
        
        if model is None and os.path.exists(f'{base_path}/best_model.pkl'):
            with open(f'{base_path}/best_model.pkl', 'rb') as f:
                model = pickle.load(f)

        if scaler is None and os.path.exists(f'{base_path}/scaler.pkl'):
            with open(f'{base_path}/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
                
        if label_encoder is None and os.path.exists(f'{base_path}/label_encoder.pkl'):
            with open(f'{base_path}/label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)

        if feature_names is None and os.path.exists(f'{base_path}/feature_names.pkl'):
            with open(f'{base_path}/feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
        
        if model is not None and scaler is not None and label_encoder is not None:
            return model, scaler, label_encoder, feature_names, True
        else:
            return None, None, None, None, False
            
    except Exception as e:
        return None, None, None, None, False

@st.cache_data
def load_comparison_data():
    try:
        base_path = './saved_models'
        comp_path = f'{base_path}/model_comparison.csv'
        imp_path = f'{base_path}/feature_importance.csv'
        
        if os.path.exists(comp_path) and os.path.exists(imp_path):
            comparison_df = pd.read_csv(comp_path)
            importance_df = pd.read_csv(imp_path)
            return comparison_df, importance_df
        else:
            return None, None
    except:
        return None, None

@st.cache_data
def load_recommendation_data():
    try:
        path = './saved_models/features_30_sec.csv'
        if not os.path.exists(path):
             if os.path.exists('features_30_sec.csv'):
                 path = 'features_30_sec.csv'
             else:
                return None, None, None, None, False

        rec_data = pd.read_csv(path, index_col='filename')
        rec_labels = rec_data[['label']]
        rec_features = rec_data.drop(columns=['length', 'label'], errors='ignore')
        
        from sklearn.preprocessing import StandardScaler
        scaler_rec = StandardScaler()
        rec_scaled = scaler_rec.fit_transform(rec_features)
        
        return rec_features, rec_labels, rec_scaled, scaler_rec, True
    except:
        return None, None, None, None, False

def apply_digital_filters(y, sr, use_filter='gentle'):
    nyquist = sr / 2
    order = 5

    if use_filter == 'gentle':
        cutoff = 18000 / nyquist
        if cutoff < 1.0:
            b, a = signal.butter(order, cutoff, btype='low')
            y_lowpass = signal.filtfilt(b, a, y)
        else:
            y_lowpass = y

        cutoff_high = 20 / nyquist
        b_hp, a_hp = signal.butter(order, cutoff_high, btype='high')
        y_highpass = signal.filtfilt(b_hp, a_hp, y)

        low = 20 / nyquist
        high = 18000 / nyquist
        if high < 1.0:
            b_bp, a_bp = signal.butter(order, [low, high], btype='band')
            y_bandpass = signal.filtfilt(b_bp, a_bp, y)
        else:
            y_bandpass = y_highpass
    else:
        y_lowpass = y
        y_highpass = y
        y_bandpass = y

    return {
        'original': y,
        'lowpass': y_lowpass,
        'highpass': y_highpass,
        'bandpass': y_bandpass
    }

def extract_features_gtzan_format(audio_file, expected_feature_order=None):
    tmp_path = None
    try:
        suffix = os.path.splitext(audio_file.name)[1]
        if not suffix: suffix = ".mp3"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
            
        try:
            y, sr = librosa.load(tmp_path, duration=30)
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            if len(y_trimmed) < 100 or np.max(np.abs(y_trimmed)) < 0.0001:
                return None, None, None
                
        except Exception:
            return None, None, None

        filters = apply_digital_filters(y_trimmed, sr, use_filter='gentle')
        y_proc = filters['bandpass']

        features = OrderedDict()
        
        features['length'] = len(y_proc)

        chroma = librosa.feature.chroma_stft(y=y_proc, sr=sr)
        features['chroma_stft_mean'] = np.mean(chroma)
        features['chroma_stft_var'] = np.var(chroma)

        rms = librosa.feature.rms(y=y_proc)
        features['rms_mean'] = np.mean(rms)
        features['rms_var'] = np.var(rms)

        cent = librosa.feature.spectral_centroid(y=y_proc, sr=sr)
        features['spectral_centroid_mean'] = np.mean(cent)
        features['spectral_centroid_var'] = np.var(cent)

        bw = librosa.feature.spectral_bandwidth(y=y_proc, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(bw)
        features['spectral_bandwidth_var'] = np.var(bw)

        rolloff = librosa.feature.spectral_rolloff(y=y_proc, sr=sr)
        features['rolloff_mean'] = np.mean(rolloff)
        features['rolloff_var'] = np.var(rolloff)

        zcr = librosa.feature.zero_crossing_rate(y_proc)
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_var'] = np.var(zcr)

        y_harm, y_perc = librosa.effects.hpss(y_proc)
        features['harmony_mean'] = np.mean(y_harm)
        features['harmony_var'] = np.var(y_harm)
        features['perceptr_mean'] = np.mean(y_perc)
        features['perceptr_var'] = np.var(y_perc)

        tempo, _ = librosa.beat.beat_track(y=y_proc, sr=sr)
        features['tempo'] = float(tempo)

        mfccs = librosa.feature.mfcc(y=y_proc, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc{i+1}_var'] = np.var(mfccs[i])

        if expected_feature_order:
            ordered_features = OrderedDict()
            for feat in expected_feature_order:
                if feat in features:
                    ordered_features[feat] = features[feat]
                else:
                    ordered_features[feat] = 0.0
            return ordered_features, y_trimmed, sr
            
        return features, y_trimmed, sr

    except Exception:
        return None, None, None
        
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

def get_song_recommendations(features_dict, rec_features, rec_labels, rec_scaled, scaler_rec, top_n=5):
    try:
        new_input = pd.DataFrame([features_dict])
        
        common_cols = [col for col in rec_features.columns if col in new_input.columns]
        if not common_cols:
            return None
            
        new_input = new_input[common_cols]
        
        try:
            new_scaled = scaler_rec.transform(new_input)
        except:
            new_scaled = new_input.values 

        from sklearn.metrics.pairwise import cosine_similarity
        rec_subset_scaled = scaler_rec.transform(rec_features[common_cols])
        similarities = cosine_similarity(new_scaled, rec_subset_scaled)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'Track Name': rec_labels.index[idx],
                'Genre': rec_labels.iloc[idx]['label'].title(),
                'Similarity': f"{similarities[idx]:.2%}"
            })
            
        return pd.DataFrame(recommendations)
    except:
        return None

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#00C9FF')
    ax.set_title('Waveform', fontweight='bold', color='#B0BEC5')
    ax.set_xlabel('Time (s)', color='#B0BEC5')
    ax.set_ylabel('Amplitude', color='#B0BEC5')
    ax.tick_params(axis='x', colors='#B0BEC5')
    ax.tick_params(axis='y', colors='#B0BEC5')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#546E7A')
    ax.spines['left'].set_color('#546E7A')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='inferno')
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='#B0BEC5')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#B0BEC5')
    ax.set_title('Spectrogram', fontweight='bold', color='#B0BEC5')
    ax.set_xlabel('Time', color='#B0BEC5')
    ax.set_ylabel('Frequency', color='#B0BEC5')
    ax.tick_params(axis='x', colors='#B0BEC5')
    ax.tick_params(axis='y', colors='#B0BEC5')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_mel_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='plasma')
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='#B0BEC5')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#B0BEC5')
    ax.set_title('Mel-Spectrogram', fontweight='bold', color='#B0BEC5')
    ax.set_xlabel('Time', color='#B0BEC5')
    ax.set_ylabel('Mel Freq', color='#B0BEC5')
    ax.tick_params(axis='x', colors='#B0BEC5')
    ax.tick_params(axis='y', colors='#B0BEC5')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def main():
    st.markdown('<div class="main-title">Music Genre Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Digital Signal Processing and Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="author-text">by DD\' INT-24</div>', unsafe_allow_html=True)
    
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio("Select Module:", ["🏠 Home", "🎵 Audio Classifier", "📊 Model Performance", "📈 Data Insights", "ℹ️ About System"])
    
    model, scaler, label_encoder, feature_names, models_loaded = load_models()
    
    if page == "🏠 Home":
        st.markdown("### Welcome!!")
        st.markdown("This professional suite provides real-time audio analysis and genre classification capabilities using state-of-the-art machine learning algorithms trained on the GTZAN dataset.")
        
        comp_df, _ = load_comparison_data()
        best_acc = f"{comp_df['Accuracy'].max():.1%}" if comp_df is not None else "91.3%"
        n_features = len(feature_names) if feature_names else 57
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">10</div>
                <div class="metric-label">Supported Genres</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{n_features}</div>
                <div class="metric-label">Extraction Features</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{best_acc}</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="cta-box">
            👉 READY to analyze? Navigate to the <b>🎵 Audio Classifier</b> menu to upload your file!
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### System Workflow")
        
        col_w1, col_w2, col_w3, col_w4 = st.columns(4)
        
        with col_w1:
            st.markdown("""
            <div class="workflow-step">
                <div class="workflow-icon">📂</div>
                <div class="workflow-title">Input</div>
                <div class="workflow-desc">Raw Audio Upload</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_w2:
            st.markdown("""
            <div class="workflow-step">
                <div class="workflow-icon">⚡</div>
                <div class="workflow-title">Processing</div>
                <div class="workflow-desc">Noise Filtering</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_w3:
            st.markdown("""
            <div class="workflow-step">
                <div class="workflow-icon">🎛️</div>
                <div class="workflow-title">Extraction</div>
                <div class="workflow-desc">DSP Features</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_w4:
            st.markdown("""
            <div class="workflow-step">
                <div class="workflow-icon">🧠</div>
                <div class="workflow-title">Inference</div>
                <div class="workflow-desc">ML Prediction</div>
            </div>
            """, unsafe_allow_html=True)

    elif page == "🎵 Audio Classifier":
        if not models_loaded:
            st.error("⚠️ Critical Error: Model artifacts are missing. Please verify the installation.")
            st.stop()

        st.markdown("### Real-time Classifier")
        st.info("Upload an audio file (WAV, MP3, FLAC) to extract features and classify the genre in real-time. The system will process the signal and provide visual analysis.")
        
        uploaded_file = st.file_uploader("Drop audio file here", type=['wav', 'mp3', 'flac'])
        
        if uploaded_file:
            st.audio(uploaded_file)
            
            with st.spinner("Analyzing signal properties..."):
                features, y, sr = extract_features_gtzan_format(uploaded_file, expected_feature_order=feature_names)
            
            if features:
                features_df = pd.DataFrame([features])
                
                if hasattr(scaler, 'feature_names_in_'):
                     valid_cols = [col for col in scaler.feature_names_in_ if col in features_df.columns]
                     features_df = features_df[valid_cols]
                     if len(valid_cols) != len(scaler.feature_names_in_):
                         missing = set(scaler.feature_names_in_) - set(valid_cols)
                         for c in missing:
                             features_df[c] = 0.0
                     features_df = features_df[scaler.feature_names_in_]
                else:
                     if feature_names:
                        valid_cols = [col for col in feature_names if col in features_df.columns]
                        features_df = features_df[valid_cols]

                try:
                    features_scaled = scaler.transform(features_df)
                    pred_idx = model.predict(features_scaled)[0]
                    pred_genre = label_encoder.inverse_transform([pred_idx])[0]
                    
                    probs = None
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(features_scaled)[0]

                    st.markdown("---")
                    col_res1, col_res2 = st.columns([1, 2])
                    
                    with col_res1:
                        st.success(f"**Identified Genre:**\n# {pred_genre.upper()}")
                    
                    with col_res2:
                        if probs is not None:
                            prob_df = pd.DataFrame({'Genre': label_encoder.classes_, 'Confidence': probs})
                            prob_df = prob_df.sort_values('Confidence', ascending=True)
                            fig_prob = px.bar(prob_df, x='Confidence', y='Genre', orientation='h', 
                                              title="Probability Distribution", text_auto='.1%')
                            fig_prob.update_traces(marker_color='#00C9FF')
                            fig_prob.update_layout(height=300)
                            st.plotly_chart(fig_prob, use_container_width=True)

                    st.markdown("### Signal Visualization")
                    tab1, tab2, tab3 = st.tabs(["Waveform", "Spectrogram", "Mel-Spec"])
                    with tab1: plot_waveform(y, sr)
                    with tab2: plot_spectrogram(y, sr)
                    with tab3: plot_mel_spectrogram(y, sr)

                    st.markdown("### Similar Tracks")
                    rec_feat, rec_lbl, rec_scl, rec_scaler, rec_ok = load_recommendation_data()
                    if rec_ok:
                        recs = get_song_recommendations(features, rec_feat, rec_lbl, rec_scl, rec_scaler)
                        if recs is not None:
                            st.dataframe(recs, use_container_width=True, hide_index=True)
                        else:
                            st.info("Could not generate recommendations due to feature mismatch.")
                    else:
                        st.info("Recommendation engine requires the feature dataset.")

                except Exception as e:
                    st.error(f"Prediction Error: {str(e)}")
            else:
                 pass

    elif page == "📊 Model Performance":
        st.markdown("### Comparative Benchmarking")
        comp_df, imp_df = load_comparison_data()
        
        if comp_df is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Accuracy & F1-Score by Algorithm**")
                fig = px.bar(comp_df, x='Model', y=['Accuracy', 'F1-Score'], 
                             barmode='group',
                             color_discrete_sequence=['#0575E6', '#00F260', '#8E2DE2'])
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.markdown("**Leaderboard**")
                display_df = comp_df[['Model', 'Accuracy', 'F1-Score']].sort_values(by='Accuracy', ascending=False)
                st.dataframe(display_df.style.format({'Accuracy': '{:.2%}', 'F1-Score': '{:.4f}'}), use_container_width=True)
                
                best_model = display_df.iloc[0]
                st.info(f"**Top Performer:** {best_model['Model']}")
                
        else:
            st.warning("Benchmark data not available.")

    elif page == "📈 Data Insights":
        st.markdown("### Dataset Analytics")
        
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        counts = [100] * 10
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Class Balance**")
            st.caption("Distribution of audio samples across different genres to ensure balanced training.")
            fig = px.pie(values=counts, names=genres, hole=0.5, color_discrete_sequence=px.colors.qualitative.Prism)
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("**Splitting Strategy**")
            st.caption("Ratio of data allocated for model training versus validation/testing.")
            split_data = pd.DataFrame({
                'Subset': ['Train', 'Test'],
                'Size': [80, 20]
            })
            fig_split = px.bar(split_data, x='Size', y='Subset', orientation='h', text='Size', color='Subset', 
                               color_discrete_map={'Train': '#0575E6', 'Test': '#82B1FF'})
            fig_split.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_split, use_container_width=True)

        st.markdown("### Feature Importance")
        st.caption("Ranking of DSP features based on their impact on the model's decision making.")
        _, imp_df = load_comparison_data()
        
        if imp_df is not None:
            top_n_features = imp_df.head(15).sort_values(by='Importance', ascending=True)
            fig_imp = px.bar(top_n_features, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Plasma')
            fig_imp.update_layout(height=500)
            st.plotly_chart(fig_imp, use_container_width=True)

    elif page == "ℹ️ About System":
        st.markdown("### Technical Architecture")
        st.markdown("This application represents a complete end-to-end pipeline for audio signal classification, utilizing advanced Digital Signal Processing (DSP) techniques and Machine Learning algorithms.")
        
        if os.path.exists("image_3a40e6.png"):
            st.image("image_3a40e6.png", caption="System Architecture", width=600)
        
        st.markdown("#### 1. Signal Processing")
        st.markdown("Incoming audio signals are sampled at 22,050 Hz. Silence is trimmed using a -20dB threshold. Digital filters (Butterworth) are applied to remove noise and emphasize relevant frequency bands.")
            
        st.markdown("#### 2. Feature Engineering")
        st.markdown("A 57-dimensional feature vector is constructed, consisting of MFCCs (Timbre), Spectral Centroid (Brightness), Spectral Rolloff, and Zero Crossing Rate (Percussiveness).")
            
        st.markdown("#### 3. Classification")
        st.markdown("The feature vector is normalized using Standard Scaling. The classification is performed by a Support Vector Machine (SVM) with an RBF kernel, which has been validated to achieve superior performance compared to Random Forest and XGBoost.")
            
        st.markdown("---")
        st.caption("© 2024 Audio Intelligence Project.")

if __name__ == "__main__":
    main()