# --- Imports ---
import os
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
import tempfile 

import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf
from streamlit_mic_recorder import mic_recorder
from googletrans import Translator
import speech_recognition as sr
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from textblob import TextBlob

# --- Configuration ---
st.set_page_config(page_title="Emotion Fusion Analyzer", layout="centered", page_icon="🤖")

# --- Constants ---
TEXT_WEIGHT = 0.6
SPEECH_WEIGHT = 0.4
EMOTION_MAP = {
    'anger': 'angry', 'fear': 'fearful', 'joy': 'happy',
    'love': 'happy', 'sadness': 'sad', 'surprise': 'surprised'
}
EMOTION_ICONS = {
    'happy': '😊', 'sad': '😢', 'angry': '😠', 
    'fearful': '😨', 'surprised': '😲', 'neutral': '😐',
    'disgust': '🤢', 'calm': '😌'
}

# --- Model Loading ---
@st.cache_resource
def load_models():
    nltk.download('vader_lexicon')
    return (
        pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion"),
        tf.keras.models.load_model('testing10_model.h5')
    )

# --- Core Functions ---
def extract_audio_features(file_path):
    try:
        audio, sr = sf.read(file_path)
        audio = librosa.to_mono(audio.T) if audio.ndim > 1 else audio
        if sr != 22050:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
        return np.expand_dims(np.expand_dims(np.mean(mfccs.T, axis=0), -1), 0)
    except Exception as e:
        st.error(f"Audio processing error: {str(e)}")
        return None

def predict_speech_emotion(model, features):
    try:
        prediction = model.predict(features)[0]
        return ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'][np.argmax(prediction)], \
               np.max(prediction) * 100
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, 0.0

def analyze_text(text, classifier):
    try:
        sentiment = TextBlob(text).sentiment.polarity
        emotion_result = classifier(text)[0]
        return {
            'emotion': emotion_result['label'],
            # 'confidence': emotion_result['score'] * 100,
            'sentiment': 'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'
        }
    except Exception as e:
        st.error(f"Text analysis error: {str(e)}")
        return None

def fuse_results(speech_emo, speech_conf, text_analysis):
    scores = defaultdict(float)
    
    if speech_emo:
        scores[speech_emo.lower()] += speech_conf * SPEECH_WEIGHT
    
    if text_analysis:
        mapped_emo = EMOTION_MAP.get(text_analysis['emotion'].lower(), None)
        # if mapped_emo:
        #     scores[mapped_emo] += text_analysis['confidence'] * TEXT_WEIGHT
    
    if not scores:
        return None, 0.0
    
    final_emo, final_score = max(scores.items(), key=lambda x: x[1])
    return final_emo.capitalize(), round(final_score, 2)

# --- UI Components ---
def main():
    text_classifier, speech_model = load_models()
    translator = Translator()
    recognizer = sr.Recognizer()
    
    st.title("🤖 Multimodal Emotion Detection System")
    
    input_method = st.radio("Choose Input Mode:", ["🎤 Audio Input", "📝 Direct Text"])
    
    if input_method == "🎤 Audio Input":
        audio_file = st.file_uploader("Upload audio", type=["wav"])
        audio = mic_recorder(start_prompt="⏺ Record", stop_prompt="⏹ Stop", format="wav")
        
        if audio_file or (audio and audio['bytes']):
            with st.spinner("Analyzing..."):
                try:
                    file_bytes = audio_file.read() if audio_file else audio['bytes']
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        f.write(file_bytes)
                        file_path = f.name
                    
                    features = extract_audio_features(file_path)
                    if features is None:
                        return
                    
                    speech_emo, speech_conf = predict_speech_emotion(speech_model, features)
                    
                    with sr.AudioFile(file_path) as source:
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data, language="en")
                        translated = translator.translate(text, dest='en').text
                    
                    text_analysis = analyze_text(translated, text_classifier)
                    os.unlink(file_path)
                
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
                    return
            
            if speech_emo and text_analysis:
                final_emo, final_conf = fuse_results(speech_emo, speech_conf, text_analysis)

                st.markdown("---")
                cols = st.columns([1, 2])
                with cols[0]:
                    st.markdown(f"## {EMOTION_ICONS.get(final_emo.lower(), '❓')}")
                with cols[1]:
                    st.markdown(f"""
                    ### Final Emotion: **{final_emo}**
                 
                    *Combined analysis (Speech {int(SPEECH_WEIGHT*100)}% + Text {int(TEXT_WEIGHT*100)}%)*  
                    """)
                    # st.progress(min(100, int(final_conf)))

                with st.expander("▶️ Listen to Audio"):
                    st.audio(file_bytes, format='audio/wav')

                # with st.expander("📊 Detailed Analysis"):
                #     cols = st.columns(2)
                #     with cols[0]:
                #         st.markdown(f"""
                #         #### 🗣️ Speech Analysis
                #         - Emotion: **{speech_emo}**  
                       
                #         """)
                #     with cols[1]:
                #         st.markdown(f"""
                #         #### 📝 Text Analysis
                #         - Emotion: **{text_analysis['emotion']}**  
                        
                #         - Sentiment: **{text_analysis['sentiment']}**
                #         """)
                #     st.caption(f"**Transcribed Text**: *{text}*")

    else:
        text_input = st.text_area("Enter your text:", height=100)
        if st.button("Analyze"):
            with st.spinner("Processing..."):
                text_analysis = analyze_text(text_input, text_classifier)
                if text_analysis:
                    final_emo, final_conf = fuse_results(None, 0, text_analysis)

                    st.markdown("---")
                    cols = st.columns([1, 2])
                    with cols[0]:
                        st.markdown(f"## {EMOTION_ICONS.get(final_emo.lower(), '❓')}")
                    with cols[1]:
                        st.markdown(f"""
                        ### Final Emotion: **{final_emo}**
                        
                        *Text-only analysis*
                        """)
                        st.progress(min(100, int(final_conf)))

                    # with st.expander("📊 Detailed Text Analysis"):
                    #     st.markdown(f"""
                    #     - Emotion: **{text_analysis['emotion']}**  
                          
                    #     - Sentiment: **{text_analysis['sentiment']}**
                    #     """)

if __name__ == "__main__":
    main()
