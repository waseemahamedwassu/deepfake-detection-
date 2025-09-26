import streamlit as st
import cv2
import numpy as np
import librosa
import tempfile
import os
from PIL import Image
import time
import torch

# ------------------------------
# Dummy video detector
# ------------------------------
class DummyVideoDetector:
    def predict(self, video_path):
        import random
        return random.uniform(0.4, 0.6)  # Random fake probability

def load_video_detector(predict_fn=None):
    return DummyVideoDetector()

# ------------------------------
# Dummy image & audio detectors
# ------------------------------
class DummyImageDetector:
    def predict(self, image):
        import random
        return random.uniform(0.4, 0.6)

class DummyAudioDetector:
    def predict(self, audio_path):
        import random
        return random.uniform(0.4, 0.6)

def load_image_detector(device=None):
    return DummyImageDetector()

def load_audio_detector(device=None):
    return DummyAudioDetector()

# ------------------------------
# Streamlit app
# ------------------------------

# Page config
st.set_page_config(page_title="Deepfake Detection App", page_icon="🔍", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {text-align: center; color: #1f77b4; font-size: 2.5rem; margin-bottom: 2rem;}
    .result-box {padding: 1rem; border-radius: 10px; text-align: center; font-size: 1.5rem; margin: 1rem 0;}
    .real {background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb;}
    .fake {background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def _load_detectors():
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    img_det = load_image_detector(device=device)
    aud_det = load_audio_detector(device=device)
    vid_det = load_video_detector(img_det.predict)
    return img_det, aud_det, vid_det

def analyze_audio(audio_path, aud_det):
    if aud_det is None:
        return 0.5
    try:
        return aud_det.predict(audio_path)
    except:
        return 0.5

def analyze_video(video_path, vid_det):
    if vid_det is None:
        return 0.5
    try:
        return vid_det.predict(video_path)
    except:
        return 0.5

def main():
    st.markdown('<h1 class="main-header">🔍 Deepfake Detection (Demo)</h1>', unsafe_allow_html=True)
    st.markdown("### Upload a file to analyze (Image, Audio, Video)")

    img_det, aud_det, vid_det = _load_detectors()

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['jpg','jpeg','png','mp3','wav','avi','mp4','mov']
    )

    if uploaded_file is not None:
        file_extension = uploaded_file.name.lower().split('.')[-1]

        if file_extension in ['jpg', 'jpeg', 'png']:
            st.write("**Analyzing image...**")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            fake_prob = img_det.predict(image)

        elif file_extension in ['mp3', 'wav']:
            st.write("**Analyzing audio...**")
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            fake_prob = analyze_audio(tmp_path, aud_det)
            os.unlink(tmp_path)

        elif file_extension in ['mp4', 'avi', 'mov']:
            st.write("**Analyzing video...**")
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            st.video(tmp_path)
            fake_prob = analyze_video(tmp_path, vid_det)
            os.unlink(tmp_path)
        else:
            st.error("Unsupported file format!")
            return

        real_prob = 1 - fake_prob

        # Show results
        if real_prob > 0.7:
            result_class = "real"; result_text="✅ LIKELY REAL"; confidence=real_prob
        elif fake_prob > 0.7:
            result_class = "fake"; result_text="❌ LIKELY FAKE"; confidence=fake_prob
        else:
            result_class = "real"; result_text="❓ UNCERTAIN"; confidence=max(real_prob,fake_prob)

        st.markdown(f"""
        <div class="result-box {result_class}">
            <strong>{result_text}</strong><br>
            Confidence: {confidence:.1%}
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        col1, col2 = st.columns(2)
        with col1: st.metric("Real Probability", f"{real_prob:.1%}")
        with col2: st.metric("Fake Probability", f"{fake_prob:.1%}")

if __name__ == "__main__":
    main()
