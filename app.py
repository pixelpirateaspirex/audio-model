import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import tempfile
import os
import base64

# Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['English', 'Hindi']
MODEL_PATH = 'language_id_model.keras'

# Set layout to wide for more space
st.set_page_config(page_title="Lang ID AI", page_icon="🌐", layout="wide")

# Inject Custom CSS for Premium Design & Glassmorphism
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

/* Apply Outfit font to the global app container */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

/* Background gradient */
.stApp {
    background: radial-gradient(circle at top left, #1a1a2e, #16213e, #0f3460);
    color: #e0e0e0;
}

/* Hide Streamlit top margin and "Deploy" button completely */
header {visibility: hidden;}
.stDeployButton {display:none;}
#MainMenu {visibility: hidden;}

/* Custom Header Typography */
.main-title {
    font-size: 4rem;
    background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0px;
    padding-bottom: 0px;
    font-weight: 800;
}
.sub-title {
    text-align: center;
    font-size: 1.3rem;
    color: #a0a0a0;
    margin-top: 0px;
    margin-bottom: 40px;
    font-weight: 300;
}

/* Glassmorphism sections */
.glass-container {
    background: rgba(255, 255, 255, 0.03); /* subtle shine */
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 30px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    margin-bottom: 20px;
}

/* Section labels */
.section-title {
    font-size: 1.5rem;
    color: #ffffff;
    font-weight: 600;
    margin-bottom: 15px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 10px;
}

/* Custom Result Card */
.result-card {
    background: linear-gradient(135deg, rgba(23, 25, 36, 0.8), rgba(15, 52, 96, 0.8));
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    animation: fadeIn 0.8s ease-in-out;
}
.result-lang {
    font-size: 3.5rem;
    font-weight: 800;
    margin: 15px 0;
    letter-spacing: 2px;
}
.english-lang { color: #00C9FF; text-shadow: 0 0 20px rgba(0,201,255,0.6); }
.hindi-lang { color: #FF007F; text-shadow: 0 0 20px rgba(255,0,127,0.6); }

.confidence-text {
    font-size: 1.3rem;
    color: #cccccc;
    font-weight: 300;
}

/* Idle state styling */
.idle-state {
    text-align: center;
    color: #6c757d;
    padding: 60px 20px;
    border: 2px dashed rgba(255,255,255,0.1);
    border-radius: 15px;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Custom st.progress bar color */
.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #00C9FF, #92FE9D);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return keras.models.load_model(MODEL_PATH)
    except Exception as e:
        return None

def process_audio(audio_bytes, suffix):
    """
    Converts raw audio bytes into a mel-spectrogram image array.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name
        
    try:
        y, sr = librosa.load(temp_audio_path, sr=22050, mono=True)
        mel_spec_db = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000),
            ref=np.max
        )

        fig, ax = plt.subplots(figsize=(IMG_WIDTH / 100, IMG_HEIGHT / 100), dpi=100)
        librosa.display.specshow(mel_spec_db, sr=sr, fmax=8000, ax=ax)
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        img_array = np.array(
            Image.open(buf).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT)),
            dtype=np.float32
        ) / 255.0
        
        return img_array
    except Exception as e:
        st.error(f"Failed to process audio file: {e}")
        return None
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def main():
    # Animated header title
    st.markdown("<h1 class='main-title'>Language ID Network</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>High-fidelity <b>English vs Hindi</b> spoken language detection using spectrograms.</p>", unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("🚨 Neural network model `language_id_model.keras` could not be loaded. Please ensure it exists.")
        return

    # Two Column Layout
    col1, spacer, col2 = st.columns([1.2, 0.1, 1])

    img_array = None
    uploaded_file = None
    file_extension = None

    with col1:
        st.markdown("<div class='section-title'>1. Input Data</div>", unsafe_allow_html=True)
        
        # We use a standard st.container to group elements visually, 
        # but apply the glass class manually to a div to wrap content.
        # However, to wrap streamlit interactive widgets in a class, we rely on standard layout,
        # and CSS targets general areas. We'll add text below to keep styled.
        st.markdown("<p style='color: #bbb;'>Please select an audio file (WAV, MP3) or a pre-computed Spectrogram image.</p>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "", 
            type=["wav", "mp3", "ogg", "flac", "png", "jpg", "jpeg"]
        )
        
        if uploaded_file is not None:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension in ['.png', '.jpg', '.jpeg']:
                st.image(uploaded_file, caption="Uploaded Data", use_container_width=True)
            else:
                st.audio(uploaded_file)
        else:
            st.info("Upload a file above to begin the inference process.")

    with col2:
        st.markdown("<div class='section-title'>2. Analysis & Results</div>", unsafe_allow_html=True)
        
        if uploaded_file is None:
            st.markdown("""
            <div class='glass-container idle-state'>
                <h3 style='color: #777; margin-bottom: 10px;'>Awaiting Input</h3>
                <p>Neural inference results will be displayed here once data is uploaded and processed.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner('Neural Network processing...'):
                if file_extension in ['.png', '.jpg', '.jpeg']:
                    image = Image.open(uploaded_file).convert('RGB')
                    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
                    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                else:
                    img_array = process_audio(uploaded_file.read(), file_extension)

            if img_array is not None:
                img_array = np.expand_dims(img_array, axis=0)
                prediction = model.predict(img_array)[0][0]
                
                is_hindi = prediction > 0.5
                predicted_class = CLASS_NAMES[1] if is_hindi else CLASS_NAMES[0]
                confidence = prediction if is_hindi else (1 - prediction)
                
                lang_class_css = "hindi-lang" if is_hindi else "english-lang"

                st.markdown(f"""
                <div class='result-card glass-container'>
                    <div style='text-transform: uppercase; letter-spacing: 2px; color: #bbb; font-size: 0.9rem;'>Detected Language</div>
                    <div class='result-lang {lang_class_css}'>{predicted_class}</div>
                    <div class='confidence-text'>Network Confidence: <b>{confidence * 100:.2f}%</b></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.progress(float(confidence), text="Confidence Metric")

if __name__ == '__main__':
    main()
