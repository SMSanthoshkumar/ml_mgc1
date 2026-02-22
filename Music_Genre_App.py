import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import os
import base64
import gdown
from tensorflow.image import resize


# ---------------- MODEL DOWNLOAD + LOAD ---------------- #

@st.cache_resource()
def load_model():

    model_path = "Trained_model.h5"

    # Download from Google Drive if not exists
    if not os.path.exists(model_path):
        file_id = "1VnuC-M0MjxU-6rl4FphC-1vH6GGJg-a1"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, model_path, quiet=False)

    model = tf.keras.models.load_model(model_path)
    return model


# ---------------- PREPROCESS FUNCTION ---------------- #

def load_and_preprocess_data(file_path, target_shape=(150, 150)):

    data = []

    audio_data, sample_rate = librosa.load(file_path, sr=None)

    chunk_duration = 4
    overlap_duration = 2

    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    num_chunks = int(
        np.ceil((len(audio_data) - chunk_samples) /
                (chunk_samples - overlap_samples))
    ) + 1

    for i in range(num_chunks):

        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples

        chunk = audio_data[start:end]

        if len(chunk) < chunk_samples:
            continue

        mel_spectrogram = librosa.feature.melspectrogram(
            y=chunk,
            sr=sample_rate,
            n_mels=128
        )

        mel_spectrogram = resize(
            np.expand_dims(mel_spectrogram, axis=-1),
            target_shape
        )

        data.append(mel_spectrogram)

    return np.array(data)


# ---------------- PREDICTION FUNCTION ---------------- #

def model_prediction(X_test):

    model = load_model()
    y_pred = model.predict(X_test, verbose=0)

    predicted_categories = np.argmax(y_pred, axis=1)

    unique_elements, counts = np.unique(
        predicted_categories, return_counts=True)

    return unique_elements[np.argmax(counts)]


# ---------------- GLOBAL STYLE ---------------- #

st.markdown("""
<style>
.stApp {
    background-color: #F4F8FF;
    color: #1F2937;
}

h1, h2, h3, h4, h5, h6 {
    color: #1F2937 !important;
}

section[data-testid="stSidebar"] {
    background-color: #E6F0FF;
}

section[data-testid="stSidebar"] * {
    color: #1F2937 !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: white !important;
    color: #1F2937 !important;
}

.stButton > button {
    background-color: #4F8EF7 !important;
    color: white !important;
    border-radius: 8px;
    border: none;
    padding: 8px 16px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #3A6EDC !important;
    color: white !important;
}

.bottom-right {
    position: fixed;
    bottom: 25px;
    right: 25px;
    width: 300px;
    opacity: 0.95;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ---------------- #

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page", ["Home", "About Project", "Prediction"]
)


# ---------------- HOME PAGE ---------------- #

if app_mode == "Home":

    st.markdown("## Welcome to the")
    st.markdown("## Music Genre Classification System! 🎶🎧")

    st.markdown("""
    **Upload an audio file and let AI identify its genre.**

    ### How It Works
    - Upload audio file  
    - Convert audio to Mel Spectrogram  
    - CNN model predicts the genre  

    ### Supported Genres
    Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, Rock
    """)

    if os.path.exists("music_genre_home.png"):
        with open("music_genre_home.png", "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()

        st.markdown(
            f'<img src="data:image/png;base64,{encoded}" class="bottom-right">',
            unsafe_allow_html=True
        )


# ---------------- ABOUT PAGE ---------------- #

elif app_mode == "About Project":

    st.markdown("""
    ### About Project
    This project classifies music genres using deep learning and audio signal processing.
                
    ### Team Members
    - **Sanjay M (23BIT095)** – Team Lead  
    - **Santhoshkumar S M (23BIT096)** – ML Engineer  
    - **Ram Prakesh A (23BIT080)** – Data Engineer  
    - **Rohith V (23BIT086)** – Quality Analyst  
    - **Mukesh S R (23BIT063)** – Deployment Engineer  

    ### Dataset
    - GTZAN Dataset  
    - 10 genres  
    - 100 audio files per genre  
    - 30 seconds each  

    ### Features Used
    - Mel Spectrogram  
    - Convolutional Neural Network (CNN)  
    """)


# ---------------- PREDICTION PAGE ---------------- #

elif app_mode == "Prediction":

    st.header("🎵 Music Genre Prediction")

    test_audio = st.file_uploader(
        "Upload an audio file",
        type=["mp3", "wav"]
    )

    if test_audio is not None:

        os.makedirs("Test_Music", exist_ok=True)
        filepath = os.path.join("Test_Music", test_audio.name)

        with open(filepath, "wb") as f:
            f.write(test_audio.getbuffer())

        if st.button("▶️ Play Audio"):
            st.audio(test_audio)

        if st.button("🔮 Predict"):

            with st.spinner("Analyzing audio..."):

                X_test = load_and_preprocess_data(filepath)

                result_index = model_prediction(X_test)

                labels = [
                    "blues", "classical", "country", "disco",
                    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
                ]

                predicted_genre = labels[result_index]

                st.markdown(
                    f"""
                    <div style="
                        background-color:#d4edda;
                        padding:12px;
                        border-radius:8px;
                        font-size:18px;
                        font-weight:bold;
                        color:black;
                        border:1px solid #c3e6cb;">
                        🎧 Predicted Genre: {predicted_genre.upper()}
                    </div>
                    """,
                    unsafe_allow_html=True
                )