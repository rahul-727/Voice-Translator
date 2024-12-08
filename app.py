import streamlit as st
import librosa
import numpy as np
import pickle
import os
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoder
def load_model():
    with open("voice_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("label_encoder.pkl", "rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)
    return model, label_encoder

# Extract MFCC features from an audio file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=2.5, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))

        # Combine features
        features = np.hstack([
            np.mean(mfcc.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(spectral_contrast.T, axis=0),
            zero_crossing_rate
        ])
        return features
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None


# Function to authenticate user via voice
def authenticate_user(audio_file, allowed_users):
    model, label_encoder = load_model()
    features = extract_features(audio_file)
    if features is not None:
        try:
            # Get probabilities for each class
            probabilities = model.predict_proba([features])
            max_prob = max(probabilities[0])
            predicted_label = label_encoder.inverse_transform([np.argmax(probabilities[0])])[0]

            # Only authenticate if the maximum probability exceeds a threshold
            if max_prob > 0.8 and predicted_label in allowed_users:
                return predicted_label
            else:
                return None
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None
    return None


# Record audio from user
def record_audio(filename, duration=5, fs=44100):
    st.info(f"Recording for {duration} seconds...")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  # Wait for the recording to finish
        write(filename, fs, recording)  # Save as WAV file
        st.success("Recording completed.")
    except Exception as e:
        st.error(f"Error during recording: {e}")

# Streamlit app
def main():
    st.title("Voice Authentication App 🎙️")
    st.subheader("Your voice is your key to access 🔐")

    # List of allowed users
    allowed_users = ["rahul", "margaret", "jens"]

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("### Provide your voice for authentication:")
        option = st.radio("Choose an option:", ["🎤 Record Voice", "📁 Upload Voice File"])

        if option == "🎤 Record Voice":
            if st.button("Start Recording"):
                record_audio("temp.wav", duration=5)
                st.success("Recording complete. Authenticating...")
                user = authenticate_user("temp.wav", allowed_users)
                if user:
                    st.success(f"Welcome, {user.capitalize()}! Access granted.")
                    st.session_state.authenticated = True
                else:
                    st.error("Authentication failed. Voice not recognized.")
        
        elif option == "📁 Upload Voice File":
            uploaded_file = st.file_uploader("Upload a .wav file:", type=["wav"])
            if uploaded_file:
                with open("temp.wav", "wb") as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                st.success("File uploaded. Authenticating...")
                user = authenticate_user("temp.wav", allowed_users)
                if user:
                    st.success(f"Welcome, {user.capitalize()}! Access granted.")
                    st.session_state.authenticated = True
                else:
                    st.error("Authentication failed. Voice not recognized.")
    
    if st.session_state.authenticated:
        st.header("Welcome to the App 🎉")
        st.write("You now have access to all the features.")


if __name__ == "__main__":
    main()
