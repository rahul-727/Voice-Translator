import streamlit as st
import librosa
import numpy as np
import pickle
import os
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
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# Function to authenticate user via voice
def authenticate_user(audio_file, allowed_users):
    model, label_encoder = load_model()
    features = extract_features(audio_file)
    if features is not None:
        try:
            prediction = model.predict([features])
            predicted_label = label_encoder.inverse_transform(prediction)
            if predicted_label[0] in allowed_users:
                return predicted_label[0]
            else:
                return None
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None
    return None

# Streamlit app
def main():
    st.title("Voice Authentication App")

    # List of allowed users
    allowed_users = ["rahul", "margaret", "jens"]

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.header("Voice Authentication Required")
        uploaded_file = st.file_uploader("Upload your voice (.wav file)", type=["wav"])
        
        if uploaded_file is not None:
            with open("temp.wav", "wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())

            # Authenticate the user
            user = authenticate_user("temp.wav", allowed_users)
            if user:
                st.success(f"Welcome {user.capitalize()}! You are authenticated.")
                st.session_state.authenticated = True
            else:
                st.error("Voice not recognized. You are not authenticated.")

    if st.session_state.authenticated:
        st.header("Main Application")
        st.write("This is the main interface of the application.")
        # Add more features here later

if __name__ == "__main__":
    main()
