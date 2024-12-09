import logging
import torch
from transformers import pipeline
import pyttsx3
import speech_recognition as sr
import streamlit as st  # Ensure Streamlit is imported
from streamlit.components.v1 import html

# Setting up logging for better visibility of the process
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the translation pipeline
translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

# Initialize the speech recognition model
recognizer = sr.Recognizer()

def speech_to_text_from_mic(source_language="en-US"):
    try:
        with sr.Microphone() as source:
            logger.info("Please speak now...")
            recognizer.adjust_for_ambient_noise(source)  # Adjusting for ambient noise
            audio = recognizer.listen(source)  # Listen to the user's speech
            text = recognizer.recognize_google(audio, language=source_language)
            logger.info(f"Recognized Text: {text}")
            return text
    except sr.UnknownValueError:
        logger.error("Could not understand the audio.")
    except sr.RequestError as e:
        logger.error(f"Could not request results from the speech recognition service; {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during speech-to-text conversion: {e}")
    return None

def translate_text(text):
    try:
        logger.info(f"Translating text: {text}")
        translated_text = translation_pipeline(text)[0]['translation_text']
        logger.info(f"Translated Text: {translated_text}")
        return translated_text
    except Exception as e:
        logger.error(f"An error occurred during translation: {e}")
    return None

def text_to_speech_with_pyttsx3(text):
    try:
        logger.info("Converting text to speech with pyttsx3...")
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"An error occurred during text-to-speech conversion: {e}")

def main():
    st.title("🎤 **Voice Translator** 🌍")
    st.subheader("Speak to translate your words across languages")

    st.markdown("""
    Welcome to the **Voice Translator** app! Simply press the button below, speak your words, 
    and watch as they are automatically translated into a different language. You will also hear 
    the translation through voice feedback.
    """)
    
    st.markdown("<hr>", unsafe_allow_html=True)

    # Styling the button and adding a background color for emphasis
    button_style = """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 12px;
        width: 200px;
        height: 60px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

    # Button for starting speech input
    if st.button("🎙️ Start Speaking"):
        st.spinner("Listening... please speak clearly into your microphone.")
        source_language = "en-US"
        recognized_text = speech_to_text_from_mic(source_language)
        if recognized_text:
            st.markdown(f"### 📜 **Recognized Text**: _{recognized_text}_")
            translated_text = translate_text(recognized_text)
            if translated_text:
                st.markdown(f"### ✨ **Translated Text**: _{translated_text}_")
                text_to_speech_with_pyttsx3(translated_text)
            else:
                st.error("❌ Translation failed. Please try again.")
        else:
            st.error("❌ Speech recognition failed. Please try again.")

    # Footer Information
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; font-size: 14px; color: grey;">
    Created with dedication by Noel, Sanin, and Rahul. Powered by Streamlit, HuggingFace, and Pyttsx3.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
