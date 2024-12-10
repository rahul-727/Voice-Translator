import os
import streamlit as st
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
import uuid  # To generate unique filenames
from playsound import playsound

# Supported languages
LANGUAGES = {
    "Afrikaans": "af",
    "Albanian": "sq",
    "Amharic": "am",
    "Arabic": "ar",
    "Armenian": "hy",
    "Azerbaijani": "az",
    "Basque": "eu",
    "Belarusian": "be",
    "Bengali": "bn",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Cebuano": "ceb",
    "Chinese (Simplified)": "zh-cn",
    "Chinese (Traditional)": "zh-tw",
    "Corsican": "co",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Esperanto": "eo",
    "Estonian": "et",
    "Filipino": "tl",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Gujarati": "gu",
    "Haitian Creole": "ht",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Kannada": "kn",
    "Korean": "ko",
    "Latin": "la",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Malay": "ms",
    "Malayalam": "ml",
    "Maltese": "mt",
    "Norwegian": "no",
    "Polish": "pl",
    "Portuguese": "pt",
    "Punjabi": "pa",
    "Romanian": "ro",
    "Russian": "ru",
    "Spanish": "es",
    "Swedish": "sv",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Vietnamese": "vi",
    "Zulu": "zu",
}

# Function to translate and play speech
def translate_and_speak(text, target_language):
    try:
        # Translate text
        translated_text = GoogleTranslator(source="auto", target=target_language).translate(text)
        st.success(f"Translated Text: {translated_text}")

        # Generate unique audio file name
        audio_file = f"translated_speech_{uuid.uuid4().hex}.mp3"

        # Convert to speech
        tts = gTTS(translated_text, lang=target_language)
        tts.save(audio_file)

        # Play the speech
        playsound(audio_file)

        # Clean up audio file
        os.remove(audio_file)
    except Exception as e:
        st.error(f"Error: {e}")

# Main function to run the Streamlit app
def main():
    st.title("Speech Translator")

    st.markdown("### Select Target Language:")
    target_language = st.selectbox("Choose a language", list(LANGUAGES.keys()))

    if st.button("Start Recording"):
        st.info("Listening for 5 seconds... Speak now!")
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source, timeout=5)
                st.info("Processing your input...")
                captured_text = recognizer.recognize_google(audio)
                st.success(f"Recognized Speech: {captured_text}")
                translate_and_speak(captured_text, LANGUAGES[target_language])
            except sr.WaitTimeoutError:
                st.error("Listening timed out. Please try again.")
            except sr.UnknownValueError:
                st.error("Could not understand your speech. Please try again.")
            except sr.RequestError as e:
                st.error(f"Could not request results; {e}")

if __name__ == "__main__":
    main()
