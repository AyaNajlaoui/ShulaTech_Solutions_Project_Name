import streamlit as st
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import xgboost as xgb

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')  # Ajout de cette ligne
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the trained model and vectorizer
try:
    best_model = joblib.load('xgboost_model.pkl')
    vectorization = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure they are saved correctly.")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[\*.,!?;:()"\'-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Prediction function
def predict_news(text):
    text_cleaned = preprocess_text(text)
    text_vectorized = vectorization.transform([text_cleaned])
    prediction = best_model.predict(text_vectorized)
    return "Fake News" if prediction[0] == 0 else "True News"

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        color: #2c3e50;
        text-align: center;
        font-family: 'Arial', sans-serif;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 20px;
        color: #7f8c8d;
        text-align: center;
        font-family: 'Arial', sans-serif;
        margin-bottom: 30px;
    }
    .stTextArea textarea {
        border: 2px solid #3498db;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        font-family: 'Arial', sans-serif;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-family: 'Arial', sans-serif;
        margin-top: 20px;
    }
    .fake {
        background-color: #e74c3c;
        color: white;
    }
    .true {
        background-color: #2ecc71;
        color: white;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 18px;
        font-family: 'Arial', sans-serif;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app layout
def main():
    st.markdown('<h1 class="title">Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Enter a news article below to check if it’s fake or true!</p>', unsafe_allow_html=True)

    # Text input area
    user_input = st.text_area("Paste your news article here:", height=200)

    # Predict button
    if st.button("Check News"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            result = predict_news(user_input)
            # Display result with styled box
            if result == "Fake News":
                st.markdown(f'<div class="result-box fake">{result}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box true">{result}</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <hr>
        <p style='text-align: center; color: #7f8c8d; font-family: Arial, sans-serif;'>
            Built with ❤️ by Aya
        </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()