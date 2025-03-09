Fake News Detector
Overview
The Fake News Detector is a web application designed to classify news articles as either "Fake" or "True" using machine learning. Built with Python, this project leverages an XGBoost model trained on a dataset of news articles, combined with natural language processing (NLP) techniques to preprocess and analyze text. The application is powered by Streamlit, providing an intuitive and interactive user interface where users can input news text and receive instant predictions.

Features
Text Classification: Predicts whether a news article is fake or true based on its content.
NLP Preprocessing: Utilizes NLTK for tokenization, stopword removal, and text cleaning to prepare input data.
Interactive UI: A sleek, user-friendly interface built with Streamlit, featuring custom styling for an engaging experience.
Model: Employs a pre-trained XGBoost classifier with TF-IDF vectorization for accurate predictions.
Easy Deployment: Runs locally with a simple streamlit run app.py command.
Technologies Used
Python: Core programming language.
XGBoost: Machine learning model for classification.
Scikit-learn: For TF-IDF vectorization and model utilities.
NLTK: For text preprocessing (tokenization, stopwords).
Streamlit: Framework for building the web application.
Pandas & NumPy: Data manipulation and processing.
Joblib: Model persistence.