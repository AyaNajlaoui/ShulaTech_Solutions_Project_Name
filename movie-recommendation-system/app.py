import streamlit as st
import pandas as pd
import joblib

# Charger les données et le modèle
try:
    movies = pd.read_pickle('movies_data.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    cosine_sim = joblib.load('cosine_sim.pkl')
except FileNotFoundError:
    st.error("Fichiers nécessaires introuvables. Assurez-vous que movies_data.pkl, tfidf_vectorizer.pkl et cosine_sim.pkl sont dans le dossier.")

# Fonction de recommandation
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 10
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# CSS pour le style
st.markdown("""
    <style>
    .title { font-size: 40px; color: #2c3e50; text-align: center; font-family: 'Arial', sans-serif; margin-bottom: 20px; }
    .subtitle { font-size: 20px; color: #7f8c8d; text-align: center; font-family: 'Arial', sans-serif; margin-bottom: 30px; }
    .stSelectbox { font-size: 16px; font-family: 'Arial', sans-serif; }
    .result-box { 
        padding: 15px; 
        border-radius: 10px; 
        background-color: #34495e; 
        color: #ffffff; 
        text-align: center; 
        font-size: 18px; 
        font-family: 'Arial', sans-serif; 
        margin-top: 10px; 
    }
    .stButton>button { background-color: #3498db; color: white; border-radius: 10px; padding: 10px 20px; font-size: 18px; font-family: 'Arial', sans-serif; border: none; width: 100%; }
    .stButton>button:hover { background-color: #2980b9; }
    </style>
""", unsafe_allow_html=True)

# Interface Streamlit
def main():
    st.markdown('<h1 class="title">Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Select a movie and get recommendations based on its content!</p>', unsafe_allow_html=True)

    # Liste déroulante pour choisir un film
    movie_list = movies['title'].tolist()
    selected_movie = st.selectbox("Choose a movie:", movie_list)

    # Bouton pour obtenir les recommandations
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(selected_movie)
        st.write("### Recommended Movies:")
        for i, movie in enumerate(recommendations, 1):
            st.markdown(f'<div class="result-box">{i}. {movie}</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <hr>
        <p style='text-align: center; color: #7f8c8d; font-family: Arial, sans-serif;'>
            Built with ❤️ by Aya Najlaoui 
        </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()