import streamlit as st
import requests

# URL de base de l'API
API_URL = "http://localhost:8000"

st.title("NLP Client avec Streamlit")

# Prétraitement du texte
st.header("Prétraitement du texte")
text_to_preprocess = st.text_area("Entrez le texte à prétraiter")
if st.button("Prétraiter"):
    preprocess_response = requests.post(f"{API_URL}/preprocess", json={"text": text_to_preprocess})
    if preprocess_response.status_code == 200:
        preprocessed_data = preprocess_response.json()
        st.write("Texte prétraité:", preprocessed_data["preprocessed_text"])
        st.write("Entités:", preprocessed_data["entities"])
    else:
        st.error("Erreur lors du prétraitement")

# Analyse des sentiments
st.header("Analyse des sentiments")
text_to_analyze = st.text_area("Entrez le texte à analyser")
if st.button("Analyser le sentiment"):
    sentiment_response = requests.post(f"{API_URL}/sentiment", json={"text": text_to_analyze})
    if sentiment_response.status_code == 200:
        sentiment_data = sentiment_response.json()
        st.write("Résultat de l'analyse des sentiments:", sentiment_data["sentiment"])
    else:
        st.error("Erreur lors de l'analyse des sentiments")

# Génération de texte
st.header("Génération de texte")
text_to_generate = st.text_area("Entrez le texte pour la génération")
if st.button("Générer un résumé"):
    generate_response = requests.post(f"{API_URL}/generate", json={"text": text_to_generate})
    if generate_response.status_code == 200:
        generated_data = generate_response.json()
        st.write("Résumé généré:", generated_data["summary"])
    else:
        st.error("Erreur lors de la génération du texte")
