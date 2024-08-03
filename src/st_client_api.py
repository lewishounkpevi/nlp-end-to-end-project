import streamlit as st
import requests

# Définir l'URL de base de l'API FastAPI
API_BASE_URL = "http://localhost:8000"

def call_api(endpoint, text):
    response = requests.post(f"{API_BASE_URL}/{endpoint}/", json={"text": text})
    return response.json()

def main():
    st.title("Portail d'analyse de l'avis du client")

    # Boîte de texte pour la saisie de l'utilisateur
    text = st.text_area("Entrez votre Commentaire:", height=150)
    
    if st.button("Points Clés"):
        result = call_api("preprocess", text)
        st.write("mots clés:")
        st.json(result)

    if st.button("Sentiment du client"):
        result = call_api("sentiment", text)
        st.write("Analyse des sentiments:")
        st.json(result)

    if st.button("Reformuler"):
        result = call_api("generate", text)
        st.write("Reformulation:")
        st.write(result)

if __name__ == "__main__":
    main()
