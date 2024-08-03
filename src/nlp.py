import spacy
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
# !python3.11 -m spacy download fr_core_news_sm

# Charger le modèle spaCy pour le prétraitement
nlp = spacy.load('fr_core_news_sm')

# Charger le modèle BERT pour l'analyse des sentiments en français
sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_analyzer = pipeline('sentiment-analysis', model=sentiment_model, tokenizer=sentiment_tokenizer)

# Charger le modèle GPT pour la génération de texte
gpt_model = pipeline("text-generation", model="gpt2")

# Fonction pour prétraiter le texte avec spaCy
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return " ".join(tokens), entities


# Fonction pour analyser les sentiments avec BERT
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result



# Fonction pour générer un résumé ou une réponse avec GPT
def generate_summary(text):
    result = gpt_model(text, max_length=50, num_return_sequences=1)
    return result[0]['generated_text']

# Sauvegarder le modèle BERT pour l'analyse des sentiments
sentiment_model.save_pretrained("./models/sentiment_model")
sentiment_tokenizer.save_pretrained("./models/sentiment_tokenizer")



# Exemple d'utilisation
if __name__ == "__main__":
    reviews = [
        "J'adore le nouveau design du produit ! Il est incroyable et fonctionne parfaitement.",
        "Le service était terrible et j'ai dû attendre des heures. Pas recommandé.",
        "Super expérience, l'équipe était très serviable et le produit vaut chaque centime."
    ]

    for review in reviews:
        print(f"Avis original : {review}")
        
        preprocessed_text, entities = preprocess_text(review)
        print(f"Texte prétraité : {preprocessed_text}")
        print(f"Entités : {entities}")
        
        sentiment = analyze_sentiment(review)
        print(f"Sentiment : {sentiment}")
        
        summary = generate_summary(review)
        print(f"Résumé généré : {summary}")
        print("-" * 50)

