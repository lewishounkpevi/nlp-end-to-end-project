from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

app = FastAPI()

# Charger le modèle spaCy pour le prétraitement
nlp = spacy.load('fr_core_news_sm')

# Recharger le modèle et le tokenizer BERT pour l'analyse des sentiments
sentiment_model = AutoModelForSequenceClassification.from_pretrained("./models/sentiment_model")
sentiment_tokenizer = AutoTokenizer.from_pretrained("./models/sentiment_tokenizer")
sentiment_analyzer = pipeline('sentiment-analysis', model=sentiment_model, tokenizer=sentiment_tokenizer)

# Charger le modèle GPT pour la génération de texte
gpt_model = pipeline("text-generation", model="gpt2")

class Text(BaseModel):
    text: str

@app.post("/preprocess/")
def preprocess(text: Text):
    doc = nlp(text.text)
    tokens = [token.lemma_ for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return {"tokens": tokens, "entities": entities}

@app.post("/sentiment/")
def sentiment(text: Text):
    result = sentiment_analyzer(text.text)
    return result

@app.post("/generate/")
def generate(text: Text):
    result = gpt_model(text.text, max_length=50, num_return_sequences=1)
    return result[0]['generated_text']


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)