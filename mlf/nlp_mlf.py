import spacy
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import mlflow
import mlflow.pyfunc
import torch

# Charger le modèle spaCy pour le prétraitement
nlp = spacy.load('fr_core_news_sm')

# Charger le modèle BERT pour l'analyse des sentiments en français
sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_analyzer = pipeline('sentiment-analysis', model=sentiment_model, tokenizer=sentiment_tokenizer)

# Charger le modèle GPT pour la génération de texte
# gpt_model = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
gpt_model = pipeline("text-generation", model="gpt2")

# Classe personnalisée pour encapsuler les modèles
class NLPModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.nlp = nlp
        self.sentiment_analyzer = sentiment_analyzer
        self.gpt_model = gpt_model

    def preprocess_text(self, text):
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc]
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return " ".join(tokens), entities

    def analyze_sentiment(self, text):
        result = self.sentiment_analyzer(text)
        return result

    def generate_summary(self, text):
        result = self.gpt_model(text, max_length=50, num_return_sequences=1)
        return result[0]['generated_text']

    def predict(self, context, model_input):
        task = model_input['task']
        text = model_input['text']
        if task == 'preprocess':
            return self.preprocess_text(text)
        elif task == 'sentiment':
            return self.analyze_sentiment(text)
        elif task == 'generate':
            return self.generate_summary(text)
        else:
            return "Invalid task"

# Enregistrer le modèle NLP avec MLflow
mlflow.set_experiment("NLP_Models")
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=NLPModel(),
        conda_env=None
    )

