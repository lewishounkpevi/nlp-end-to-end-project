# import mlflow
# import mlflow.pyfunc
# import os
# from datetime import datetime

# # Télécharger le modèle spaCy pour le prétraitement si ce n'est pas déjà fait
# os.system("python -m spacy download fr_core_news_sm")

# # Définir un chemin unique pour le modèle
# timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
# model_path = f"nlp_model_{timestamp}"

# mlflow.set_experiment("NLP_Experiment")

# class NLPModel(mlflow.pyfunc.PythonModel):
#     def load_context(self, context):
#         import spacy
#         from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

#         # Charger le modèle spaCy pour le prétraitement
#         self.nlp = spacy.load('fr_core_news_sm')

#         # Charger le modèle BERT pour l'analyse des sentiments en français
#         self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
#         self.sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
#         self.sentiment_analyzer = pipeline('sentiment-analysis', model=self.sentiment_model, tokenizer=self.sentiment_tokenizer)

#         # Charger le modèle GPT pour la génération de texte
#         self.gpt_model = pipeline("text-generation", model="gpt2")

#     def preprocess_text(self, text):
#         doc = self.nlp(text)
#         tokens = [token.lemma_ for token in doc]
#         entities = [(ent.text, ent.label_) for ent in doc.ents]
#         return " ".join(tokens), entities

#     def analyze_sentiment(self, text):
#         result = self.sentiment_analyzer(text)
#         return result

#     def generate_summary(self, text):
#         result = self.gpt_model(text, max_length=50, num_return_sequences=1)
#         return result[0]['generated_text']

#     def predict(self, context, model_input):
#         text = model_input["text"]
#         preprocessed_text, entities = self.preprocess_text(text)
#         sentiment = self.analyze_sentiment(text)
#         summary = self.generate_summary(text)
#         return {"preprocessed_text": preprocessed_text, "entities": entities, "sentiment": sentiment, "summary": summary}

# # Enregistrer le modèle NLP avec MLflow
# if __name__ == "__main__":
#     with mlflow.start_run():
#         mlflow.pyfunc.save_model(path=model_path, python_model=NLPModel())


import mlflow
import mlflow.pyfunc
import os
import shutil

# Télécharger le modèle spaCy pour le prétraitement si ce n'est pas déjà fait
# os.system("python -m spacy download fr_core_news_sm")

# Définir le chemin du modèle
model_path = "nlp_model2"

# Supprimer le répertoire existant s'il existe
if os.path.exists(model_path):
    shutil.rmtree(model_path)

mlflow.set_experiment("NLP_Experiment")

class NLPModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import spacy
        from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

        # Charger le modèle spaCy pour le prétraitement
        self.nlp = spacy.load('fr_core_news_sm')

        # Charger le modèle BERT pour l'analyse des sentiments en français
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.sentiment_analyzer = pipeline('sentiment-analysis', model=self.sentiment_model, tokenizer=self.sentiment_tokenizer)

        # Charger le modèle GPT pour la génération de texte
        self.gpt_model = pipeline("text-generation", model="gpt2")

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

    def predict(self, model_input):
        # Vérifier que la clé "text" existe dans model_input
        if "text" not in model_input:
            raise ValueError("The input data must contain a 'text' key")
        
        text = model_input["text"]
        preprocessed_text, entities = self.preprocess_text(text)
        sentiment = self.analyze_sentiment(text)
        summary = self.generate_summary(text)
        return {"preprocessed_text": preprocessed_text, "entities": entities, "sentiment": sentiment, "summary": summary}

# Enregistrer le modèle NLP avec MLflow
if __name__ == "__main__":
    with mlflow.start_run():
        mlflow.pyfunc.save_model(
            path=model_path,
            python_model=NLPModel(),
            conda_env={
                'channels': ['defaults'],
                'dependencies': [
                    'python=3.8.0',
                    'pip',
                    {
                        'pip': [
                            'mlflow',
                            'numpy==1.26.4',
                            'spacy==3.7.5',
                            'transformers==4.43.3',
                            'torch==2.4.0',
                            'thinc==8.2.2',
                            'cloudpickle==2.2.1',
                        ],
                    },
                ],
                'name': 'mlflow-env'
            }
        )
    
    # Charger le modèle pour le test de prédiction
    nlp_model = mlflow.pyfunc.load_model(model_path)
    
    # Effectuer un test de prédiction
    test_input = {"text": "J'adore le nouveau design du produit !"}
    prediction = nlp_model.predict(test_input)
    print("Test de prédiction :", prediction)






