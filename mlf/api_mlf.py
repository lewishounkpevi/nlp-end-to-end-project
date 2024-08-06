from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import os

app = FastAPI()

class TextInput(BaseModel):
    text: str

# Charger le modèle depuis le chemin correct
# model_path = os.path.join("mlruns", "0", "5ab651091e2243a5b7fbd8da3f129df7", "artifacts", "nlp_model")
model_path = 'runs:/5ab651091e2243a5b7fbd8da3f129df7/model'
model = mlflow.pyfunc.load_model(model_path)

@app.post("/preprocess")
def preprocess(input: TextInput):
    result = model.predict({"task": "preprocess", "text": input.text})
    return {"preprocessed_text": result[0], "entities": result[1]}

@app.post("/sentiment")
def sentiment(input: TextInput):
    result = model.predict({"task": "sentiment", "text": input.text})
    return {"sentiment": result}

@app.post("/generate")
def generate(input: TextInput):
    result = model.predict({"task": "generate", "text": input.text})
    return {"summary": result}

# Client pour tester l'API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

