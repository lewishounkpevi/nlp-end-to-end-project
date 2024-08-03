# nlp-end-to-end-project

Un projet simple de mise en place d'un modèle NLP préentrainé de Hugging Face depuis la modélisation jusqu'au déploiement sur docker. ( A but éducatif).

Je le complèterai au fur et à mesure mais il reste toujours fonctionel sur le repo.


# Etapes pour prendre en main le projet

A exectuer avec un terminal 

## Installer le requiremenets.txt
```bash
make install
```

## Lancer le mdèle et l'enregister
```bash
mkdir models
```

```bash
python src/nlp.py
```

## Lancer en local

### Avec Fastapi sur le port 8000
```bash
python src/client_api.py
```
### Avec streamlit
```bash
streamlit run st_client_api.py
```

## build l'image Docker
```bash
docker build -t nlp-project:v1 .
```

## run l'image Docker
```bash
docker run -p 8000:8000 -p 8501:8501 nlp-project:v1
```
## Récupérer l'image directement depuis Dockerhub
```bash
docker pull lewisdumesnil/nlp-project:v1
```
