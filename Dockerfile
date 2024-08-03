# Utiliser une image de base Python
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app
RUN mkdir models

# Installer les dépendances
COPY src/requirements.txt .
RUN pip install -r requirements.txt

# Télécharger le modèle spaCy français
RUN python -m spacy download fr_core_news_sm

# Copier les fichiers du projet
COPY ./src .
COPY ./models ./models

# Copier le fichier de configuration supervisord
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Exposer les ports pour FastAPI et Streamlit
EXPOSE 8000 8501

# Démarrer les services via supervisord
CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]