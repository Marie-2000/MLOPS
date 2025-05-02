# Étape 1: Utiliser une image de base Python officielle
FROM python:3.9-slim

# Étape 2: Définir le répertoire de travail à l'intérieur du conteneur
WORKDIR /app

# Étape 3: Copier les fichiers nécessaires dans le conteneur
COPY . /app
# (Assurez-vous que spam_classifier.pkl est dans le même répertoire que votre Dockerfile)
#COPY spam_classifier.pkl /app/spam_classifier.pkl

# Étape 4: Installer les dépendances via le fichier requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5: Spécifier la commande à exécuter lorsqu'un conteneur est lancé
CMD ["python", "app.py"]
