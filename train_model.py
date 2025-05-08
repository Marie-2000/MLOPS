import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Assurez-vous que NLTK est prêt pour le traitement
nltk.download('stopwords')

# Lire le fichier CSV en spécifiant que les colonnes sont séparées par des points-virgules
data = pd.read_csv("spam.csv", encoding="ISO-8859-1", sep=";", on_bad_lines='skip')

# Sélectionner uniquement les colonnes pertinentes (par exemple, 'v1' et 'v2')
data = data[['v1', 'v2']]

# Vérifier les premières lignes
print(data.head())

# Supprimer les lignes où la colonne 'v2' est vide (NaN)
data = data.dropna(subset=['v2'])

# Nettoyage des textes
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Supprimer tout sauf les lettres et les espaces
    text = text.lower()  # Convertir en minuscules
    return text

# Appliquer le nettoyage aux messages
data['v2'] = data['v2'].apply(clean_text)

# Suppression des mots vides (stopwords)
stop_words = set(stopwords.words('english'))  # Liste des mots vides en anglais
data['v2'] = data['v2'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Diviser les données en 80% pour l'entraînement et 20% pour le test
X = data['v2']  # Les messages (features)
y = data['v1']  # Les étiquettes (spam/ham)

# Diviser les données en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorisation des textes
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)  # Transformer les messages d'entraînement
X_test_tfidf = vectorizer.transform(X_test)  # Transformer les messages de test

# Entraînement du modèle Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Sauvegarde du modèle et du vectorizer
joblib.dump(model, r'C:/Users/marie/MLOPS/spam_classifier.pkl')  # Sauvegarder le modèle dans le dossier MLOPS
joblib.dump(vectorizer, r'C:/Users/marie/MLOPS/vectorizer.pkl')  # Sauvegarder le vectorizer dans le dossier MLOPS
print("Modèle et vectorizer sauvegardés sous 'MLOPS/spam_classifier.pkl' et 'MLOPS/vectorizer.pkl'")

# Évaluation du modèle
from sklearn.metrics import classification_report
y_pred = model.predict(X_test_tfidf)  # Prédire les labels pour les données de test
print(classification_report(y_test, y_pred))  # Afficher les performances du modèle
