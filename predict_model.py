import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Charger le modèle et le vectorizer sauvegardés
model = joblib.load(r'C:/Users/marie/MLOPS/spam_classifier.pkl')
vectorizer = joblib.load(r'C:/Users/marie/MLOPS/vectorizer.pkl')

# Exemple de texte à prédire
text = input("Entrez un message à classer : ")

# Préparer le texte
text_tfidf = vectorizer.transform([text])  # Appliquer la transformation TF-IDF

# Faire la prédiction
prediction = model.predict(text_tfidf)

# Afficher la prédiction (spam ou ham)
if prediction[0] == 'spam':
    print("Le message est classifié comme SPAM.")
else:
    print("Le message est classifié comme HAM.")
