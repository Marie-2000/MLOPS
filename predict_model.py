import joblib
import re
import nltk
from nltk.corpus import stopwords

# Charger modèle et vectorizer
model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Télécharger les stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Fonction de nettoyage
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return ' '.join([word for word in text.split() if word not in stop_words])

# Entrée utilisateur
text = input("Entrez un message à classer : ")
cleaned_text = clean_text(text)

# Prédiction
text_tfidf = vectorizer.transform([cleaned_text])
prediction = model.predict(text_tfidf)

# Résultat
if prediction[0] == 'spam':
    print("Le message est classifié comme SPAM.")
else:
    print("Le message est classifié comme HAM.")
