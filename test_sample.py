import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Charger le modèle et le vectorizer sauvegardés
model = joblib.load(r'C:/Users/marie/MLOPS/spam_classifier.pkl')
vectorizer = joblib.load(r'C:/Users/marie/MLOPS/vectorizer.pkl')

# Exemple de message à tester
def test_spam_prediction():
    test_message = "Congratulations! You've won a free ticket to Bahamas."
    vect_msg = vectorizer.transform([test_message])  # Vectorisation du message
    prediction = model.predict(vect_msg)  # Prédiction avec le modèle

    # Convertir la prédiction en chaîne de caractères pour éviter les problèmes de type
    assert str(prediction[0]) == 'spam', f"Expected 'spam', but got {str(prediction[0])}"  # Test si c'est bien un spam

def test_ham_prediction():
    test_message = "How are you doing today?"
    vect_msg = vectorizer.transform([test_message])  # Vectorisation du message
    prediction = model.predict(vect_msg)  # Prédiction avec le modèle

    # Convertir la prédiction en chaîne de caractères pour éviter les problèmes de type
    assert str(prediction[0]) == 'ham', f"Expected 'ham', but got {str(prediction[0])}"  # Test si c'est bien un ham
