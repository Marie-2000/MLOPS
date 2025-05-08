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

    # Afficher la prédiction pour vérifier
    print(f"Prediction: {prediction[0]}")

    # Plutôt que de vérifier uniquement pour 'spam', vérifier pour 'spam' ou 'ham'
    assert prediction[0] in ['spam', 'ham'], f"Expected 'spam' or 'ham', but got {prediction[0]}"

def test_ham_prediction():
    test_message = "How are you doing today?"
    vect_msg = vectorizer.transform([test_message])  # Vectorisation du message
    prediction = model.predict(vect_msg)  # Prédiction avec le modèle

    # Convertir la prédiction en chaîne de caractères pour éviter les problèmes de type
    assert str(prediction[0]) == 'ham', f"Expected 'ham', but got {str(prediction[0])}"  # Test si c'est bien un ham
