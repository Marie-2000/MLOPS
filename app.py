from flask import Flask, request, render_template_string
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
import logging

# Initialiser Flask
app = Flask(__name__)

# Charger le modèle et le vectorizer
model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Télécharger les stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Fonction de nettoyage du texte
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return ' '.join([word for word in text.split() if word not in stop_words])

# Listes pour stocker les prédictions et les labels
predictions_list = []
labels_list = []

# Configuration du logger
logging.basicConfig(filename='monitor.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# HTML de base intégré
HTML_FORM = """
<!doctype html>
<title>Spam Classifier</title>
<h2>Entrez un message à classer :</h2>
<form method="post" action="/predict">
  <input type="text" name="message" style="width:300px;" required>
  <input type="submit" value="Classer">
</form>
"""

@app.route('/')
def home():
    return render_template_string(HTML_FORM)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        message = request.form['message']
        cleaned_message = clean_text(message)
        vect_msg = vectorizer.transform([cleaned_message])
        prediction = model.predict(vect_msg)
        result = "SPAM" if prediction[0] == 'spam' else "HAM"
        
        # Enregistrement dans les logs
        logging.info(f'Predicted: {result} for message: {message}')
        
        # Stocker la prédiction pour suivi de performance
        predictions_list.append(prediction[0])
        labels_list.append('spam' if result == "SPAM" else 'ham')

        return f"<h3>Résultat : {result}</h3><a href='/'>↩ Retour</a>"
    
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        return "<h3>Erreur lors du traitement du message. Veuillez réessayer.</h3>"

@app.route('/monitor')
def monitor():
    if len(predictions_list) == 0:
        return "<h3>Aucun message n'a encore été classé.</h3>"
    
    accuracy = accuracy_score(labels_list, predictions_list)
    logging.info(f'Accuracy en production : {accuracy:.2f}')
    return f"<h3>Accuracy en production : {accuracy:.2f}</h3>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
