from flask import Flask, request, render_template_string
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Charger le modèle et le vectorizer
model = joblib.load('/app/spam_classifier.pkl')
vectorizer = joblib.load('/app/vectorizer.pkl')

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
    message = request.form['message']
    vect_msg = vectorizer.transform([message])
    prediction = model.predict(vect_msg)
    result = "SPAM" if prediction[0] == 'spam' else "HAM"
    return f"<h3>Résultat : {result}</h3><a href='/'>↩ Retour</a>"

@app.route('/monitor')
def monitor():
    # Exemple de données de test pour la surveillance
    test_data = pd.DataFrame({
        'message': ['Free money!!!', 'How are you?', 'Win a prize now!', 'See you tomorrow'],
        'label': ['spam', 'ham', 'spam', 'ham']
    })
    
    X_test = vectorizer.transform(test_data['message'])
    y_true = test_data['label']
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_true, y_pred)
    return f"<h3>Accuracy en production : {accuracy:.2f}</h3>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
