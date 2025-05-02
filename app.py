from flask import Flask, request, render_template_string
import joblib
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Charger le modèle et le vectorizer
model = joblib.load('/app/spam_classifier.pkl')
vectorizer = joblib.load('/app/vectorizer.pkl')

# HTML de base intégré (pas besoin de fichier externe)
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
    
    # Prédiction
    prediction = model.predict(vect_msg)
    result = "SPAM" if prediction[0] == 'spam' else "HAM"
    
    # Calcul de l'accuracy si vous avez des données de vérité terrain
    # Exemple avec une vérité terrain fictive. Remplacez-la avec des vraies étiquettes si disponible.
    y_true = ['spam']  # Remplacez ceci par la vraie étiquette dans un cas réel
    accuracy = accuracy_score(y_true, prediction)
    
    # Affichage de l'accuracy dans la page
    return f"<h3>Résultat : {result}</h3><p>Accuracy: {accuracy}</p><a href='/'>↩ Retour</a>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
