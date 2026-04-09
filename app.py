from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Charger le modèle
MODEL_PATH = 'data/customer_churn_model.pkl'

# Vérifier si le modèle existe
if not os.path.exists(MODEL_PATH):
    print(f"⚠️  Le modèle n'existe pas à {MODEL_PATH}")
    print("Veuillez d'abord exécuter train.py pour entraîner le modèle")
    model = None
else:
    model = joblib.load(MODEL_PATH)
    print("✓ Modèle chargé avec succès")

@app.route('/')
def index():
    """Servir la page HTML"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint API pour faire une prédiction"""
    try:
        # Récupérer les données JSON
        data = request.get_json()
        
        # Valider les données
        required_fields = ['age', 'account_manager', 'years', 'num_sites']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Champ manquant: {field}'}), 400
        
        # Vérifier que le modèle est chargé
        if model is None:
            return jsonify({'error': 'Le modèle n\'est pas disponible. Exécutez d\'abord train.py'}), 500
        
        # Créer un DataFrame avec les données
        features = pd.DataFrame({
            'Age': [float(data['age'])],
            'Account_Manager': [float(data['account_manager'])],
            'Years': [float(data['years'])],
            'Num_Sites': [float(data['num_sites'])]
        })
        
        # Faire la prédiction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'prediction_text': 'Risque de churn' if prediction == 1 else 'Pas de churn',
            'probability_no_churn': float(probability[0]),
            'probability_churn': float(probability[1]),
            'confidence': float(max(probability)) * 100
        }), 200
    
    except ValueError as e:
        return jsonify({'error': f'Erreur de validation: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

@app.route('/api/info', methods=['GET'])
def info():
    """Obtenir des infos sur le modèle"""
    return jsonify({
        'model_status': 'Chargé' if model is not None else 'Non disponible',
        'features': ['Age', 'Account_Manager', 'Years', 'Num_Sites'],
        'target': 'Churn (0: Pas de churn, 1: Risque de churn)'
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
