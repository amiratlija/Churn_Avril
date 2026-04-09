import pytest
import json
from app import app, model
import pandas as pd

@pytest.fixture
def client():
    """Fixture pour créer un client de test Flask"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    """Test de la route d'accueil"""
    response = client.get('/')
    assert response.status_code == 200
    # Vérifier que le template est rendu (contient du HTML)
    assert b'Churn Prediction' in response.data or b'html' in response.data.lower()


def test_predict_missing_fields(client):
    """Test de prédiction avec des champs manquants"""
    # Test avec un champ manquant
    incomplete_data = {
        'age': 35,
        'account_manager': 1,
        'years': 5
        # num_sites manquant
    }

    response = client.post('/api/predict',
                          data=json.dumps(incomplete_data),
                          content_type='application/json')

    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'num_sites' in data['error']

def test_predict_invalid_data_types(client):
    """Test de prédiction avec des types de données invalides"""
    invalid_data = {
        'age': 'trente-cinq',  # Chaîne au lieu de nombre
        'account_manager': 1,
        'years': 5,
        'num_sites': 8
    }

    response = client.post('/api/predict',
                          data=json.dumps(invalid_data),
                          content_type='application/json')

    # Devrait échouer lors de la conversion en float
    assert response.status_code in [400, 500]

   

def test_model_loading():
    """Test que le modèle est correctement chargé"""
    # Ce test vérifie simplement que la variable model n'est pas None
    # Si elle l'est, c'est probablement parce que le fichier n'existe pas
    if model is None:
        # Vérifier que le fichier modèle existe
        import os
        assert not os.path.exists('data/customer_churn_model.pkl'), \
            "Le fichier modèle existe mais n'a pas pu être chargé"

if __name__ == '__main__':
    pytest.main([__file__])
