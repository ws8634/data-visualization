import pytest
import json
import os
from app import app
from mcdonald_predictor import McDonaldPredictor


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def cleanup_model():
    yield
    if os.path.exists('model.joblib'):
        os.remove('model.joblib')


class TestIntegration:
    
    def test_full_workflow(self, client):
        if os.path.exists('model.joblib'):
            os.remove('model.joblib')
        
        predictor = McDonaldPredictor()
        assert os.path.exists('model.joblib')
        
        features = {
            'age': 25,
            'gender': 'male',
            'income': 5000.0,
            'visit_frequency': 'monthly',
            'satisfaction_level': 'medium'
        }
        
        direct_result = predictor.predict_single(features)
        
        api_response = client.post('/predict', 
                                  data=json.dumps(features),
                                  content_type='application/json')
        assert api_response.status_code == 200
        api_result = json.loads(api_response.data)
        
        assert direct_result['prediction'] == api_result['prediction']
        assert abs(direct_result['confidence'] - api_result['confidence']) < 0.001
    
    def test_model_reusability(self, client):
        if os.path.exists('model.joblib'):
            os.remove('model.joblib')
        
        predictor1 = McDonaldPredictor()
        features1 = {
            'age': 30,
            'gender': 'male',
            'income': 6000.0,
            'visit_frequency': 'weekly',
            'satisfaction_level': 'medium'
        }
        result1 = predictor1.predict_single(features1)
        
        predictor2 = McDonaldPredictor()
        result2 = predictor2.predict_single(features1)
        
        assert result1['prediction'] == result2['prediction']
        assert abs(result1['confidence'] - result2['confidence']) < 0.001
    
    def test_multiple_predictions_consistency(self, client):
        test_cases = [
            {'age': 18, 'gender': 'male', 'income': 3000.0, 'visit_frequency': 'rarely', 'satisfaction_level': 'low'},
            {'age': 25, 'gender': 'female', 'income': 4500.0, 'visit_frequency': 'monthly', 'satisfaction_level': 'medium'},
            {'age': 40, 'gender': 'male', 'income': 7000.0, 'visit_frequency': 'weekly', 'satisfaction_level': 'high'},
            {'age': 60, 'gender': 'female', 'income': 8000.0, 'visit_frequency': 'daily', 'satisfaction_level': 'high'}
        ]
        
        predictor = McDonaldPredictor()
        
        for case in test_cases:
            direct_result = predictor.predict_single(case)
            api_response = client.post('/predict', 
                                      data=json.dumps(case),
                                      content_type='application/json')
            assert api_response.status_code == 200
            api_result = json.loads(api_response.data)
            assert direct_result['prediction'] == api_result['prediction']
