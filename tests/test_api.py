import pytest
import json
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestAPI:
    
    def test_index_route(self, client):
        response = client.get('/')
        assert response.status_code == 200
    
    def test_predict_route_post(self, client):
        data = {
            'age': 25,
            'gender': 'male',
            'income': 5000.0,
            'visit_frequency': 'monthly',
            'satisfaction_level': 'medium'
        }
        response = client.post('/predict', 
                              data=json.dumps(data),
                              content_type='application/json')
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'feature_importance' in result
    
    def test_predict_route_get_method(self, client):
        response = client.get('/predict')
        assert response.status_code == 405
    
    def test_predict_route_invalid_data(self, client):
        data = {
            'age': 'invalid',
            'gender': 'male',
            'income': 5000.0,
            'visit_frequency': 'monthly',
            'satisfaction_level': 'medium'
        }
        response = client.post('/predict', 
                              data=json.dumps(data),
                              content_type='application/json')
        assert response.status_code in [400, 500]
    
    def test_predict_route_missing_fields(self, client):
        data = {
            'age': 25
        }
        response = client.post('/predict', 
                              data=json.dumps(data),
                              content_type='application/json')
        assert response.status_code in [400, 500]
    
    def test_predict_response_structure(self, client):
        data = {
            'age': 30,
            'gender': 'female',
            'income': 6000.0,
            'visit_frequency': 'weekly',
            'satisfaction_level': 'high'
        }
        response = client.post('/predict', 
                              data=json.dumps(data),
                              content_type='application/json')
        assert response.status_code == 200
        result = json.loads(response.data)
        assert isinstance(result['prediction'], int)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['feature_importance'], dict)
        assert 0 <= result['confidence'] <= 1
        assert result['prediction'] in [0, 1]
