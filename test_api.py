import pytest
from app import app
import json


@pytest.fixture
def client():
    """Fixture to create a test client for the Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """Test the index route returns a 200 OK response"""
    response = client.get('/')
    assert response.status_code == 200


def test_predict_route_valid_input(client):
    """Test the predict route with valid input matching training data format"""
    valid_data = {
        'age': 25,
        'gender': 'male',
        'income': 5000.0,
        'visit_frequency': 'weekly',
        'satisfaction_level': 'high'
    }
    
    response = client.post('/predict', 
                          data=json.dumps(valid_data),
                          content_type='application/json')
    
    assert response.status_code == 200
    
    result = json.loads(response.data)
    assert 'prediction' in result
    assert 'confidence' in result
    assert 'feature_importance' in result
    assert result['prediction'] in [0, 1]
    assert 0 <= result['confidence'] <= 1


def test_predict_route_invalid_content_type(client):
    """Test predict route with invalid content type"""
    response = client.post('/predict', 
                          data='not json',
                          content_type='text/plain')
    
    # Flask returns 415 for unsupported media types
    assert response.status_code in [400, 415, 500]


def test_predict_route_missing_fields(client):
    """Test predict route with missing fields"""
    incomplete_data = {
        'age': 25,
        'gender': 'male'
        # Missing other fields
    }
    
    response = client.post('/predict', 
                          data=json.dumps(incomplete_data),
                          content_type='application/json')
    
    # Should return 400 with error message
    assert response.status_code == 400
    result = json.loads(response.data)
    assert 'error' in result


def test_predict_route_empty_body(client):
    """Test predict route with empty request body"""
    response = client.post('/predict', 
                          data=json.dumps({}),
                          content_type='application/json')
    
    # Should return 400 with error message
    assert response.status_code == 400
    result = json.loads(response.data)
    assert 'error' in result


def test_predict_route_invalid_http_method(client):
    """Test predict route with invalid HTTP method"""
    response = client.get('/predict')
    assert response.status_code == 405  # Method Not Allowed


def test_predict_route_different_categories(client):
    """Test predict route with different categorical values"""
    test_cases = [
        {'age': 30, 'gender': 'female', 'income': 6000.0, 'visit_frequency': 'monthly', 'satisfaction_level': 'medium'},
        {'age': 45, 'gender': 'male', 'income': 8000.0, 'visit_frequency': 'daily', 'satisfaction_level': 'low'},
        {'age': 20, 'gender': 'female', 'income': 3000.0, 'visit_frequency': 'rarely', 'satisfaction_level': 'high'}
    ]
    
    for test_data in test_cases:
        response = client.post('/predict', 
                              data=json.dumps(test_data),
                              content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result['prediction'] in [0, 1]
        assert 0 <= result['confidence'] <= 1
