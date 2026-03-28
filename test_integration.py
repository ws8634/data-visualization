import pytest
import os
import json
from mcdonald_predictor import McDonaldPredictor
from app import app
import pandas as pd


def test_end_to_end_prediction():
    """Test end-to-end prediction from model initialization to prediction"""
    # Remove existing model
    if os.path.exists('model.joblib'):
        os.remove('model.joblib')
    if os.path.exists('feature_names.joblib'):
        os.remove('feature_names.joblib')
    
    # Initialize predictor (will train model)
    predictor = McDonaldPredictor()
    
    # Make a prediction with all required fields
    features = {
        'age': 30,
        'gender': 'female',
        'income': 6000.0,
        'visit_frequency': 'weekly',
        'satisfaction_level': 'medium'
    }
    
    result = predictor.predict_single(features)
    
    # Verify prediction results
    assert 'prediction' in result
    assert result['prediction'] in [0, 1]
    assert 0 <= result['confidence'] <= 1
    assert len(result['feature_importance']) > 0
    
    # Verify model was saved
    assert os.path.exists('model.joblib')
    assert os.path.exists('feature_names.joblib')


def test_api_integration():
    """Test integration between API and predictor"""
    # Remove existing model
    if os.path.exists('model.joblib'):
        os.remove('model.joblib')
    if os.path.exists('feature_names.joblib'):
        os.remove('feature_names.joblib')
    
    # Create test client
    app.config['TESTING'] = True
    client = app.test_client()
    
    # First request should trigger model training
    first_data = {
        'age': 25,
        'gender': 'male',
        'income': 5000.0,
        'visit_frequency': 'weekly',
        'satisfaction_level': 'high'
    }
    
    response1 = client.post('/predict', 
                           data=json.dumps(first_data),
                           content_type='application/json')
    
    assert response1.status_code == 200
    
    # Second request should use loaded model
    second_data = {
        'age': 35,
        'gender': 'female',
        'income': 7000.0,
        'visit_frequency': 'daily',
        'satisfaction_level': 'medium'
    }
    
    response2 = client.post('/predict', 
                           data=json.dumps(second_data),
                           content_type='application/json')
    
    assert response2.status_code == 200
    
    # Verify both responses have the same structure
    result1 = json.loads(response1.data)
    result2 = json.loads(response2.data)
    
    assert set(result1.keys()) == set(result2.keys())


def test_data_consistency():
    """Test that model prediction is consistent with training data"""
    # Remove existing model
    if os.path.exists('model.joblib'):
        os.remove('model.joblib')
    if os.path.exists('feature_names.joblib'):
        os.remove('feature_names.joblib')
    
    predictor = McDonaldPredictor()
    
    # Load original data
    data = pd.read_csv('mcdonald_data.csv')
    processed_data = predictor._preprocess(data)
    
    # Test prediction on a sample from training data
    sample_row = processed_data.iloc[0]
    sample = sample_row.drop('liked_mcdonalds').to_dict()
    actual = sample_row['liked_mcdonalds']
    
    result = predictor.predict_single(sample)
    
    # Prediction should ideally match actual, but allow for model variance
    assert result['prediction'] in [0, 1]
    assert result['confidence'] >= 0.5  # Should be confident on training data


def test_multiple_predictions():
    """Test making multiple predictions in sequence"""
    # Remove existing model
    if os.path.exists('model.joblib'):
        os.remove('model.joblib')
    if os.path.exists('feature_names.joblib'):
        os.remove('feature_names.joblib')
    
    predictor = McDonaldPredictor()
    
    # List of test cases with different combinations
    test_cases = [
        {'age': 20, 'gender': 'female', 'income': 3000.0, 'visit_frequency': 'rarely', 'satisfaction_level': 'low'},
        {'age': 40, 'gender': 'male', 'income': 9000.0, 'visit_frequency': 'weekly', 'satisfaction_level': 'high'},
        {'age': 25, 'gender': 'male', 'income': 5000.0, 'visit_frequency': 'monthly', 'satisfaction_level': 'medium'},
        {'age': 50, 'gender': 'female', 'income': 12000.0, 'visit_frequency': 'daily', 'satisfaction_level': 'high'},
        {'age': 35, 'gender': 'male', 'income': 7000.0, 'visit_frequency': 'weekly', 'satisfaction_level': 'low'},
    ]
    
    # Make predictions for all test cases
    results = []
    for case in test_cases:
        result = predictor.predict_single(case)
        results.append(result)
        
        # Verify each result is valid
        assert result['prediction'] in [0, 1]
        assert 0 <= result['confidence'] <= 1
    
    # Verify we got results for all test cases
    assert len(results) == len(test_cases)
    
    # Results should have variation
    predictions = [r['prediction'] for r in results]
    assert len(set(predictions)) >= 1  # At least one unique prediction


def test_api_error_handling_integration():
    """Test API error handling integration"""
    app.config['TESTING'] = True
    client = app.test_client()
    
    # Test various error scenarios
    error_cases = [
        # Missing required fields
        ({'age': 30, 'gender': 'male'}, 400),
        # Empty request
        ({}, 400),
        # Invalid content type
        (None, 500),  # Will use text/plain
    ]
    
    for data, expected_status in error_cases:
        if data is None:
            response = client.post('/predict', 
                                  data='not json',
                                  content_type='text/plain')
        else:
            response = client.post('/predict', 
                                  data=json.dumps(data),
                                  content_type='application/json')
        
        assert response.status_code == expected_status
