import pytest
import pandas as pd
import os
from mcdonald_predictor import McDonaldPredictor
import joblib


@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'age': [25, 30, 35, 40, 17, 50],
        'gender': ['male', 'female', 'male', 'female', 'male', 'female'],
        'income': [5000.0, 6000.0, 4500.0, 7000.0, 3000.0, 8000.0],
        'visit_frequency': ['monthly', 'weekly', 'rarely', 'daily', 'monthly', 'rarely'],
        'satisfaction_level': ['medium', 'high', 'low', 'medium', 'high', 'low'],
        'liked_mcdonalds': [1, 1, 0, 1, 0, 0]
    })
    return data


@pytest.fixture
def predictor():
    if os.path.exists('model.joblib'):
        os.remove('model.joblib')
    return McDonaldPredictor()


class TestMcDonaldPredictor:
    
    def test_initialization(self, predictor):
        assert hasattr(predictor, 'model')
        assert hasattr(predictor, 'model_path')
        assert os.path.exists(predictor.model_path)
    
    def test_preprocess(self, predictor, sample_data):
        processed = predictor._preprocess(sample_data)
        assert len(processed) == 5
        assert all(processed['age'] >= 18)
        assert all(processed['visit_frequency'] > 0)
    
    def test_train_model(self, predictor):
        assert predictor.model is not None
        assert hasattr(predictor.model, 'predict')
        assert hasattr(predictor.model, 'predict_proba')
    
    def test_predict_single(self, predictor):
        features = {
            'age': 25,
            'gender': 'male',
            'income': 5000.0,
            'visit_frequency': 'monthly',
            'satisfaction_level': 'medium'
        }
        result = predictor.predict_single(features)
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'feature_importance' in result
        assert result['prediction'] in [0, 1]
        assert 0 <= result['confidence'] <= 1
        assert isinstance(result['feature_importance'], dict)
    
    def test_model_loading(self):
        if os.path.exists('model.joblib'):
            os.remove('model.joblib')
        
        predictor1 = McDonaldPredictor()
        assert os.path.exists('model.joblib')
        
        predictor2 = McDonaldPredictor()
        assert hasattr(predictor2, 'model')
        assert predictor2.model is not None
    
    def test_predict_single_different_inputs(self, predictor):
        test_cases = [
            {'age': 18, 'gender': 'male', 'income': 3000.0, 'visit_frequency': 'rarely', 'satisfaction_level': 'low'},
            {'age': 60, 'gender': 'female', 'income': 8000.0, 'visit_frequency': 'daily', 'satisfaction_level': 'high'},
            {'age': 30, 'gender': 'male', 'income': 6000.0, 'visit_frequency': 'weekly', 'satisfaction_level': 'medium'}
        ]
        for case in test_cases:
            result = predictor.predict_single(case)
            assert 'prediction' in result
            assert result['prediction'] in [0, 1]
