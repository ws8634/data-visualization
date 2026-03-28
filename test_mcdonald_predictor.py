import pytest
import pandas as pd
import os
from mcdonald_predictor import McDonaldPredictor


@pytest.fixture(scope="module")
def predictor():
    """Fixture to create a McDonaldPredictor instance for testing"""
    # Remove existing model to force retraining for consistent testing
    if os.path.exists('model.joblib'):
        os.remove('model.joblib')
    if os.path.exists('feature_names.joblib'):
        os.remove('feature_names.joblib')
    return McDonaldPredictor()


@pytest.fixture
def sample_features():
    """Fixture to provide sample input features that match training data format"""
    return {
        'age': 25,
        'gender': 'male',
        'income': 5000.0,
        'visit_frequency': 'weekly',
        'satisfaction_level': 'high'
    }


def test_init_model_loading(predictor):
    """Test that model initializes correctly and loads or trains model"""
    assert predictor is not None
    assert hasattr(predictor, 'model')
    assert hasattr(predictor, 'feature_names')
    assert os.path.exists('model.joblib')
    assert os.path.exists('feature_names.joblib')


def test_preprocess():
    """Test data preprocessing functionality"""
    predictor = McDonaldPredictor()
    
    # Create test data with invalid entries
    test_data = pd.DataFrame({
        'age': [17, 25, 30, 16],
        'gender': ['male', 'female', 'male', 'female'],
        'income': [5000, 6000, 7000, 4000],
        'visit_frequency': ['monthly', 'weekly', 'daily', 'rarely'],
        'satisfaction_level': ['low', 'medium', 'high', 'medium'],
        'liked_mcdonalds': [0, 1, 1, 0]
    })
    
    processed = predictor._preprocess(test_data)
    
    # Should filter out under 18 and drop NA
    assert len(processed) == 2
    assert all(processed['age'] >= 18)
    assert all(processed['visit_frequency'].isin(['weekly', 'daily']))


def test_train_model(predictor):
    """Test model training functionality"""
    # Model should already be trained by fixture
    assert hasattr(predictor.model.named_steps['classifier'], 'feature_importances_')
    
    # Check pipeline components
    assert 'scaler' in predictor.model.named_steps
    assert 'classifier' in predictor.model.named_steps
    
    # Check feature names are stored
    assert hasattr(predictor, 'feature_names')
    assert len(predictor.feature_names) > 0


def test_predict_single(predictor, sample_features):
    """Test single prediction functionality"""
    result = predictor.predict_single(sample_features)
    
    # Check returned structure
    assert 'prediction' in result
    assert 'confidence' in result
    assert 'feature_importance' in result
    
    # Check data types
    assert isinstance(result['prediction'], int)
    assert isinstance(result['confidence'], float)
    assert isinstance(result['feature_importance'], dict)
    
    # Check prediction is 0 or 1
    assert result['prediction'] in [0, 1]
    
    # Check confidence is between 0 and 1
    assert 0 <= result['confidence'] <= 1
    
    # Check feature importance contains encoded feature names
    assert all(col in predictor.feature_names for col in result['feature_importance'].keys())


def test_predict_single_different_genders(predictor):
    """Test predictions with different gender values"""
    # Test with male
    male_features = {
        'age': 30,
        'gender': 'male',
        'income': 6000.0,
        'visit_frequency': 'monthly',
        'satisfaction_level': 'medium'
    }
    
    male_result = predictor.predict_single(male_features)
    assert male_result['prediction'] in [0, 1]
    
    # Test with female
    female_features = {
        'age': 30,
        'gender': 'female',
        'income': 6000.0,
        'visit_frequency': 'monthly',
        'satisfaction_level': 'medium'
    }
    
    female_result = predictor.predict_single(female_features)
    assert female_result['prediction'] in [0, 1]


def test_predict_single_different_visit_frequencies(predictor):
    """Test predictions with different visit frequencies"""
    visit_frequencies = ['rarely', 'monthly', 'weekly', 'daily']
    
    for freq in visit_frequencies:
        features = {
            'age': 25,
            'gender': 'male',
            'income': 5000.0,
            'visit_frequency': freq,
            'satisfaction_level': 'high'
        }
        
        result = predictor.predict_single(features)
        assert result['prediction'] in [0, 1]
        assert 0 <= result['confidence'] <= 1


def test_feature_importance_sum(predictor, sample_features):
    """Test that feature importances sum to approximately 1"""
    result = predictor.predict_single(sample_features)
    importance_sum = sum(result['feature_importance'].values())
    
    # Allow for floating point precision
    assert pytest.approx(importance_sum, abs=1e-6) == 1.0


def test_predict_single_edge_case_age(predictor):
    """Test predictions with minimum valid age"""
    edge_features = {
        'age': 18,
        'gender': 'male',
        'income': 3000.0,
        'visit_frequency': 'rarely',
        'satisfaction_level': 'low'
    }
    
    result = predictor.predict_single(edge_features)
    assert result['prediction'] in [0, 1]
    assert 0 <= result['confidence'] <= 1


def test_predict_single_high_income(predictor):
    """Test predictions with high income"""
    high_income_features = {
        'age': 40,
        'gender': 'female',
        'income': 15000.0,
        'visit_frequency': 'weekly',
        'satisfaction_level': 'high'
    }
    
    result = predictor.predict_single(high_income_features)
    assert result['prediction'] in [0, 1]
    assert 0 <= result['confidence'] <= 1
