import os
import sys
import tempfile
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from app import app
from mcdonald_predictor import McDonaldPredictor


class TestIntegration:
    """测试端到端的集成功能"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_full_prediction_workflow(self, client):
        """测试完整的预测工作流程"""
        predictor = McDonaldPredictor()
        predictor.train_model()
        test_data = {
            'age': 30,
            'gender': 'male',
            'income': 5000,
            'visit_frequency': 'monthly',
            'satisfaction_level': 'low'
        }
        response = client.post('/predict', json=test_data)
        assert response.status_code == 200
        data = response.get_json()
        assert 'prediction' in data
        assert 'confidence' in data
        assert 'feature_importance' in data

    def test_model_training_and_prediction(self):
        """测试模型训练和预测的集成"""
        predictor = McDonaldPredictor()
        predictor.train_model()
        test_cases = [
            {'age': 25, 'gender': 'male', 'income': 6000, 'visit_frequency': 'daily', 'satisfaction_level': 'low'},
            {'age': 40, 'gender': 'female', 'income': 7000, 'visit_frequency': 'weekly', 'satisfaction_level': 'medium'},
            {'age': 55, 'gender': 'male', 'income': 8000, 'visit_frequency': 'rarely', 'satisfaction_level': 'high'},
        ]
        for test_case in test_cases:
            result = predictor.predict_single(test_case)
            assert result['prediction'] in [0, 1]
            assert 0 <= result['confidence'] <= 1

    def test_data_preprocessing_and_model_training(self):
        """测试数据预处理和模型训练的集成"""
        predictor = McDonaldPredictor()
        data = pd.read_csv('mcdonald_data.csv')
        processed_data = predictor._preprocess(data)
        assert len(processed_data) > 0
        assert 'liked_mcdonalds' in processed_data.columns
        predictor.train_model()
        assert predictor.model is not None

    def test_multiple_predictions(self, client):
        """测试多次预测的集成"""
        test_cases = [
            {'age': 20, 'gender': 'female', 'income': 4000, 'visit_frequency': 'monthly', 'satisfaction_level': 'low'},
            {'age': 35, 'gender': 'male', 'income': 5500, 'visit_frequency': 'weekly', 'satisfaction_level': 'medium'},
            {'age': 45, 'gender': 'female', 'income': 6500, 'visit_frequency': 'daily', 'satisfaction_level': 'high'},
            {'age': 50, 'gender': 'male', 'income': 7500, 'visit_frequency': 'rarely', 'satisfaction_level': 'low'},
        ]
        for test_case in test_cases:
            response = client.post('/predict', json=test_case)
            assert response.status_code == 200
            data = response.get_json()
            assert 'prediction' in data
            assert 'confidence' in data

    def test_model_persistence(self):
        """测试模型持久化的集成"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.joblib')
            predictor = McDonaldPredictor()
            predictor.model_path = model_path
            predictor.train_model()
            assert os.path.exists(model_path)
            loaded_predictor = McDonaldPredictor()
            loaded_predictor.model_path = model_path
            loaded_predictor.model = joblib.load(model_path)
            test_data = {
                'age': 30,
                'gender': 'male',
                'income': 5000,
                'visit_frequency': 'monthly',
                'satisfaction_level': 'low'
            }
            result1 = predictor.predict_single(test_data)
            result2 = loaded_predictor.predict_single(test_data)
            assert result1['prediction'] == result2['prediction']

    def test_api_response_consistency(self, client):
        """测试API响应的一致性"""
        test_data = {
            'age': 30,
            'gender': 'male',
            'income': 5000,
            'visit_frequency': 'monthly',
            'satisfaction_level': 'low'
        }
        responses = []
        for _ in range(3):
            response = client.post('/predict', json=test_data)
            assert response.status_code == 200
            responses.append(response.get_json())
        predictions = [r['prediction'] for r in responses]
        assert len(set(predictions)) == 1

    def test_feature_importance_integration(self):
        """测试特征重要性的集成"""
        predictor = McDonaldPredictor()
        predictor.train_model()
        test_data = {
            'age': 30,
            'gender': 'male',
            'income': 5000,
            'visit_frequency': 'monthly',
            'satisfaction_level': 'low'
        }
        result = predictor.predict_single(test_data)
        assert 'feature_importance' in result
        feature_names = ['age', 'gender', 'income', 'visit_frequency', 'satisfaction_level']
        for feature in feature_names:
            assert feature in result['feature_importance']

    def test_end_to_end_prediction(self, client):
        """测试端到端的完整预测流程"""
        response = client.get('/')
        assert response.status_code == 200
        test_data = {
            'age': 28,
            'gender': 'female',
            'income': 6200,
            'visit_frequency': 'weekly',
            'satisfaction_level': 'medium'
        }
        predict_response = client.post('/predict', json=test_data)
        assert predict_response.status_code == 200
        data = predict_response.get_json()
        assert 'prediction' in data
        assert 'confidence' in data
        assert 'feature_importance' in data
        assert data['prediction'] in [0, 1]
        assert 0 <= data['confidence'] <= 1
