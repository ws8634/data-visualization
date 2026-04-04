import os
import sys
import tempfile
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mcdonald_predictor import McDonaldPredictor


class TestMcDonaldPredictor:
    """测试McDonaldPredictor类的核心功能"""

    def test_init_with_existing_model(self):
        """测试当模型文件存在时的初始化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.joblib')
            predictor = McDonaldPredictor()
            predictor.model_path = model_path
            predictor.train_model()
            assert os.path.exists(model_path)

    def test_init_without_existing_model(self):
        """测试当模型文件不存在时的初始化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.joblib')
            if os.path.exists(model_path):
                os.remove(model_path)
            predictor = McDonaldPredictor()
            predictor.model_path = model_path
            predictor.train_model()
            assert os.path.exists(model_path)

    def test_train_model(self):
        """测试模型训练功能"""
        predictor = McDonaldPredictor()
        predictor.train_model()
        assert predictor.model is not None
        assert hasattr(predictor.model, 'predict')
        assert hasattr(predictor.model, 'predict_proba')

    def test_preprocess(self):
        """测试数据预处理功能"""
        predictor = McDonaldPredictor()
        test_data = pd.DataFrame({
            'age': [25, 17, 30, 45],
            'gender': ['male', 'female', 'male', 'female'],
            'income': [5000, 3000, 6000, 7000],
            'visit_frequency': ['monthly', 'weekly', 'rarely', 'daily'],
            'satisfaction_level': ['low', 'medium', 'high', 'low'],
            'liked_mcdonalds': [1, 0, 1, 0]
        })
        processed = predictor._preprocess(test_data)
        assert len(processed) == 3
        assert all(processed['age'] >= 18)

    def test_predict_single(self):
        """测试单条数据预测功能"""
        predictor = McDonaldPredictor()
        predictor.train_model()
        test_features = {
            'age': 30,
            'gender': 'male',
            'income': 5000,
            'visit_frequency': 'monthly',
            'satisfaction_level': 'low'
        }
        result = predictor.predict_single(test_features)
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'feature_importance' in result
        assert result['prediction'] in [0, 1]
        assert 0 <= result['confidence'] <= 1

    def test_predict_single_confidence_range(self):
        """测试预测置信度在合理范围内"""
        predictor = McDonaldPredictor()
        predictor.train_model()
        test_features = {
            'age': 25,
            'gender': 'female',
            'income': 6000,
            'visit_frequency': 'weekly',
            'satisfaction_level': 'medium'
        }
        result = predictor.predict_single(test_features)
        assert isinstance(result['confidence'], float)
        assert 0.0 <= result['confidence'] <= 1.0

    def test_predict_single_feature_importance(self):
        """测试特征重要性返回正确"""
        predictor = McDonaldPredictor()
        predictor.train_model()
        test_features = {
            'age': 40,
            'gender': 'male',
            'income': 7000,
            'visit_frequency': 'daily',
            'satisfaction_level': 'high'
        }
        result = predictor.predict_single(test_features)
        assert isinstance(result['feature_importance'], dict)
        assert len(result['feature_importance']) > 0

    def test_model_saving_and_loading(self):
        """测试模型保存和加载功能"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.joblib')
            predictor = McDonaldPredictor()
            predictor.model_path = model_path
            predictor.train_model()
            assert os.path.exists(model_path)
            loaded_model = joblib.load(model_path)
            assert loaded_model is not None
