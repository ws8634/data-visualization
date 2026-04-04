import pytest
import os
import pandas as pd
from mcdonald_predictor import McDonaldPredictor
from app import app

class TestIntegration:
    """集成测试类"""

    def setup_method(self):
        """设置测试环境"""
        # 保存原始模型文件路径
        self.model_path = 'model.joblib'
        self.original_model_exists = os.path.exists(self.model_path)
        # 如果模型存在，先备份
        if self.original_model_exists:
            os.rename(self.model_path, f'{self.model_path}.bak')
        # 创建预测器实例
        self.predictor = McDonaldPredictor()
        # 创建测试客户端
        app.config['TESTING'] = True
        self.client = app.test_client()

    def teardown_method(self):
        """清理测试环境"""
        # 恢复原始模型文件
        if self.original_model_exists:
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            os.rename(f'{self.model_path}.bak', self.model_path)
        else:
            # 如果原始模型不存在，删除测试生成的模型
            if os.path.exists(self.model_path):
                os.remove(self.model_path)

    def test_end_to_end_prediction(self):
        """测试端到端预测流程"""
        # 1. 测试数据加载和模型训练
        assert os.path.exists(self.model_path)
        assert hasattr(self.predictor, 'model')
        
        # 2. 测试模型预测
        test_features = {
            'age': 25,
            'gender': 'male',
            'income': 5000,
            'visit_frequency': 'weekly',
            'satisfaction_level': 'medium'
        }
        result = self.predictor.predict_single(test_features)
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'feature_importance' in result
        
        # 3. 测试 API 调用
        response = self.client.post('/predict', json=test_features)
        assert response.status_code == 200
        api_result = response.get_json()
        assert 'prediction' in api_result
        assert 'confidence' in api_result
        assert 'feature_importance' in api_result

    def test_model_persistence(self):
        """测试模型持久化功能"""
        # 1. 测试模型保存
        assert os.path.exists(self.model_path)
        
        # 2. 测试模型加载
        new_predictor = McDonaldPredictor()
        assert hasattr(new_predictor, 'model')
        
        # 3. 测试预测结果一致性
        test_features = {
            'age': 25,
            'gender': 'male',
            'income': 5000,
            'visit_frequency': 'weekly',
            'satisfaction_level': 'medium'
        }
        result1 = self.predictor.predict_single(test_features)
        result2 = new_predictor.predict_single(test_features)
        assert result1['prediction'] == result2['prediction']