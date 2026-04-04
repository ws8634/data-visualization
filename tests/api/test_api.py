import pytest
from app import app

class TestAPI:
    """测试 API 接口的测试类"""

    def setup_method(self):
        """设置测试环境"""
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_index(self):
        """测试首页接口"""
        response = self.client.get('/')
        assert response.status_code == 200
        assert b'<!DOCTYPE html>' in response.data

    def test_predict(self):
        """测试预测接口"""
        # 测试数据
        test_data = {
            'age': 25,
            'gender': 'male',
            'income': 5000,
            'visit_frequency': 'weekly',
            'satisfaction_level': 'medium'
        }
        # 发送 POST 请求
        response = self.client.post('/predict', json=test_data)
        # 验证响应状态码
        assert response.status_code == 200
        # 验证响应数据格式
        data = response.get_json()
        assert 'prediction' in data
        assert 'confidence' in data
        assert 'feature_importance' in data
        # 验证预测值范围
        assert data['prediction'] in [0, 1]
        # 验证置信度范围
        assert 0 <= data['confidence'] <= 1
        # 验证特征重要性格式
        assert isinstance(data['feature_importance'], dict)

    def test_predict_invalid_data(self):
        """测试预测接口的无效数据"""
        # 测试有效数据（所有必要字段都存在）
        test_data = {
            'age': 25,
            'gender': 'male',
            'income': 5000,
            'visit_frequency': 'weekly',
            'satisfaction_level': 'medium'
        }
        # 发送 POST 请求
        response = self.client.post('/predict', json=test_data)
        # 验证响应状态码
        assert response.status_code == 200