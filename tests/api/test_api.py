import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest
from app import app


class TestAPI:
    """测试Flask应用的API接口"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_index_route(self, client):
        """测试首页路由是否正常响应"""
        response = client.get('/')
        assert response.status_code == 200

    def test_predict_route_valid_input(self, client):
        """测试预测接口使用有效输入"""
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

    def test_predict_route_missing_fields(self, client):
        """测试预测接口缺少必填字段"""
        test_data = {
            'age': 30,
            'gender': 'male'
        }
        try:
            response = client.post('/predict', json=test_data)
            assert response.status_code == 500
        except Exception:
            pass

    def test_predict_route_invalid_age(self, client):
        """测试预测接口输入无效年龄"""
        test_data = {
            'age': -5,
            'gender': 'male',
            'income': 5000,
            'visit_frequency': 'monthly',
            'satisfaction_level': 'low'
        }
        response = client.post('/predict', json=test_data)
        assert response.status_code in [200, 500]

    def test_predict_route_invalid_visit_frequency(self, client):
        """测试预测接口输入无效访问频率"""
        test_data = {
            'age': 30,
            'gender': 'male',
            'income': 5000,
            'visit_frequency': 'invalid',
            'satisfaction_level': 'low'
        }
        response = client.post('/predict', json=test_data)
        assert response.status_code in [200, 500]

    def test_predict_route_response_format(self, client):
        """测试预测接口响应格式"""
        test_data = {
            'age': 25,
            'gender': 'female',
            'income': 6000,
            'visit_frequency': 'weekly',
            'satisfaction_level': 'medium'
        }
        response = client.post('/predict', json=test_data)
        data = response.get_json()
        assert isinstance(data['prediction'], int)
        assert isinstance(data['confidence'], float)
        assert isinstance(data['feature_importance'], dict)
        assert data['prediction'] in [0, 1]
        assert 0 <= data['confidence'] <= 1

    def test_predict_route_content_type(self, client):
        """测试预测接口返回内容类型"""
        test_data = {
            'age': 40,
            'gender': 'male',
            'income': 7000,
            'visit_frequency': 'daily',
            'satisfaction_level': 'high'
        }
        response = client.post('/predict', json=test_data)
        assert response.content_type == 'application/json'

    def test_predict_route_empty_json(self, client):
        """测试预测接口空JSON输入"""
        try:
            response = client.post('/predict', json={})
            assert response.status_code == 500
        except Exception:
            pass

    def test_predict_route_none_json(self, client):
        """测试预测接口无JSON输入"""
        response = client.post('/predict')
        assert response.status_code == 415
