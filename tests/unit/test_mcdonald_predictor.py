import pytest
import pandas as pd
import os
from mcdonald_predictor import McDonaldPredictor

class TestMcDonaldPredictor:
    """测试 McDonaldPredictor 类的单元测试"""

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

    def test_init(self):
        """测试初始化方法"""
        assert self.predictor is not None
        assert hasattr(self.predictor, 'model')
        assert os.path.exists(self.model_path)

    def test_preprocess(self):
        """测试数据预处理方法"""
        # 创建测试数据
        test_data = pd.DataFrame({
            'age': [17, 18, 25, 30],
            'gender': ['male', 'female', 'male', 'female'],
            'income': [3000, 4000, 5000, 6000],
            'visit_frequency': ['rarely', 'weekly', 'monthly', 'daily'],
            'satisfaction_level': ['low', 'medium', 'high', 'medium'],
            'liked_mcdonalds': [0, 1, 1, 0]
        })
        # 测试预处理
        processed_data = self.predictor._preprocess(test_data)
        # 验证结果
        assert len(processed_data) == 3  # 排除年龄小于18的记录
        assert all(processed_data['age'] >= 18)

    def test_predict_single(self):
        """测试单条数据预测方法"""
        # 测试数据
        test_features = {
            'age': 25,
            'gender': 'male',
            'income': 5000,
            'visit_frequency': 'weekly',
            'satisfaction_level': 'medium'
        }
        # 测试预测
        result = self.predictor.predict_single(test_features)
        # 验证结果格式
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'feature_importance' in result
        # 验证预测值范围
        assert result['prediction'] in [0, 1]
        # 验证置信度范围
        assert 0 <= result['confidence'] <= 1
        # 验证特征重要性格式
        assert isinstance(result['feature_importance'], dict)