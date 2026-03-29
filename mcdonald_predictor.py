import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os


class McDonaldPredictor:
    def __init__(self):
        self.model_path = 'model.joblib'
        self.prediction_history = []  # 存储预测历史
        
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.train_model()

    def train_model(self):
        """
        使用完整特征训练模型
        包含：年龄、性别、收入、访问频率、满意度
        """
        data = pd.read_csv('mcdonald_data.csv')
        processed_data = self._preprocess(data)

        # 分离特征和目标变量
        X = processed_data.drop('liked_mcdonalds', axis=1)
        y = processed_data['liked_mcdonalds']

        # 定义分类和数值特征
        categorical_features = ['gender', 'visit_frequency', 'satisfaction_level']
        numeric_features = ['age', 'income']

        # 创建预处理管道
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])

        # 创建完整模型管道
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(max_depth=5))
        ])

        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)

    def _preprocess(self, data):
        """
        数据预处理
        """
        # 确保所有列名一致
        if 'likes_mcdonald' in data.columns:
            data = data.rename(columns={'likes_mcdonald': 'liked_mcdonalds'})
        
        # 基本过滤
        data = data[
            (data['age'] >= 18) &
            (data['visit_frequency'].notna())
        ].dropna()
        
        return data

    def predict_single(self, features):
        """
        预测单个样本并记录历史
        
        @param features: 特征字典，包含age, gender, income, visit_frequency, satisfaction_level
        @return: 预测结果字典
        """
        df = pd.DataFrame([features])
        
        # 获取预测概率
        proba = self.model.predict_proba(df)[0]
        prediction = int(self.model.predict(df)[0])
        
        # 记录到历史
        history_entry = features.copy()
        history_entry.update({
            'prediction': prediction,
            'confidence': float(max(proba)),
            'timestamp': pd.Timestamp.now().isoformat()
        })
        self.prediction_history.append(history_entry)
        
        # 获取特征重要性
        feature_importance = self._get_feature_importance()
        
        return {
            'prediction': prediction,
            'confidence': float(max(proba)),
            'feature_importance': feature_importance
        }
    
    def get_prediction_history(self):
        """
        获取预测历史记录
        
        @return: 预测历史列表
        """
        return self.prediction_history
    
    def get_dataset_statistics(self):
        """
        获取数据集统计信息
        
        @return: 统计信息字典
        """
        data = pd.read_csv('mcdonald_data.csv')
        return {
            'total_records': len(data),
            'liked_count': int(data['liked_mcdonalds'].sum()),
            'age_stats': {
                'mean': float(data['age'].mean()),
                'min': int(data['age'].min()),
                'max': int(data['age'].max())
            },
            'income_stats': {
                'mean': float(data['income'].mean()),
                'min': float(data['income'].min()),
                'max': float(data['income'].max())
            }
        }
    
    def _get_feature_importance(self):
        """
        获取特征重要性
        
        @return: 特征重要性字典
        """
        # 获取分类特征的编码后名称
        cat_features = self.model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(['gender', 'visit_frequency', 'satisfaction_level'])
        all_features = ['age', 'income'] + list(cat_features)
        
        # 获取特征重要性
        importances = self.model.named_steps['classifier'].feature_importances_
        
        # 合并同一类特征的重要性
        feature_importance = {}
        for feature in ['age', 'income', 'gender', 'visit_frequency', 'satisfaction_level']:
            indices = [i for i, name in enumerate(all_features) if name.startswith(feature)]
            feature_importance[feature] = float(sum(importances[indices]))
        
        return feature_importance
