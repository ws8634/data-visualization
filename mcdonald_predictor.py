import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os
from datetime import datetime


class McDonaldPredictor:
    def __init__(self):
        self.model_path = 'model.joblib'
        self.history_path = 'prediction_history.csv'
        self.history_columns = [
            'timestamp', 'age', 'gender', 'income', 'visit_frequency',
            'satisfaction_level', 'prediction', 'confidence'
        ]
        self._init_history()
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.train_model()

    def _init_history(self):
        """初始化预测历史记录文件"""
        if not os.path.exists(self.history_path):
            pd.DataFrame(columns=self.history_columns).to_csv(self.history_path, index=False)

    def train_model(self):
        """训练预测模型"""
        data = self._load_data()
        processed_data = self._preprocess(data)

        X = processed_data.drop('liked_mcdonalds', axis=1)
        y = processed_data['liked_mcdonalds']

        numeric_features = ['age', 'income']
        categorical_features = ['gender', 'visit_frequency', 'satisfaction_level']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first'), categorical_features)
            ])

        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(max_depth=5))
        ])

        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)

    def _load_data(self):
        """加载原始数据"""
        return pd.read_csv('mcdonald_data.csv')

    def _preprocess(self, data):
        """数据预处理"""
        return data[
            (data['age'] >= 18) &
            (data['visit_frequency'].notna())
        ].dropna()

    def predict_single(self, features):
        """单个样本预测"""
        df = pd.DataFrame([features])
        proba = self.model.predict_proba(df)[0]
        prediction = int(self.model.predict(df)[0])
        confidence = float(max(proba))

        self._save_history(features, prediction, confidence)

        return {
            'prediction': prediction,
            'confidence': confidence,
            'feature_importance': self._get_feature_importance()
        }

    def _save_history(self, features, prediction, confidence):
        """保存预测历史"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'age': features['age'],
            'gender': features['gender'],
            'income': features['income'],
            'visit_frequency': features['visit_frequency'],
            'satisfaction_level': features['satisfaction_level'],
            'prediction': prediction,
            'confidence': confidence
        }
        df = pd.DataFrame([history_entry])
        df.to_csv(self.history_path, mode='a', header=False, index=False)

    def get_prediction_history(self, limit=50):
        """获取预测历史记录"""
        if os.path.exists(self.history_path):
            history = pd.read_csv(self.history_path)
            return history.tail(limit).to_dict('records')
        return []

    def get_statistics(self):
        """获取统计数据"""
        data = self._load_data()
        history = self.get_prediction_history()

        stats = {
            'total_samples': len(data),
            'positive_ratio': float(data['liked_mcdonalds'].mean()),
            'avg_age': float(data['age'].mean()),
            'avg_income': float(data['income'].mean()),
            'gender_distribution': data['gender'].value_counts().to_dict(),
            'visit_distribution': data['visit_frequency'].value_counts().to_dict(),
            'satisfaction_distribution': data['satisfaction_level'].value_counts().to_dict(),
            'history_count': len(history)
        }
        return stats

    def _get_feature_importance(self):
        """获取特征重要性"""
        classifier = self.model.named_steps['classifier']
        preprocessor = self.model.named_steps['preprocessor']
        
        numeric_features = ['age', 'income']
        categorical_features = ['gender', 'visit_frequency', 'satisfaction_level']
        
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = numeric_features + list(cat_feature_names)
        
        importance = dict(zip(all_feature_names, classifier.feature_importances_))
        return {k: float(v) for k, v in importance.items()}
