import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os


class McDonaldPredictor:
    def __init__(self):
        self.model_path = 'model.joblib'
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.train_model()

    def train_model(self):
        data = pd.read_csv('mcdonald_data.csv')
        processed_data = self._preprocess(data)

        X = processed_data.drop('likes_mcdonald', axis=1)
        y = processed_data['likes_mcdonald']

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', DecisionTreeClassifier(max_depth=5))
        ])

        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)

    def _preprocess(self, data):
        visit_mapping = {
            'rarely': 1,
            'monthly': 2,
            'weekly': 3,
            'daily': 4
        }
        gender_mapping = {
            'male': 0,
            'female': 1
        }
        satisfaction_mapping = {
            'low': 1,
            'medium': 2,
            'high': 3
        }
        data = data.copy()
        data['visit_frequency'] = data['visit_frequency'].map(visit_mapping)
        data['gender'] = data['gender'].map(gender_mapping)
        data['satisfaction_level'] = data['satisfaction_level'].map(satisfaction_mapping)
        return data[
            (data['age'] >= 18) &
            (data['visit_frequency'] > 0)
            ].dropna().rename(columns={'liked_mcdonalds': 'likes_mcdonald'})

    def predict_single(self, features):
        df = pd.DataFrame([features])
        visit_mapping = {
            'rarely': 1,
            'monthly': 2,
            'weekly': 3,
            'daily': 4
        }
        gender_mapping = {
            'male': 0,
            'female': 1
        }
        satisfaction_mapping = {
            'low': 1,
            'medium': 2,
            'high': 3
        }
        df = df.copy()
        
        required_columns = ['age', 'gender', 'income', 'visit_frequency', 'satisfaction_level']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"缺少必要字段: {col}")
        
        try:
            df['visit_frequency'] = df['visit_frequency'].map(visit_mapping)
            df['gender'] = df['gender'].map(gender_mapping)
            df['satisfaction_level'] = df['satisfaction_level'].map(satisfaction_mapping)
        except Exception as e:
            raise ValueError("字段映射失败，请检查输入数据")
        
        if df.isnull().any().any():
            raise ValueError("输入数据包含无效值")
        
        proba = self.model.predict_proba(df)[0]
        return {
            'prediction': int(self.model.predict(df)[0]),
            'confidence': float(max(proba)),
            'feature_importance': dict(zip(
                df.columns,
                self.model.named_steps['classifier'].feature_importances_
            ))
        }
