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

        X = processed_data.drop('liked_mcdonalds', axis=1)
        y = processed_data['liked_mcdonalds']

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', DecisionTreeClassifier(max_depth=5))
        ])

        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)

    def _preprocess(self, data):
        # 处理分类特征
        data = data.copy()
        data['gender'] = data['gender'].map({'male': 0, 'female': 1})
        data['visit_frequency'] = data['visit_frequency'].map({'daily': 0, 'weekly': 1, 'monthly': 2, 'rarely': 3})
        data['satisfaction_level'] = data['satisfaction_level'].map({'low': 0, 'medium': 1, 'high': 2})
        
        return data[
            (data['age'] >= 18)
            ].dropna()

    def predict_single(self, features):
        # 处理分类特征
        processed_features = features.copy()
        processed_features['gender'] = 0 if processed_features['gender'] == 'male' else 1
        processed_features['visit_frequency'] = {'daily': 0, 'weekly': 1, 'monthly': 2, 'rarely': 3}[processed_features['visit_frequency']]
        processed_features['satisfaction_level'] = {'low': 0, 'medium': 1, 'high': 2}[processed_features['satisfaction_level']]
        
        df = pd.DataFrame([processed_features])
        proba = self.model.predict_proba(df)[0]
        return {
            'prediction': int(self.model.predict(df)[0]),
            'confidence': float(max(proba)),
            'feature_importance': dict(zip(
                df.columns,
                self.model.named_steps['classifier'].feature_importances_
            ))
        }
