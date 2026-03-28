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
        return data[
            (data['age'] >= 18) &
            (data['visit_frequency'] > 0)
            ].dropna()

    def predict_single(self, features):
        df = pd.DataFrame([features])
        proba = self.model.predict_proba(df)[0]
        return {
            'prediction': int(self.model.predict(df)[0]),
            'confidence': float(max(proba)),
            'feature_importance': dict(zip(
                df.columns,
                self.model.named_steps['classifier'].feature_importances_
            ))
        }
