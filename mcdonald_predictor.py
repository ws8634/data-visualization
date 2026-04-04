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
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.train_model()

    def train_model(self):
        data = pd.read_csv('mcdonald_data.csv')
        processed_data = self._preprocess(data)

        X = processed_data.drop('liked_mcdonalds', axis=1)
        y = processed_data['liked_mcdonalds']

        # 定义分类特征和数值特征
        categorical_features = ['gender', 'visit_frequency', 'satisfaction_level']
        numerical_features = ['age', 'income']

        # 创建预处理器
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(), categorical_features)
            ]
        )

        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(max_depth=5))
        ])

        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)

    def _preprocess(self, data):
        # 过滤年龄大于等于18的记录
        return data[data['age'] >= 18].dropna()

    def predict_single(self, features):
        df = pd.DataFrame([features])
        proba = self.model.predict_proba(df)[0]
        return {
            'prediction': int(self.model.predict(df)[0]),
            'confidence': float(max(proba)),
            'feature_importance': dict(zip(
                self.model.named_steps['preprocessor'].get_feature_names_out(),
                self.model.named_steps['classifier'].feature_importances_
            ))
        }
