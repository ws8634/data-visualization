import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, model_path: str, data_path: str):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self._load_or_train_model()

    def _load_or_train_model(self) -> None:
        if os.path.exists(self.model_path):
            logger.info(f"Loading existing model from {self.model_path}")
            self.model = joblib.load(self.model_path)
        else:
            logger.info("No existing model found, training new model")
            self._train_model()

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[
            (data['age'] >= 18) &
            (data['visit_frequency'] > 0)
        ].dropna()

    def _train_model(self) -> None:
        data = pd.read_csv(self.data_path)
        processed_data = self._preprocess_data(data)

        X = processed_data.drop('likes_mcdonald', axis=1)
        y = processed_data['likes_mcdonald']

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', DecisionTreeClassifier(max_depth=5))
        ])

        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model trained and saved to {self.model_path}")

    def predict(self, features: Dict[str, Any]) -> Tuple[int, float, Dict[str, float]]:
        df = pd.DataFrame([features])
        proba = self.model.predict_proba(df)[0]
        prediction = int(self.model.predict(df)[0])
        confidence = float(max(proba))
        
        feature_importance = dict(zip(
            df.columns,
            self.model.named_steps['classifier'].feature_importances_
        ))
        
        return prediction, confidence, feature_importance

    def reload_model(self) -> None:
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info("Model reloaded successfully")
        else:
            logger.warning("Model file not found, cannot reload")
