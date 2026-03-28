import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os


class McDonaldPredictor:
    def __init__(self):
        self.model_path = 'model.joblib'
        self.feature_names_path = 'feature_names.joblib'
        if os.path.exists(self.model_path) and os.path.exists(self.feature_names_path):
            self.model = joblib.load(self.model_path)
            self.feature_names = joblib.load(self.feature_names_path)
        else:
            self.train_model()

    def train_model(self):
        data = pd.read_csv('mcdonald_data.csv')
        processed_data = self._preprocess(data)

        # Encode categorical variables
        data_encoded = pd.get_dummies(processed_data, drop_first=True)
        
        X = data_encoded.drop('liked_mcdonalds', axis=1)
        y = data_encoded['liked_mcdonalds']

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', DecisionTreeClassifier(max_depth=5))
        ])

        self.model.fit(X, y)
        self.feature_names = X.columns.tolist()
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.feature_names, self.feature_names_path)

    def _preprocess(self, data):
        # Filter valid age and remove NA values
        return data[
            (data['age'] >= 18)
            ].dropna()

    def predict_single(self, features):
        # Validate input features
        required_fields = ['age', 'gender', 'income', 'visit_frequency', 'satisfaction_level']
        missing_fields = [field for field in required_fields if field not in features]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Convert input features to DataFrame
        df = pd.DataFrame([features])
        
        # Apply same preprocessing as training
        df = self._preprocess(df)
        
        # Encode categorical variables
        df_encoded = pd.get_dummies(df, drop_first=True)
        
        # Ensure all expected features are present
        for col in self.feature_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Reorder columns to match training data
        df_encoded = df_encoded[self.feature_names]
        
        proba = self.model.predict_proba(df_encoded)[0]
        
        return {
            'prediction': int(self.model.predict(df_encoded)[0]),
            'confidence': float(max(proba)),
            'feature_importance': dict(zip(
                self.feature_names,
                self.model.named_steps['classifier'].feature_importances_
            ))
        }
