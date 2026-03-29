import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    FLASK_APP = os.getenv('FLASK_APP', 'app.py')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = bool(int(os.getenv('FLASK_DEBUG', '0')))
    PORT = int(os.getenv('PORT', '5000'))
    MODEL_PATH = os.getenv('MODEL_PATH', 'model.joblib')
    DATA_PATH = os.getenv('DATA_PATH', 'mcdonald_data.csv')
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
