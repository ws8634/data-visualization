from flask import Flask, request, jsonify
from flask_cors import CORS
from config import Config
from model_service import ModelService
from schemas import PredictionRequest, PredictionResponse, ErrorResponse
from pydantic import ValidationError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(config: Config) -> Flask:
    app = Flask(__name__)
    
    CORS(app, origins=config.CORS_ORIGINS)
    
    model_service = ModelService(config.MODEL_PATH, config.DATA_PATH)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'model_loaded': model_service.model is not None
        })
    
    @app.route('/api/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            validated_data = PredictionRequest(**data)
            
            prediction, confidence, feature_importance = model_service.predict(
                validated_data.model_dump()
            )
            
            response = PredictionResponse(
                prediction=prediction,
                confidence=confidence,
                feature_importance=feature_importance
            )
            
            return jsonify(response.model_dump()), 200
            
        except ValidationError as e:
            error_response = ErrorResponse(
                error='Invalid input data',
                details=e.errors()
            )
            return jsonify(error_response.model_dump()), 400
            
        except Exception as e:
            logger.exception(f"Error during prediction: {str(e)}")
            error_response = ErrorResponse(
                error='Internal server error',
                details={'message': str(e)}
            )
            return jsonify(error_response.model_dump()), 500
    
    @app.route('/api/model/reload', methods=['POST'])
    def reload_model():
        try:
            model_service.reload_model()
            return jsonify({'message': 'Model reloaded successfully'}), 200
        except Exception as e:
            logger.exception(f"Error reloading model: {str(e)}")
            error_response = ErrorResponse(
                error='Failed to reload model',
                details={'message': str(e)}
            )
            return jsonify(error_response.model_dump()), 500
    
    return app

if __name__ == '__main__':
    config = Config()
    app = create_app(config)
    app.run(host='0.0.0.0', port=config.PORT, debug=config.FLASK_DEBUG)
