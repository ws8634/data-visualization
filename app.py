"""
Flask API服务模块
该模块提供REST API接口，用于模型预测
支持单样本预测和批量预测
"""

from flask import Flask, request, jsonify, send_from_directory
import joblib
import os
import numpy as np


app = Flask(__name__)

model = None
scaler = None
encoder = None


def load_model():
    """
    加载模型和预处理器
    """
    global model, scaler, encoder
    model_dir = 'models'
    model = joblib.load(os.path.join(model_dir, 'ensemble_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    encoder = joblib.load(os.path.join(model_dir, 'encoder.pkl'))


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    单样本预测接口
    请求方式: POST
    请求参数:
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    响应格式:
        {
            "success": true,
            "prediction": "setosa",
            "confidence": 0.95,
            "probabilities": [0.95, 0.03, 0.02]
        }
    """
    try:
        data = request.get_json(force=True)
        
        required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for field in required_fields:
            if field not in data:
                return jsonify({
            "success": False,
            "error": f"缺少必要参数: {field}"
        }), 400
        
        features = [
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]
        
        features_scaled = scaler.transform([features])
        
        prediction_encoded = model.predict(features_scaled)[0]
        prediction = encoder.inverse_transform([prediction_encoded])[0]
        
        probabilities = model.predict_proba(features_scaled)[0].tolist()
        confidence = max(probabilities)
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    """
    批量预测接口
    请求方式: POST
    请求参数:
        {
            "samples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                },
                {
                    "sepal_length": 7.0,
                    "sepal_width": 3.2,
                    "petal_length": 4.7,
                    "petal_width": 1.4
                }
            ]
        }
    响应格式:
        {
            "success": true,
            "predictions": [
                {
                    "prediction": "setosa",
                    "confidence": 0.95
                },
                {
                    "prediction": "versicolor",
                    "confidence": 0.88
                }
            ]
        }
    """
    try:
        data = request.get_json(force=True)
        
        if 'samples' not in data:
            return jsonify({
                "success": False,
                "error": "缺少必要参数: samples"
            }), 400
        
        samples = data['samples']
        if not isinstance(samples, list):
            return jsonify({
                "success": False,
                "error": "samples必须是列表"
            }), 400
        
        required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        features_list = []
        
        for sample in samples:
            for field in required_fields:
                if field not in sample:
                    return jsonify({
                        "success": False,
                        "error": f"样本中缺少必要参数: {field}"
                    }), 400
            
            features = [
                sample['sepal_length'],
                sample['sepal_width'],
                sample['petal_length'],
                sample['petal_width']
            ]
            features_list.append(features)
        
        features_scaled = scaler.transform(features_list)
        
        predictions_encoded = model.predict(features_scaled)
        predictions = encoder.inverse_transform(predictions_encoded)
        
        probabilities_list = model.predict_proba(features_scaled)
        
        results = []
        for pred, probs in zip(predictions, probabilities_list):
            results.append({
                "prediction": pred,
                "confidence": float(max(probs))
            })
        
        return jsonify({
            "success": True,
            "predictions": results
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    请求方式: GET
    响应格式:
        {
            "status": "healthy",
            "model_loaded": true
        }
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


@app.route('/')
def index():
    """
    提供前端页面
    """
    return send_from_directory('static', 'index.html')


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=8001, debug=True)
