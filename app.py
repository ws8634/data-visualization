from flask import Flask, render_template, request, jsonify
from mcdonald_predictor import McDonaldPredictor
import os

app = Flask(__name__)
predictor = McDonaldPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    处理预测请求
    """
    try:
        data = request.json
        result = predictor.predict_single(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/history', methods=['GET'])
def get_history():
    """
    获取预测历史记录
    """
    try:
        history = predictor.get_prediction_history()
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """
    获取数据集统计信息
    """
    try:
        stats = predictor.get_dataset_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
