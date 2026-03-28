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
    try:
        data = request.json
        if not data:
            return jsonify({'error': '请提供JSON数据'}), 400
        result = predictor.predict_single(data)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': '服务器内部错误'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
