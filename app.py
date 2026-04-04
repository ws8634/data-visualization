from flask import Flask, render_template, request, jsonify
from mcdonald_predictor import McDonaldPredictor
import os

app = Flask(__name__)
predictor = McDonaldPredictor()


@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """处理预测请求"""
    try:
        data = request.json
        result = predictor.predict_single(data)
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/history', methods=['GET'])
def get_history():
    """获取预测历史记录"""
    try:
        history = predictor.get_prediction_history()
        return jsonify({
            'success': True,
            'data': history
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/statistics', methods=['GET'])
def get_statistics():
    """获取统计数据"""
    try:
        stats = predictor.get_statistics()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
