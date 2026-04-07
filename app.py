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
    data = request.json
    result = predictor.predict_single(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
