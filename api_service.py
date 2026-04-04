"""REST API服务模块

本模块使用Flask框架提供REST API接口，用于外部调用预测服务。
包含一个预测端点，接收POST请求并返回预测结果，同时提供静态文件服务。
"""

import pickle
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

# 加载模型
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# 类别映射
class_names = ['setosa', 'versicolor', 'virginica']


@app.route('/')
def index():
    """根路径，返回前端页面"""
    return send_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """预测接口
    
    请求参数：
        - sepal_length: 花萼长度
        - sepal_width: 花萼宽度
        - petal_length: 花瓣长度
        - petal_width: 花瓣宽度
    
    响应格式：
        {
            "prediction": "预测的类别名称",
            "prediction_id": 预测的类别ID
        }
    """
    try:
        # 获取请求数据
        data = request.json
        
        # 提取特征
        features = [
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]
        
        # 预测
        prediction = model.predict([features])[0]
        
        # 构建响应
        response = {
            "prediction": class_names[prediction],
            "prediction_id": int(prediction)
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)