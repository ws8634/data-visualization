"""模型评估模块

本模块负责加载训练好的模型，对测试数据进行预测，并输出准确率、F1-score等评估指标。
"""

import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from train_model import load_data


def load_model(filename='model.pkl'):
    """加载模型
    
    Args:
        filename: 模型文件名
    
    Returns:
        model: 加载的模型
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def evaluate_model(model, X_test, y_test):
    """评估模型性能
    
    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
    
    Returns:
        metrics: 评估指标字典
    """
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # 生成分类报告
    report = classification_report(y_test, y_pred)
    
    # 构建指标字典
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report
    }
    
    return metrics


def print_metrics(metrics):
    """打印评估指标
    
    Args:
        metrics: 评估指标字典
    """
    print("===== 模型评估结果 =====")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print("\n分类报告:")
    print(metrics['classification_report'])


if __name__ == "__main__":
    # 加载数据
    X_train, X_test, y_train, y_test = load_data()
    
    # 加载模型
    model = load_model()
    
    # 评估模型
    metrics = evaluate_model(model, X_test, y_test)
    
    # 打印评估结果
    print_metrics(metrics)