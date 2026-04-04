"""
模型评估模块
该模块负责加载训练好的模型，在测试集上进行评估
输出准确率、精确率、召回率、F1-score等指标
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
import os
from train import load_data, preprocess_data


def load_model(model_dir='models'):
    """
    加载训练好的模型和预处理器
    参数:
        model_dir: 模型保存目录
    返回:
        model: 训练后的模型
        scaler: 特征标准化器
        encoder: 标签编码器
    """
    model = joblib.load(os.path.join(model_dir, 'ensemble_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    encoder = joblib.load(os.path.join(model_dir, 'encoder.pkl'))
    return model, scaler, encoder


def evaluate_model(model, X_test, y_test, encoder):
    """
    评估模型性能
    参数:
        model: 训练后的模型
        X_test: 测试特征集
        y_test: 测试标签集
        encoder: 标签编码器
    返回:
        metrics: 评估指标字典
        y_pred: 预测结果
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics, y_pred


def print_evaluation_report(metrics, y_test, y_pred, encoder):
    """
    打印评估报告
    参数:
        metrics: 评估指标字典
        y_test: 真实标签
        y_pred: 预测标签
        encoder: 标签编码器
    """
    print("\n" + "="*50)
    print("模型评估报告")
    print("="*50)
    
    print(f"\n准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall): {metrics['recall']:.4f}")
    print(f"F1分数 (F1-score): {metrics['f1_score']:.4f}")
    
    print("\n" + "-"*50)
    print("分类报告:")
    print("-"*50)
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    
    print("\n" + "-"*50)
    print("混淆矩阵:")
    print("-"*50)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\n类别对应关系:")
    for i, class_name in enumerate(encoder.classes_):
        print(f"  {i}: {class_name}")
    print("="*50 + "\n")


def main():
    """
    主函数：执行完整的评估流程
    """
    data_path = os.path.join('data', 'iris.csv')
    model_dir = 'models'
    
    print("加载数据...")
    X, y = load_data(data_path)
    
    print("加载模型...")
    model, scaler, encoder = load_model(model_dir)
    
    print("预处理数据...")
    _, X_test, _, y_test, _, _ = preprocess_data(X, y)
    
    print("评估模型...")
    metrics, y_pred = evaluate_model(model, X_test, y_test, encoder)
    
    print_evaluation_report(metrics, y_test, y_pred, encoder)
    
    return metrics


if __name__ == '__main__':
    main()
