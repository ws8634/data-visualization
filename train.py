"""
模型训练模块
该模块负责数据加载、模型训练和模型保存
包含决策树和K近邻两种算法的集成方案
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import joblib
import os


def load_data(data_path):
    """
    加载数据集
    参数:
        data_path: 数据集文件路径
    返回:
        X: 特征矩阵
        y: 标签向量
    """
    df = pd.read_csv(data_path)
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']
    return X, y


def preprocess_data(X, y):
    """
    数据预处理
    参数:
        X: 特征矩阵
        y: 标签向量
    返回:
        X_train, X_test, y_train, y_test: 划分后的训练集和测试集
        scaler: 特征标准化器
        encoder: 标签编码器
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoder


def build_ensemble_model():
    """
    构建集成模型
    返回:
        ensemble: 集成分类器
    """
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    
    ensemble = VotingClassifier(
        estimators=[('dt', dt), ('knn', knn)],
        voting='soft'
    )
    return ensemble


def train_model(model, X_train, y_train):
    """
    训练模型
    参数:
        model: 待训练的模型
        X_train: 训练特征集
        y_train: 训练标签集
    返回:
        model: 训练后的模型
    """
    model.fit(X_train, y_train)
    return model


def save_model(model, scaler, encoder, model_dir='models'):
    """
    保存模型和预处理器
    参数:
        model: 训练后的模型
        scaler: 特征标准化器
        encoder: 标签编码器
        model_dir: 模型保存目录
    """
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'ensemble_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(encoder, os.path.join(model_dir, 'encoder.pkl'))
    print(f"模型已保存到 {model_dir} 目录")


def main():
    """
    主函数：执行完整的训练流程
    """
    data_path = os.path.join('data', 'iris.csv')
    
    print("加载数据...")
    X, y = load_data(data_path)
    
    print("预处理数据...")
    X_train, X_test, y_train, y_test, scaler, encoder = preprocess_data(X, y)
    
    print("构建集成模型...")
    model = build_ensemble_model()
    
    print("训练模型...")
    model = train_model(model, X_train, y_train)
    
    print("保存模型...")
    save_model(model, scaler, encoder)
    
    print("训练完成！")


if __name__ == '__main__':
    main()
