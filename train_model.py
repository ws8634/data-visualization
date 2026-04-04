"""模型训练模块

本模块负责加载数据、训练多种机器学习模型，并将它们集成起来。
包含决策树和随机森林两种算法，并使用投票集成方法。
"""

import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier


def load_data():
    """加载数据集
    
    Returns:
        X_train: 训练特征
        X_test: 测试特征
        y_train: 训练标签
        y_test: 测试标签
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """训练多种模型并集成
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
    
    Returns:
        ensemble_model: 集成模型
    """
    # 定义单个模型
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(random_state=42)
    
    # 定义集成模型
    ensemble_model = VotingClassifier(
        estimators=[('dt', dt), ('rf', rf)],
        voting='hard'
    )
    
    # 训练模型
    ensemble_model.fit(X_train, y_train)
    
    return ensemble_model


def save_model(model, filename='model.pkl'):
    """保存模型
    
    Args:
        model: 训练好的模型
        filename: 保存文件名
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存至 {filename}")


if __name__ == "__main__":
    # 加载数据
    X_train, X_test, y_train, y_test = load_data()
    
    # 训练模型
    model = train_models(X_train, y_train)
    
    # 保存模型
    save_model(model)