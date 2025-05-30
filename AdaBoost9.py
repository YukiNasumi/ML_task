import numpy as np
import pandas as pd
from LogisticRegressionBase import LogisticRegression

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        y = np.where(y==0,-1,1)
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            # 根据类型创建基分类器
           
            model = LogisticRegression(max_iter=100)

            # 训练基分类器
            model.fit(X, y, sample_weights)
            predictions = model.predict(X)

            # 计算错误率和alpha
            incorrect = (predictions != y)
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

            # 更新样本权重
            sample_weights *= np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)

            # 保存模型和alpha
            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        model_preds = np.array([model.predict(X) for model in self.models])
        weighted_preds = np.dot(self.alphas, model_preds)
        return np.sign(weighted_preds)