from sklearn.tree import DecisionTreeClassifier
import numpy as np
class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []
        self.loss_history = []  # 记录每轮的指数损失

    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples
        F = np.zeros(n_samples)  # 集成模型的累积输出

        for t in range(self.n_estimators):
            # 训练决策树桩
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=sample_weights)
            predictions = stump.predict(X)

            # 计算错误率和权重alpha
            error = np.sum(sample_weights * (predictions != y))
            error = np.clip(error, 1e-10, 1 - 1e-10)  # 限制error范围
            alpha = 0.5 * np.log((1 - error) / error)

            # 更新样本权重
            sample_weights *= np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)  # 归一化

            # 更新集成模型输出
            F += alpha * predictions

            # 计算当前指数损失（可选）
            loss = np.mean(np.exp(-y * F))
            self.loss_history.append(loss)

            # 保存模型参数
            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])
        weighted_preds = np.dot(self.alphas, stump_preds)
        return np.sign(weighted_preds)