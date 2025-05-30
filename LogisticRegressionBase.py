import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.001, max_iter=1000, tol=1e-4, verbose=False):
        """
        优化的对数几率回归实现

        Parameters:
        - learning_rate: 学习率
        - max_iter: 最大迭代次数
        - tol: 收敛阈值
        - verbose: 是否显示训练信息
        """
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def _sigmoid(self, z):
        # 数值稳定的sigmoid实现
        z = np.clip(z, -500, 500)  # 防止数值溢出
        return 1 / (1 + np.exp(-z))

    def _loss(self, y, y_pred, sample_weights):
        # 加权交叉熵损失
        return -np.sum(sample_weights * (y * np.log(y_pred + 1e-10) +
                                         (1 - y) * np.log(1 - y_pred + 1e-10)))

    def fit(self, X, y, sample_weights=None):
        # 确保y在[0,1]范围内（为了兼容性）
        y = (y + 1) / 2  # 转换-1/1到0/1

        n_samples, n_features = X.shape

        # 初始化样本权重
        if sample_weights is None:
            sample_weights = np.ones(n_samples) / n_samples
        else:
            sample_weights = sample_weights / np.sum(sample_weights)  # 归一化

        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # 特征标准化（提高收敛速度）
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-10
        X_normalized = (X - X_mean) / X_std

        # 梯度下降优化
        for i in range(self.max_iter):
            # 前向传播
            linear_model = np.dot(X_normalized, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # 计算损失
            loss = self._loss(y, y_pred, sample_weights)
            self.loss_history.append(loss)

            # 计算梯度
            dw = np.dot(X_normalized.T, (y_pred - y) * sample_weights)
            db = np.sum((y_pred - y) * sample_weights)

            # 更新参数
            prev_weights = np.copy(self.weights)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # 检查收敛
            weight_change = np.linalg.norm(self.weights - prev_weights)
            if weight_change < self.tol:
                if self.verbose:
                    print(f"收敛于第{i+1}次迭代")
                break

            # 学习率衰减
            self.lr *= 0.9999

            if self.verbose and i % 100 == 0:
                print(f"Iteration {i+1}, Loss: {loss:.4f}")

        # 保存标准化参数用于预测
        self.X_mean = X_mean
        self.X_std = X_std

    def predict_proba(self, X):
        """预测概率"""
        X_normalized = (X - self.X_mean) / self.X_std
        linear_model = np.dot(X_normalized, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X):
        """预测类别（返回-1或1）"""
        probas = self.predict_proba(X)
        return np.where(probas >= 0.5, 1, -1)