import numpy as np

class LogisticRegressionBase:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4):
        """
        初始化对数几率回归分类器
        :param learning_rate: 学习率（梯度下降步长）
        :param max_iter: 最大迭代次数
        :param tol: 损失函数收敛阈值
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None  # 模型参数（权重向量）
        self.bias = None     # 偏置项

    def _sigmoid(self, z):
        """Sigmoid 函数，将线性输出映射到 [0, 1]"""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, sample_weight=None):
        """
        带样本权重的梯度下降训练
        :param X: 特征矩阵 (n_samples, n_features)
        :param y: 标签向量 (n_samples,), 取值 {-1, 1} 或 {0, 1}
        :param sample_weight: 样本权重 (n_samples,), 默认等权重
        """
        n_samples, n_features = X.shape
        
        # 初始化权重和偏置
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 若样本权重未提供，则设为均匀分布
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        
        # 将标签统一转换为 {0, 1}（方便计算交叉熵损失）
        y = (y > 0).astype(int)
        
        prev_loss = np.inf
        for _ in range(self.max_iter):
            # 计算当前预测概率
            linear_output = np.dot(X, self.weights) + self.bias
            probabilities = self._sigmoid(linear_output)
            
            # 计算加权交叉熵损失
            loss = -np.sum(sample_weight * (y * np.log(probabilities + 1e-10) + 
                                          (1 - y) * np.log(1 - probabilities + 1e-10)))
            
            # 检查收敛
            if np.abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
            
            # 计算梯度（带样本权重）
            dw = np.dot(X.T, sample_weight * (probabilities - y))  # 权重梯度
            db = np.sum(sample_weight * (probabilities - y))      # 偏置梯度
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self

    def predict(self, X):
        """
        预测类别（二分类，返回 {-1, 1}）
        :param X: 特征矩阵 (n_samples, n_features)
        :return: 预测标签 (n_samples,)
        """
        linear_output = np.dot(X, self.weights) + self.bias
        probabilities = self._sigmoid(linear_output)
        return np.where(probabilities >= 0.5, 1, -1)  # 默认阈值 0.5