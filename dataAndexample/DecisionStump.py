import numpy as np

class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.alpha = None
        self.polarity = 1  # 默认方向：x <= t -> +1

    def fit(self, X, y, sample_weights=None):
        n_samples, n_features = X.shape
        
        # 初始化样本权重（若未提供，则均匀分布）
        if sample_weights is None:
            sample_weights = np.ones(n_samples) / n_samples

        min_error = float('inf')

        # 遍历所有特征
        for feature_idx in range(n_features):
            feature_values = np.unique(X[:, feature_idx])
            # 生成候选阈值（取相邻特征值的中点）
            thresholds = (feature_values[:-1] + feature_values[1:]) / 2

            for threshold in thresholds:
                # 尝试两种分裂方向
                for polarity in [1, -1]:
                    # 预测规则：polarity=1时为x<=t -> +1；polarity=-1时为x>t -> +1
                    predictions = np.where(
                        polarity * X[:, feature_idx] <= polarity * threshold, 
                        1, -1
                    )
                    # 计算加权错误率
                    error = np.sum(sample_weights * (predictions != y))

                    # 更新最优参数
                    if error < min_error:
                        min_error = error
                        self.feature_idx = feature_idx
                        self.threshold = threshold
                        self.polarity = polarity

        # 计算分类器权重（用于AdaBoost）
        self.alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-10))  # 避免除以0

    def predict(self, X):
        # 根据训练好的参数预测
        predictions = np.where(
            self.polarity * X[:, self.feature_idx] <= self.polarity * self.threshold,
            1, -1
        )
        return predictions