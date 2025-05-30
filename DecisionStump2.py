import numpy as np

class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1  # 默认方向：x <= t -> +1

    def fit(self, X, y, sample_weights=None):
        n_samples, n_features = X.shape
        
        if sample_weights is None:
            sample_weights = np.ones(n_samples) / n_samples

        min_gini = float('inf')

        for feature_idx in range(n_features):
            feature_column = X[:, feature_idx]
            sorted_indices = np.argsort(feature_column)
            sorted_feature = feature_column[sorted_indices]
            sorted_weights = sample_weights[sorted_indices]
            sorted_y = y[sorted_indices]

            thresholds = []
            for i in range(n_samples - 1):
                if sorted_feature[i] != sorted_feature[i + 1]:
                    threshold = (sorted_feature[i] + sorted_feature[i + 1]) / 2
                    thresholds.append(threshold)
            
            if not thresholds:
                continue  # 该特征无可用分割点

            for threshold in thresholds:
                for polarity in [1, -1]:
                    # 根据当前极性和阈值确定分割点
                    if polarity == 1:
                        left_mask = feature_column <= threshold
                    else:
                        left_mask = feature_column > threshold

                    # 计算基尼不纯度
                    weights_left = sample_weights[left_mask]
                    y_left = y[left_mask]
                    pos_left = np.sum((y_left == 1) * weights_left)
                    neg_left = np.sum((y_left == -1) * weights_left)
                    w_left = pos_left + neg_left

                    weights_right = sample_weights[~left_mask]
                    y_right = y[~left_mask]
                    pos_right = np.sum((y_right == 1) * weights_right)
                    neg_right = np.sum((y_right == -1) * weights_right)
                    w_right = pos_right + neg_right

                    total_weight = w_left + w_right
                    if total_weight == 0:
                        continue

                    # 处理空子节点
                    gini_left = 0.0
                    if w_left > 0:
                        p_left = pos_left / w_left
                        gini_left = 1 - (p_left ** 2 + (1 - p_left) ** 2)
                    
                    gini_right = 0.0
                    if w_right > 0:
                        p_right = pos_right / w_right
                        gini_right = 1 - (p_right ** 2 + (1 - p_right) ** 2)
                    
                    weighted_gini = (w_left * gini_left + w_right * gini_right) / total_weight

                    if weighted_gini < min_gini:
                        min_gini = weighted_gini
                        self.feature_idx = feature_idx
                        self.threshold = threshold
                        self.polarity = polarity

    def predict(self, X):
        if self.feature_idx is None or self.threshold is None:
            return np.ones(X.shape[0])  # 默认返回+1，避免随机性
        if self.polarity == 1:
            return np.where(X[:, self.feature_idx] <= self.threshold, 1, -1)
        else:
            return np.where(X[:, self.feature_idx] > self.threshold, 1, -1)