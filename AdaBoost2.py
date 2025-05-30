from sklearn.tree import DecisionTreeClassifier
import numpy as np

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators  # Number of weak learners
        self.stumps = []  # List to store weak learners (decision stumps)
        self.alphas = []  # List to store weights for each weak learner
        self.loss_history = []  # To track exponential loss during training

    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples  # Initialize uniform weights
        F = np.zeros(n_samples)  # Ensemble model's cumulative predictions

        for t in range(self.n_estimators):
            # 1. Train a weak learner (decision stump)
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=sample_weights)
            predictions = stump.predict(X)

            # 2. Calculate weighted error and alpha
            incorrect = (predictions != y)
            error = np.sum(sample_weights * incorrect)
            
            # Handle edge cases where error is 0 or 1
            error = np.clip(error, 1e-10, 1 - 1e-10)
            alpha = 0.5 * np.log((1 - error) / error)

            # 3. Update sample weights
            sample_weights *= np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)  # Normalize

            # 4. Update ensemble predictions
            F += alpha * predictions

            # 5. Calculate and store exponential loss (optional for monitoring)
            current_loss = np.mean(np.exp(-y * F))
            self.loss_history.append(current_loss)

            # 6. Save the weak learner and its weight
            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        # Get predictions from all stumps
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])
        
        # Weighted sum of predictions
        weighted_preds = np.dot(self.alphas, stump_preds)
        
        # Final prediction (sign of the weighted sum)
        return np.sign(weighted_preds)