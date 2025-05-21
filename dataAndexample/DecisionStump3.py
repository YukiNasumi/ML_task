import numpy as np

class DecisionStump:
    def __init__(self, criterion='gini', max_depth=1, min_samples_split=2, 
                 min_samples_leaf=1, verbose=False):
        """
        Enhanced Decision Stump with configurable splitting criteria
        
        Parameters:
        - criterion: 'gini', 'entropy', or 'misclassification'
        - max_depth: Maximum depth of the stump (fixed to 1 for stump)
        - min_samples_split: Minimum samples required to split a node
        - min_samples_leaf: Minimum samples required at a leaf node
        - verbose: Whether to print debugging info
        """
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1  # Default direction: x <= t -> +1
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose
        self.classes_ = None
        
    def _calculate_impurity(self, y, weights):
        """Calculate impurity based on selected criterion"""
        if len(y) == 0:
            return 0
        
        # Convert to weighted class counts
        unique_classes = np.unique(y)
        class_counts = np.array([np.sum((y == c) * weights) for c in unique_classes])
        class_probs = class_counts / np.sum(class_counts)
        
        if self.criterion == 'gini':
            return 1 - np.sum(class_probs ** 2)
        elif self.criterion == 'entropy':
            return -np.sum(class_probs * np.log2(class_probs + 1e-10))
        elif self.criterion == 'misclassification':
            return 1 - np.max(class_probs)
        else:
            raise ValueError("Invalid criterion. Choose 'gini', 'entropy', or 'misclassification'")
    
    def _find_best_split(self, X, y, sample_weights):
        """Find the best split point for a single feature"""
        n_samples, n_features = X.shape
        min_impurity = float('inf')
        best_params = {}
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            sorted_indices = np.argsort(feature_values)
            sorted_feature = feature_values[sorted_indices]
            sorted_weights = sample_weights[sorted_indices]
            sorted_y = y[sorted_indices]
            
            # Generate candidate thresholds
            thresholds = []
            for i in range(n_samples - 1):
                if sorted_feature[i] != sorted_feature[i + 1]:
                    threshold = (sorted_feature[i] + sorted_feature[i + 1]) / 2
                    thresholds.append(threshold)
            
            if not thresholds:
                continue  # No valid splits for this feature
                
            for threshold in thresholds:
                for polarity in [1, -1]:
                    if polarity == 1:
                        left_mask = feature_values <= threshold
                    else:
                        left_mask = feature_values > threshold
                        
                    # Check minimum samples constraints
                    n_left = np.sum(left_mask)
                    n_right = n_samples - n_left
                    
                    if (n_left < self.min_samples_leaf or 
                        n_right < self.min_samples_leaf):
                        continue
                        
                    # Calculate weighted impurity
                    y_left = y[left_mask]
                    w_left = sample_weights[left_mask]
                    impurity_left = self._calculate_impurity(y_left, w_left)
                    
                    y_right = y[~left_mask]
                    w_right = sample_weights[~left_mask]
                    impurity_right = self._calculate_impurity(y_right, w_right)
                    
                    total_weight = np.sum(w_left) + np.sum(w_right)
                    if total_weight == 0:
                        continue
                        
                    weighted_impurity = (np.sum(w_left) * impurity_left + 
                                        np.sum(w_right) * impurity_right) / total_weight
                    
                    if weighted_impurity < min_impurity:
                        min_impurity = weighted_impurity
                        best_params = {
                            'feature_idx': feature_idx,
                            'threshold': threshold,
                            'polarity': polarity,
                            'impurity': weighted_impurity
                        }
        
        return best_params
    
    def fit(self, X, y, sample_weights=None):
        """Fit the decision stump to the training data"""
        n_samples, n_features = X.shape
        
        if sample_weights is None:
            sample_weights = np.ones(n_samples) / n_samples
            
        self.classes_ = np.unique(y)
        
        if len(self.classes_) > 2:
            # For multi-class, use one-vs-rest approach
            self._fit_multiclass(X, y, sample_weights)
            return
            
        if n_samples < self.min_samples_split:
            if self.verbose:
                print("Not enough samples to split")
            return
            
        best_split = self._find_best_split(X, y, sample_weights)
        
        if best_split:  # If a valid split was found
            self.feature_idx = best_split['feature_idx']
            self.threshold = best_split['threshold']
            self.polarity = best_split['polarity']
            
            if self.verbose:
                print(f"Best split: Feature {self.feature_idx}, "
                      f"Threshold {self.threshold:.3f}, "
                      f"Polarity {self.polarity}, "
                      f"Impurity {best_split['impurity']:.4f}")
    
    def _fit_multiclass(self, X, y, sample_weights):
        """Handle multi-class classification using one-vs-rest"""
        self.stumps = []
        self.class_weights = []
        
        for c in self.classes_:
            # Create binary problem: current class vs rest
            binary_y = np.where(y == c, 1, -1)
            stump = DecisionStump(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                verbose=self.verbose
            )
            stump.fit(X, binary_y, sample_weights)
            self.stumps.append(stump)
            
            # Calculate class weight (proportion of samples in this class)
            self.class_weights.append(np.sum((y == c) * sample_weights))
        
        # Normalize class weights
        self.class_weights = np.array(self.class_weights)
        self.class_weights /= np.sum(self.class_weights)
        
    def predict(self, X):
        """Predict class labels for samples in X"""
        if self.classes_ is None:
            return np.ones(X.shape[0])  # Default prediction if not fitted
            
        if len(self.classes_) > 2:
            return self._predict_multiclass(X)
            
        if self.feature_idx is None or self.threshold is None:
            return np.ones(X.shape[0])  # Default to positive class
            
        if self.polarity == 1:
            return -np.where(X[:, self.feature_idx] <= self.threshold, 1, -1)
        else:
            return -np.where(X[:, self.feature_idx] > self.threshold, 1, -1)
    
    def _predict_multiclass(self, X):
        """Predict multi-class labels using one-vs-rest"""
        predictions = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, stump in enumerate(self.stumps):
            pred = stump.predict(X)
            # Convert predictions to probabilities
            predictions[:, i] = np.where(pred == 1, self.class_weights[i], 0)
        
        # Return class with highest weighted probability
        return self.classes_[np.argmax(predictions, axis=1)]