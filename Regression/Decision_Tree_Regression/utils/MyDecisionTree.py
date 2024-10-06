import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split:
            return np.mean(y)

        # Find the best split
        best_split = self._find_best_split(X, y)
        if best_split is None:
            return np.mean(y)

        # Split the data
        left_indices = X[:, best_split['feature_index']] < best_split['threshold']
        right_indices = ~left_indices

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            'feature_index': best_split['feature_index'],
            'threshold': best_split['threshold'],
            'left': left_tree,
            'right': right_tree
        }

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_se = float('inf')
        best_split = None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] < threshold
                right_indices = ~left_indices

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                se = self._calculate_se(y[left_indices], y[right_indices])
                if se < best_se:
                    best_se = se
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold
                    }

        return best_split

    def _calculate_se(self, left, right):
        total_samples = len(left) + len(right)
        left_mean = np.mean(left) if len(left) > 0 else 0
        right_mean = np.mean(right) if len(right) > 0 else 0
        se = (np.sum((left - left_mean) ** 2) + np.sum((right - right_mean) ** 2))
        return se

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _predict_sample(self, sample, tree):
        if isinstance(tree, dict):
            if sample[tree['feature_index']] < tree['threshold']:
                return self._predict_sample(sample, tree['left'])
            else:
                return self._predict_sample(sample, tree['right'])
        else:
            return tree