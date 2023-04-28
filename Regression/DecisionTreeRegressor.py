import numpy as np


class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def _find_best_split(self, X, y):
        best_feature = None
        best_value = None
        best_cost = float('inf')

        for feature in range(X.shape[1]):
            for value in np.unique(X[:, feature]):
                left_mask = X[:, feature] <= value
                right_mask = X[:, feature] > value
                
                left_cost = np.sum((y[left_mask] - np.mean(y[left_mask])) ** 2)
                right_cost = np.sum((y[right_mask] - np.mean(y[right_mask])) ** 2)
                total_cost = left_cost + right_cost

                if total_cost < best_cost:
                    best_feature = feature
                    best_value = value
                    best_cost = total_cost

        return best_feature, best_value

    def _split(self, X, y, feature, value):
        mask = X[:, feature] <= value
        return X[mask], y[mask], X[~mask], y[~mask]

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return {'prediction': np.mean(y)}

        feature, value = self._find_best_split(X, y)
        left_X, left_y, right_X, right_y = self._split(X, y, feature, value)

        if len(left_y) == 0 or len(right_y) == 0:
            return {'prediction': np.mean(y)}

        return {
            'feature': feature,
            'value': value,
            'left': self._build_tree(left_X, left_y, depth + 1),
            'right': self._build_tree(right_X, right_y, depth + 1)
        }

    def fit(self, X, y):
        self.tree_ = self._build_tree(X, y, 0)

    def _predict_sample(self, x, tree):
        if 'prediction' in tree:
            return tree['prediction']

        feature = tree['feature']
        value = tree['value']

        if x[feature] <= value:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree_) for x in X])

    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - np.sum((y_pred - y) ** 2) / np.sum((y - np.mean(y)) ** 2)