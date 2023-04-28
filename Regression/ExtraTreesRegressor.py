import numpy as np
import DecisionTreeRegressor


class ExtraTreesRegressor:
    """
    An extra trees regressor.

    Extra trees is a meta estimator that fits a number of randomized decision
    trees on various sub-samples of the dataset and uses averaging to improve
    the predictive accuracy and control overfitting.

    Parameters
    ----------
    n_estimators : int, default=10
        The number of trees in the forest.

    max_depth : int, default=None
        The maximum depth of the tree.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, n_estimators=10, max_depth=None, max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _find_random_split(self, X, y, feature):
        values = np.unique(X[:, feature])
        value = np.random.choice(values)
        return feature, value

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return {'prediction': np.mean(y)}

        feature = np.random.randint(X.shape[1])
        value = np.random.choice(np.unique(X[:, feature]))
        left_mask = X[:, feature] <= value
        right_mask = X[:, feature] > value

        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return {'prediction': np.mean(y)}

        return {
            'feature': feature,
            'value': value,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        self.trees_ = []

        np.random.seed(self.random_state)

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.tree_ = self._build_tree(X_sample, y_sample, 0)
            self.trees_.append(tree)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_estimators))

        for i, tree in enumerate(self.trees_):
            predictions[:, i] = tree.predict(X)

        return np.mean(predictions, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - np.sum((y_pred - y) ** 2) / np.sum((y - np.mean(y)) ** 2)