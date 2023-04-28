import numpy as np
import DecisionTreeRegressor


class GradientBoostingRegressor:
    """
    A gradient boosting regressor.

    Gradient boosting is a machine learning technique for regression problems,
    which produces a prediction model in the form of an ensemble of weak
    prediction models, typically decision trees.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting stages to perform.

    learning_rate : float, default=0.1
        The learning rate shrinks the contribution of each tree.

    max_depth : int, default=3
        The maximum depth of the individual regression estimators.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _negative_gradient(self, y, y_pred):
        return y - y_pred

    def fit(self, X, y):
        self.trees_ = []
        self.tree_weights_ = []

        np.random.seed(self.random_state)

        y_pred = np.mean(y) * np.ones_like(y)
        for _ in range(self.n_estimators):
            residuals = self._negative_gradient(y, y_pred)

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            self.trees_.append(tree)
            self.tree_weights_.append(self.learning_rate)

            y_pred += self.learning_rate * tree.predict(X)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])

        for tree, weight in zip(self.trees_, self.tree_weights_):
            y_pred += weight * tree.predict(X)

        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - np.sum((y_pred - y) ** 2) / np.sum((y - np.mean(y)) ** 2)