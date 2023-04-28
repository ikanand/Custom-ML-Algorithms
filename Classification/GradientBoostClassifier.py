import numpy as np
import DecisionTreeClassifier
class GradientBoostingClassifier:
    """
    Gradient Boosting Classifier

    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting stages to perform.

    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree.

    max_depth : int, default=3
        The maximum depth of the individual regression estimators.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        y_encoded = self._one_hot_encode(y)
        initial_pred = np.full(y_encoded.shape, 1 / self.n_classes_)
        self.trees = []

        for _ in range(self.n_estimators):
            tree_group = []
            for class_idx in range(self.n_classes_):
                residuals = y_encoded[:, class_idx] - initial_pred[:, class_idx]
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                tree.fit(X, residuals)
                tree_group.append(tree)
            self.trees.append(tree_group)
            initial_pred += self.learning_rate * np.column_stack([tree.predict(X) for tree in tree_group])

    def predict(self, X):
        initial_pred = np.full((X.shape[0], self.n_classes_), 1 / self.n_classes_)
        for tree_group in self.trees:
            initial_pred += self.learning_rate * np.column_stack([tree.predict(X) for tree in tree_group])
        return np.argmax(initial_pred, axis=1)

    def _one_hot_encode(self, y):
        y_encoded = np.zeros((y.size, self.n_classes_))
        y_encoded[np.arange(y.size), y] = 1
        return y_encoded