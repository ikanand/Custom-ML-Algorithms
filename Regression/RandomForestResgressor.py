import DecisionTreeRegressor
import numpy as np

    
class RandomForestRegressor:
    """
    A random forest regressor.

    A random forest is a meta estimator that fits a number of decision tree
    regressors on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control overfitting.

    Parameters
    ----------
    n_estimators : int, default=10
        The number of trees in the forest.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=n_features`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. Pass an int for reproducible
        output across multiple function calls.
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

    def fit(self, X, y):
        """
        Fit the random forest regressor to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (real numbers).
        """
        self.trees_ = []
        self.feature_indices_ = []

        np.random.seed(self.random_state)

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            X_sample, y_sample = self._bootstrap_sample(X, y)

            if self.max_features is None:
                n_features = X.shape[1]
            elif isinstance(self.max_features, int):
                n_features = self.max_features
            elif isinstance(self.max_features, float):
                n_features = int(self.max_features * X.shape[1])
            elif self.max_features == 'auto':
                n_features = X.shape[1]
            elif self.max_features == 'sqrt':
                n_features = int(np.sqrt(X.shape[1]))
            elif self.max_features == 'log2':
                n_features = int(np.log2(X.shape[1]))

            feature_indices = np.random.choice(X.shape[1], n_features, replace=False)
            self.feature_indices_.append(feature_indices)

            tree.fit(X_sample[:, feature_indices], y_sample)
            self.trees_.append(tree)

    def predict(self, X):
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values.
        """
        predictions = np.zeros((X.shape[0], self.n_estimators))

        for i, tree in enumerate(self.trees_):
            predictions[:, i] = tree.predict(X[:, self.feature_indices_[i]])

        return np.mean(predictions, axis=1)

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True values for X.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return 1 - np.sum((y_pred - y) ** 2) / np.sum((y - np.mean(y)) ** 2)