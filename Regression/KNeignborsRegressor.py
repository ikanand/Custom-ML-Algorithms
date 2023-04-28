import numpy as np

class KNeighborsRegressor:
    """
    A k-neighbors regressor.

    The target is predicted by local interpolation of the targets associated
    with the nearest neighbors in the training set.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for the prediction.

    metric : str or callable, default='euclidean'
        The distance metric to use. Possible values: 'euclidean', 'manhattan', or a custom callable function.
    """

    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def _distance(self, a, b):
        if self.metric == 'euclidean':
            return np.linalg.norm(a - b)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(a - b))
        elif callable(self.metric):
            return self.metric(a, b)

    def fit(self, X, y):
        """
        Fit the k-neighbors regressor to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict the target for the provided input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted target values.
        """
        y_pred = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            distances = np.array([self._distance(x, x_train) for x_train in self.X_train])
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            y_pred[i] = np.mean(self.y_train[nearest_indices])

        return y_pred

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