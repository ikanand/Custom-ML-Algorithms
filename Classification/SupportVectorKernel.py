import numpy as np


class SVM:
    """
    Support Vector Machine with Linear Kernel

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is inversely proportional to C.

    max_iter : int, default=1000
        The maximum number of iterations to be run.
    """

    def __init__(self, C=1.0, max_iter=1000):
        self.C = C
        self.max_iter = max_iter
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.C * (2 * (1 / self.max_iter) * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.C * y_[idx] * (-1)
                else:
                    self.w -= self.C * (2 * (1 / self.max_iter) * self.w)

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)