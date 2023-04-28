

'''
‘lr’ - Linear Regression

‘lasso’ - Lasso Regression

‘ridge’ - Ridge Regression

‘en’ - Elastic Net

‘lar’ - Least Angle Regression

‘llar’ - Lasso Least Angle Regression

‘omp’ - Orthogonal Matching Pursuit

‘br’ - Bayesian Ridge

‘ard’ - Automatic Relevance Determination

‘par’ - Passive Aggressive Regressor

‘ransac’ - Random Sample Consensus

‘tr’ - TheilSen Regressor

‘huber’ - Huber Regressor

‘kr’ - Kernel Ridge

‘svm’ - Support Vector Regression

‘knn’ - K Neighbors Regressor

‘dt’ - Decision Tree Regressor

‘rf’ - Random Forest Regressor

‘et’ - Extra Trees Regressor

‘ada’ - AdaBoost Regressor

‘gbr’ - Gradient Boosting Regressor

‘mlp’ - MLP Regressor

‘xgboost’ - Extreme Gradient Boosting

‘lightgbm’ - Light Gradient Boosting Machine

‘catboost’ - CatBoost Regressor


Linear Regression
Lasso Regression
Ridge Regression
K Neighbors Regressor
Decision Tree Regressor
Random Forest Regressor
Extra Trees Regressor
Gradient Boosting Regressor
Light Gradient Boosting Machine
Extreme Gradient Boosting

'''


import numpy as np

class Regression:
    def __init__(self, method='linear', degree=2, alpha=0.1, l1_ratio=0.5):
        self.method = method
        self.degree = degree
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def add_ones(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def polynomial_features(self, X):
        return np.hstack([X ** i for i in range(1, self.degree + 1)])

    def fit(self, X, y):
        if self.method == 'polynomial':
            X = self.polynomial_features(X)
        elif self.method == 'bayesian':
            X = self.add_ones(X)
            self.weights = np.linalg.inv(X.T @ X + self.alpha * np.eye(X.shape[1])) @ X.T @ y
            return

        X = self.add_ones(X)
        if self.method == 'linear':
            self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
        elif self.method == 'ridge':
            self.weights = np.linalg.inv(X.T @ X + self.alpha * np.eye(X.shape[1])) @ X.T @ y
        elif self.method == 'lasso':
            self.weights = np.zeros(X.shape[1])
            for _ in range(1000):
                for j in range(X.shape[1]):
                    tmp_weights = self.weights.copy()
                    tmp_weights[j] = 0
                    r_j = y - X @ tmp_weights
                    arg1 = X[:, j].T @ r_j
                    arg2 = self.alpha * X.shape[0] * self.l1_ratio
                    self.weights[j] = np.sign(arg1) * max(0, abs(arg1) - arg2) / (X[:, j].T @ X[:, j])
        elif self.method == 'logistic':
            self.weights = np.zeros(X.shape[1])
            for _ in range(1000):
                y_pred = 1 / (1 + np.exp(-X @ self.weights))
                gradient = X.T @ (y_pred - y)
                self.weights -= 0.01 * gradient

    def predict(self, X):
        if self.method == 'polynomial':
            X = self.polynomial_features(X)
        X = self.add_ones(X)
        if self.method == 'logistic':
            return 1 / (1 + np.exp(-X @ self.weights))
        return X @ self.weights

    def score(self, X, y):
        y_pred = self.predict(X)
        if self.method == 'logistic':
            return np.mean((y_pred > 0.5) == y)
        return 1 - np.sum((y_pred - y) ** 2) / np.sum((y - np.mean(y)) ** 2)
 
 
 
       
