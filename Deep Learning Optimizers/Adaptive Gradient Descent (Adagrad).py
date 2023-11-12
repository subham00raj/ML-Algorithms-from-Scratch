import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Adagrad:

    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = None
        self.squared_gradient_coef_ = None
        self.squared_gradient_intercept_ = None
        self.y_pred = None

    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.coef_ = np.zeros((X_train.shape[1],))
        self.intercept_ = 0
        self.velocity_coef_ = np.zeros_like(self.coef_)
        self.velocity_intercept_ = 0

        epsilon = 1e-8  # Small constant to avoid division by zero

        for epoch in range(self.epochs):
                y_hat = np.dot(self.coef_, X_train.T) + self.intercept_

                dj_dw = np.dot((y_hat - y_train), X_train) / X_train.shape[0]
                dj_db = -2 * np.mean(y_train - y_hat)

                # Adagrad updates
                self.velocity_coef_ += dj_dw ** 2
                self.velocity_intercept_ += dj_db ** 2

                self.coef_ = self.coef_ - (self.learning_rate / (np.sqrt(self.velocity_coef_) + epsilon)) * dj_dw
                self.intercept_ = self.intercept_ - (self.learning_rate / (np.sqrt(self.velocity_intercept_) + epsilon)) * dj_db

        return self.coef_, self.intercept_

    def predict(self, X_test):
        self.y_pred = np.dot(self.coef_, X_test.T) + self.intercept_
        return self.y_pred

    @staticmethod
    def mean_squared_error(y_test, y_pred):
        return ((y_test - y_pred) ** 2).mean()

    @staticmethod
    def r2_score(y_test, y_pred):
        y_mean = np.mean(y_test)
        ss_total = np.sum((y_test - y_mean) ** 2)
        ss_residual = np.sum((y_test - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2