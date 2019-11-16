#!/usr/bin/env python
# coding: utf-8


import numpy as np
import statistics as s

# var Regression
class Tree:
    def __init__(self, criterion='mse', max_depth=None, min_samples_leaf):
        """
        :param criterion: method to determine splits
        :param max_depth: maximum depth of tree. If None depth of tree is not constrained
        :param min_samples_leaf: the minimum number of samples required to be at a leaf node
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.eps
        self.tree = dict()

    def fit(self, X_train, y_train):
        """
        Fit model using gradient descent method
        :param X_train: training data
        :param y_train: target values for training data
        :return: None
        """
        if not self.max_depth:
            self.max_depth = X_train.shape[0]
        uncertainty = sum([((y - s.mean(y_train)) ** 2) for y in y_train]) / len(y_train)
        depth = 1
        n_samples = len(y_train)
        if uncertainty <= self.eps:

        while depth != self.max_depth and n_samples != self.min_samples_leaf:

            depth += 1
            n_samples = min(X_left, X_right)

    def build(self, X):
        pass

    def predict(self, X_test):
        """
        Predict using model.
        :param X_test: test data for predict in
        :return: y_test: predicted values
        """
        pass

    def get_feature_importance(self):
        """
        Get feature importance from fitted tree
        :return: weights array
        """
        pass


class TreeRegressor(Tree):
    def __init__(self, criterion='mse', max_depth=None, min_samples_leaf=1):
        """
        :param criterion: method to determine splits, 'mse' or 'mae'
        """
        super().__init__(criterion, max_depth, min_samples_leaf)


class TreeClassifier(Tree):
    def __init__(self, criterion='gini', max_depth=None, min_samples_leaf=1):
        """
        :param criterion: method to determine splits, 'gini' or 'entropy'
        """
        super().__init__(criterion, max_depth, min_samples_leaf)

    def predict_proba(self, X_test):
        """
        Predict probability using model.
        :param X_test: test data for predict in
        :return: y_test: predicted probabilities
        """
        pass


class Node():
    def __init__(self, X, feature, threshold, number):
        self.number = number
        self.data = X
        self.feature = feature
        self.threshold
        self.left = None
        self.right = None
