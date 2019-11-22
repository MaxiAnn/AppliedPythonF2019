#!/usr/bin/env python
# coding: utf-8


import numpy as np
import statistics as s

class Tree:
    def __init__(self, criterion='mse', max_depth=None, min_samples_leaf=1):
        """
        :param criterion: method to determine splits
        :param max_depth: maximum depth of tree. If None depth of tree is not constrained
        :param min_samples_leaf: the minimum number of samples required to be at a leaf node
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.depth = 0
        self.feature_importance = []
        self.n_samples = 0
        
        self.iter = 0

    def fit(self, X_train, y_train):
        """
        Fit model
        :param X_train: training data
        :param y_train: target values for training data
        :return: None
        """
        self.n_samples = X_train.shape[0]
        if not self.max_depth:
            self.max_depth = X_train.shape[0]
        self.feature_importance = [0 for _ in range(X_train.shape[1])]

        self.depth += 1
        self.root = Node()
        self.devide(X_train, y_train, self.root)

    def def_uncertainty(self, y_set):
        if self.criterion == 'mse':
            return sum([((y - s.mean(y_set)) ** 2) for y in y_set]) / len(y_set)
        else:
            return sum([abs(y - s.mean(y_set)) for y in y_set]) / len(y_set)

    def devide(self, x, y, node):
        print('Iteration', self.iter)
        self.iter += 1
        uncert = self.def_uncertainty(y)
        if self.depth < self.max_depth and (self.min_samples_leaf * 2 <= x.shape[0]):
            self.depth += 1
            quality = 0
            for ind_feature in range(x.shape[1]):
                values = np.unique(x[:, ind_feature])
                for thres in values:
                    n_left = x[x[:, ind_feature] <= thres].shape[0]
                    n_right = x[x[:, ind_feature] > thres].shape[0]
                    if n_left >= self.min_samples_leaf and n_right >= self.min_samples_leaf:
                        y_left = [y[i] for i in range(x.shape[0]) if x[i][ind_feature] <= thres]
                        y_right = [y[i] for i in range(x.shape[0]) if x[i][ind_feature] > thres]
                        new_uncert = (len(y_left) * self.def_uncertainty(y_left) + len(y_right) * self.def_uncertainty(y_right)) / x.shape[0]
                        if quality < uncert - new_uncert:
                            quality = uncert - new_uncert
                            imp_feature = ind_feature
                            threshold = thres
                    else:
                        node.y = s.median(x[:, node.feature])
                        return True
            node.feature = imp_feature
            node.threshold = threshold
            self.feature_importance[imp_feature] += quality * x[x[:, imp_feature] <= threshold].shape[0]
            node.left = Node(imp_feature, threshold)
            self.devide(x[x[:, imp_feature] <= threshold], y[x[:, imp_feature] <= threshold], node.left)
            node.right = Node(imp_feature, threshold)
            self.devide(x[x[:, imp_feature] > threshold], y[x[:, imp_feature] > threshold], node.right)
        else:
            node.y = s.median(x[:, node.feature])
            return True



    def predict(self, X_test):
        """
        Predict using model.
        :param X_test: test data for predict in
        :return: y_test: predicted values
        """
        def pr(node, x):
            if not node.left and not node.right:
                return node.y
            else:
                if x[node.feature] <= node.threshold and not node.left:
                    pr(node.left, x)
                else:
                    pr(node.right, x)
        ans = pr(self.root, X_test)
        return ans

    def get_feature_importance(self):
        """
        Get feature importance from fitted tree
        :return: weights array
        """
        return self.feature_importance / self.n_samples


class TreeRegressor(Tree):
    def __init__(self, criterion='mse', max_depth=None, min_samples_leaf=1):
        """
        :param criterion: method to determine splits, 'mse' or 'mae'
        """
        super().__init__(criterion, max_depth, min_samples_leaf)

class TreeClassifier(Tree):
    def __init__(self, criterion='mse', max_depth=None, min_samples_leaf=1):
        raise NotImplementedError

class Node():
    def __init__(self, feature=None, threshold=None):
        self.feature = feature
        self.threshold = threshold
        self.left = None
        self.right = None
        self.y = None

X = np.array(([[1, 2, 3], [4, 5, 7], [1, 2, 6], [3, 8, 7], [2, 3, 4], [7, 5, 4], [6, 4, 3], [1, 7, 9], [2, 3, 7]]))
y = np.array([1, 7, 5, 3, 2, 3, 6, 4, 7])
tr = Tree()
tr.fit(X, y)
print(tr.predict([3, 5, 2]))