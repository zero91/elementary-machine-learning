'''A simple Linear Regression model.'''
from __future__ import division
import sys
import os
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

class LinearRegression(object):
    def __init__(self, X, y, VX, Vy):
        N, W = X.shape
        self.N = N
        self.X = np.c_[np.ones(N), X]
        self.VX = np.c_[np.ones(VX.shape[0]), VX]
        self.Vy = Vy
        self.y = y
        #self.w = np.random.normal(0, 1.0, W + 1)
        self.w = np.zeros(W + 1)

    def train(self, epochs, eta, lmbda):
        cost_list = [list(), list()]
        for epoch in xrange(1, epochs + 1):
            delta = self.X.dot(self.w) - self.y
            cost = sum(delta ** 2) * 0.5 / self.N
            if epoch % 10 == 0:
                cost_list[0].append(cost)
                cost_list[1].append(sum((self.VX.dot(self.w) - self.Vy) ** 2) * 0.5 / len(self.Vy))
            self.w = self.w - eta * self.X.T.dot(delta) / self.N
            #self.w = (1 - eta * lmbda) * self.w - eta * self.X.T.dot(delta) / self.N
            #self.w = self.w - eta * self.X.T.dot(delta) / self.N - eta * np.sign(self.w) / self.N

        pred = sum(self.predict(self.VX) == self.Vy)
        sys.stderr.write("Validate: cost=%f, accuracy=%s(%s/%s)\n" % (\
                        sum((self.VX.dot(self.w) - self.Vy) ** 2) * 0.5 / len(self.Vy),
                        pred / len(self.Vy), pred, len(self.Vy)))
        return self.X.dot(self.w), cost_list

    def predict(self, X, bias=False):
        if bias is True:
            X = np.c_[np.ones(len(X)), X]
        return np.array(map(int, X.dot(self.w) + 0.5))

def normalize(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

if __name__ == "__main__":
    pd_train = pd.read_csv("../data_in/train.csv", sep=';')
    pd_test = pd.read_csv("../data_in/test.csv", sep=';')
    pd_validate = pd.read_csv("../data_in/validate.csv", sep=';')

    train_X = normalize(pd_train.drop('quality', axis=1).values)
    train_y = pd_train['quality'].values

    validate_X = normalize(pd_validate.drop('quality', axis=1).values)
    validate_y = pd_validate['quality'].values

    test_X = normalize(pd_test.drop('quality', axis=1).values)
    test_y = pd_test['quality'].values

    model = LinearRegression(train_X, train_y, validate_X, validate_y)
    pred_y, cost_list = model.train(20000, 0.5, 0.0002)

    test_pred = sum(model.predict(test_X, bias=True) == test_y)
    sys.stderr.write("Accuracy in test data %s(%s/%s)\n" % (\
                        test_pred / len(test_y), test_pred, len(test_y)))

    axis = range(1, len(cost_list[0]) + 1)
    plt.title("Cost for training data and validating data")
    plt.plot(axis, cost_list[0], 'r', axis, cost_list[1], 'b')
    plt.show()
