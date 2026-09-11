import numpy as np
import json


# TODO: going to first start with the three core funcs
# - predict(X, w)
# - loss(X, y, w)
# - gradient(X, y, w)
# The shapes of our inputs are:
# X (n x d matrix)
# w (d x 1 vector)
# y (n x 1 vector)


def predict(X, w):
    """
    This func is going to tell us what y_hat is. Usually thats a simple expression like y_hat = (weight)* x + bias.

    Since there is no bias value yet, y_hat = X * w
    """

    return np.dot(X, w) # output shape should be n x 1 vector, matching shape of y

def loss(X, y, w):
    # From our previous derivation, the loss:
    # L(w) = (1 / 2n)(Xw - y)^{T}(Xw - y)
    n = int(X.shape[0]) # num of rows
    exp_t = np.dot(X, w) - y 
    loss = (1 / 2) * n * (exp_t.T @ (np.dot(X, w) - y))
    return loss # output shape should be 1 x 1 scalar


def gradient(X, y, w):
    # our derivation shows how to compute the gradient of the loss w.r.t w
    n = int(X.shape[0])
    return (1 / n) * (X.T @ (np.dot(X, w) - y)) # output shape should be d x 1

