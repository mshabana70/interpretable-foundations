import numpy as np


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

    return X @ w # output shape should be n x 1 vector, matching shape of y

def loss(X, y, w):
    # From our previous derivation, the loss:
    # L(w) = (1 / 2n)(Xw - y)^{T}(Xw - y)
    n = int(X.shape[0]) # num of rows
    exp_t = predict(X, w) - y 
    loss = (1 / (2 * n)) * (exp_t.T @ (predict(X, w) - y))
    return loss.item() # output shape should be 1 x 1 vector, adding .item() just returns it as a scalar


def gradient(X, y, w):
    # our derivation shows how to compute the gradient of the loss w.r.t w
    n = int(X.shape[0])
    return (1 / n) * (X.T @ (predict(X, w) - y)) # output shape should be d x 1

def numerical_gradient(X, y, w, h=1e-5):
    # what if we don't have the gradient pre-computed??
    # we can use the central finite-difference method to do this.
    # we already know the loss function, and we know the value we are trying to compute the gradient on (the weight vec w),
    # so we just need to pick a value for our small step in either direction of point w => h.
    # we also need a vec e_j so that we only step some h*e_j amount for component w_j.
    grad_vec = np.zeros(w.shape)
    d = w.shape[0]
    e =  np.zeros(w.shape)
    for j in range(d):
        e[j] = 1.0 # set the index to 1 for the component we want to compute the grad on
        w_plus = w.copy() + (h * e)
        w_minus = w.copy() - (h * e)

        plus_loss = loss(X, y, w_plus)
        minus_loss = loss(X, y, w_minus)

        grad_val = (plus_loss - minus_loss) / (2 * h)

        grad_vec[j] = grad_val

        # reset our e vec for next iter
        e[j] = 0.0

    return grad_vec

