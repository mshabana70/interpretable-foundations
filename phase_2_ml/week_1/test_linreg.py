from linreg import predict, loss, gradient
import numpy as np
import pytest

rng = np.random.default_rng(seed=42)

@pytest.fixture
def create_vars(n=3, d=2):
    X = rng.random((n, d))
    w = rng.random((d, 1))
    y = rng.random((n, 1))
    return (X, y, w)

def test_predict_shape(create_vars):
    X, y, w = create_vars
    y_hat = predict(X, w)
    assert y_hat.shape == y.shape

def test_loss_shape(create_vars):
    X, y, w = create_vars
    loss_val = loss(X, y, w) 
    assert len(loss_val) == 1

def test_gradient_shape(create_vars):
    X, y, w = create_vars
    grad = gradient(X, y, w)
    assert grad.shape == w.shape
 
def test_loss_calculation(create_vars):
    X, y, w = create_vars
    n = int(X.shape[0])
    man_loss = (1 / (2 * n)) * (((X @ w) - y).T @ ((X @ w) - y))
    func_loss = loss(X, y, w)
    np.testing.assert_allclose(man_loss, func_loss) # have to use this np assert func because of floating point precision comparison; good to remember for the future

def test_loss_manual():
    X = np.array([[1, 5], [9, 12]])
    w = np.array([[2], [4]])
    y = np.array([[3], [1]])

    loss_by_hand = np.array([[1146.5]]) # on scratch paper
    func_loss = loss(X, y, w)
    np.testing.assert_allclose(loss_by_hand, func_loss)