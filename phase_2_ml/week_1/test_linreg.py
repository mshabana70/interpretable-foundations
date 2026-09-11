from linreg import predict, loss, gradient
import numpy as np
import pytest

rng = np.random.default_rng()

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
 
