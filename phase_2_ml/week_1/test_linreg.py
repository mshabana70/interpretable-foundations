import numpy as np
import pandas as pd
import pytest
import os

from linreg import predict, loss, gradient, numerical_gradient, fit_gradient_descent, fit_standardizer
import kagglehub
from pathlib import Path

rng = np.random.default_rng(seed=42)

# data download, only if it doesn't exist already
@pytest.fixture
def create_dataset():
    data_path = Path("data/vehicle_dataset")
    if os.makedirs(data_path, exist_ok=True):
        return "data/vehicle_dataset/datasets/nehalbirla/vehicle-dataset-from-cardekho/versions/4"
    os.environ["KAGGLEHUB_CACHE"] = str(data_path)
    full_path = kagglehub.dataset_download("nehalbirla/vehicle-dataset-from-cardekho")
    print(f"Dataset ready at {full_path}")
    return full_path

@pytest.fixture
def create_vars(n=3, d=2):
    X = rng.standard_normal((n, d))
    w = rng.random((d, 1))
    y = rng.standard_normal((n, 1))
    return (X, y, w)

def test_predict_shape(create_vars):
    X, y, w = create_vars
    y_hat = predict(X, w)
    assert y_hat.shape == y.shape

def test_loss_shape(create_vars):
    X, y, w = create_vars
    loss_val = loss(X, y, w) 
    assert isinstance(loss_val, float)

def test_gradient_shape(create_vars):
    X, y, w = create_vars
    grad = gradient(X, y, w)
    assert grad.shape == w.shape

def test_numerical_gradient_shape(create_vars):
    X, y, w = create_vars
    num_grad = numerical_gradient(X, y, w)
    assert num_grad.shape == w.shape
 
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

def test_numerical_gradient_calculation(create_vars):
    X, y, w = create_vars

    predef_grad = gradient(X, y, w)
    num_grad = numerical_gradient(X, y, w)
    np.testing.assert_allclose(predef_grad, num_grad)

def test_training_loop_small_dataset(create_vars):
    X, y, w = create_vars

    final_weights, loss_vals = fit_gradient_descent(X, y, w=w)
    initial_loss = loss(X, y, w)
    final_loss = loss_vals[-1]
    assert initial_loss > final_loss

def test_training_loop_real_dataset(create_dataset):
    # now we test our linreg training on a REAL dataset.
    # going to use the vehicle dataset from kaggle: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho    

    data_path = create_dataset
    vehicle_df = pd.read_csv(data_path + "/" + "car_data.csv")

    # now we build our X matrix and y vector
    X = vehicle_df[["Year", "Present_Price", "Kms_Driven"]].to_numpy()
    y = vehicle_df['Selling_Price'].to_numpy()[:, None] # have to add this so its actually a (301, 1) vector

    print(f"Cleaned up dataset values: X = {X.shape}, y = {y.shape}")

    # adding assertions for the shapes so I avoid shape mismatch errors during training loop
    assert X.ndim == 2
    assert y.shape == (X.shape[0], 1)

    # before we do training let's standardize our dataset
    X_scaled, stats = fit_standardizer(X)
    print(f"Rescaled dataset shape: {X_scaled.shape}")
    assert X_scaled.ndim == 2
    assert X_scaled.shape == (X.shape[0], X.shape[1] + 1)

    # we have values for our variables so now we can test our training loop
    final_weights, loss_vals = fit_gradient_descent(X_scaled, y, lr=1e-1, iters=100)

    # we are going to compare against numpy's linalg.lstsq method
    numpy_weights = np.linalg.lstsq(X_scaled, y, rcond=None)[0]
    np.testing.assert_allclose(final_weights, numpy_weights)



