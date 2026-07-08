import math
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
import kagglehub


def grad_weight(y_pred, y, X):
    return (1 / len(y_pred)) * sum((y_pred - y) * X)

def grad_bias(y_pred, y):
    return (1 / len(y_pred)) * sum(y_pred - y)

def MSE(y_pred, y):
    n = len(y_pred)
    return (1 / (2*n)) * sum((y_pred - y) ** 2)

def linreg_GD(X, y, alpha=1e-2):
    
    # weights and biases, which we will randomly initialize for now
    weight = random.randint(1, 100)
    bias = random.randint(1, 100)


    # all of this needs to happen in the gradient descent loop:
    old_error_rate = 0.0
    new_error_rate = 10.0
    while abs(new_error_rate - old_error_rate) > 1e-3:
        old_error_rate = new_error_rate
        # now we need to produce our predicted y values:
        y_pred = weight * X + bias

        # use derivative of func and of weight
        new_error_rate = MSE(y_pred, y)
        #print(f"Current error: {new_error_rate}")

        # update our weight and bias
        weight_new = weight - (alpha * grad_weight(y_pred, y, X))
        bias_new = bias - (alpha * grad_bias(y_pred, y))

        weight = weight_new
        bias = bias_new
    
    least_weight, least_bias = linreg_normal(X, y)
    # now we can plot the result (best fit line)
    plt.scatter(X, y, color='b')
    plt.plot(X, (weight * X + bias), color='r', linewidth=2, label=f"GD Fit: y = {weight:.2f}x + {bias:.2f}")
    plt.plot(X, (least_weight * X + least_bias), color='green', linewidth=2, label=f"LS Fit: y = {least_weight:.2f}x + {least_bias:.2f}")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.title("Linear Regression w/ Gradient Descent vs. Least Squares Approximation")
    plt.legend()
    plt.savefig('lingreg_gd_ls.png')
    print(f"Generated plot of Linear Regression with Gradient Descent.")
    
def linreg_normal(X_data, y):

    # normal equation: (X^{T}X)^{-1}X^{T}y

    # we need to stack a column of ones to our vector X so that:
    # 1. we can actually do the dot product op
    # 2. Have a column that represents the bias, so we get back two values, not just weights.
    X = np.column_stack((np.ones(len(X_data)), X_data))

    bias, weight = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # great! Now we have two values, the first being the bias, the second being the scalar weight value. Now we can plot
    return weight, bias
    


def test_salary():

    path = kagglehub.dataset_download(
        "abhishek14398/salary-dataset-simple-linear-regression",
        output_dir='./data',
        path='Salary_dataset.csv')

    # now that we have the csv, we can read it to pandas
    df = pd.read_csv(path)
    
    # we want our X values to be the year's of experience, and the target or y to predict as the salary
    X = df['YearsExperience'].to_numpy()
    y = df['Salary'].to_numpy()

    # now we pass it to our linreg model
    print(f"Running Linear Regression with Gradient Descent and Least Squares Approximation...")
    linreg_GD(X, y)

if __name__ == "__main__":
    test_salary()