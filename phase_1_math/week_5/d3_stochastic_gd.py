import math
import numpy as np
import random

rng = np.random.default_rng()

def sample_func(x):
    return (3*x) + 2 + np.random.normal(0.0, 0.5, size=len(x)) 


def SGD(starting_params, dataset, lr=0.001, batch=64, n_iters=1000, tol=1e-6):

    # we are trying ot minimize the loss between our predicted value of y and true y
    # so our objective function should be the MSE of a linear regression model and our true y 
    # L(w, b) = 1/N \sum_{i=1}^{N} ((wx_{i} + b) - y_{i})^2
    # dL/dw = 2 / N \sum_{i=1}^{N} ((wx_{i} + b) - y_{i}) * x_{i}
    # dL/db = 2 / N \sum_{i=1}^{N} ((wx_{i} + b) - y_{i}) * (1)

    theta_t = starting_params
    grad = starting_params
    step_history = [theta_t]
    t = 0
    
    while (np.linalg.norm(grad) > tol) and t < n_iters:

        batch_samples_idx = random.sample(range(len(dataset)), batch)
        batch_sample = [dataset[i] for i in batch_samples_idx]
        
        # update params based on batches and our predefined partials
        grad_batch = []
        for point in batch_sample:
            x_j, y_j = point[0], point[1]
            curr_w, curr_b = theta_t[0], theta_t[1]
            # compute w first
            grad_w =  ((curr_w * x_j + curr_b) - y_j) * x_j
            grad_b = ((curr_w * x_j + curr_b) - y_j)

            grad_batch.append([grad_w, grad_b])

        grad_L = (2 / float(batch)) * np.sum(np.array(grad_batch), axis=0)
        theta_t_1 = theta_t - lr*grad_L

        # now we do updates
        step_history.append(theta_t_1)
        theta_t = theta_t_1
        t += 1
    
    return step_history, t

def test():

    # generate our dataset of random points
    mean=2.0
    std=0.5
    num_samples = 1000 # starting off with 1000
    x = np.random.normal(mean, std, size=num_samples)
    true_y = sample_func(x)
    dataset = np.column_stack((x, true_y))
    
    init_params = rng.uniform(5.0, 20.0, size=2)
    
    step_history, num_of_iters = SGD(init_params, dataset, batch=32, n_iters=10000)
    print(step_history[-1], num_of_iters)

if __name__ == "__main__":
    test()