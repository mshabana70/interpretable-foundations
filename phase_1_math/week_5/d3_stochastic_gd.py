import math
import numpy as np
import random
import matplotlib.pyplot as plt

rng = np.random.default_rng()

def sample_func(x):
    return (3*x) + 2 + rng.normal(0.0, 0.5, size=len(x)) 

def eval_loss(point, dataset):
    x = dataset[:, 0]
    true_y = dataset[:, 1]
    w, b = point[0], point[1]
    avg_loss = (1 / len(dataset)) * sum([((w * x[i] + b) - true_y[i]) ** 2 for i in range(len(dataset))])
    return avg_loss

def SGD(starting_params, dataset, lr=0.001, batch=64, n_iters=1000, tol=0.2):

    # we are trying ot minimize the loss between our predicted value of y and true y
    # so our objective function should be the MSE of a linear regression model and our true y 
    # L(w, b) = 1/N \sum_{i=1}^{N} ((wx_{i} + b) - y_{i})^2
    # dL/dw = 2 / N \sum_{i=1}^{N} ((wx_{i} + b) - y_{i}) * x_{i}
    # dL/db = 2 / N \sum_{i=1}^{N} ((wx_{i} + b) - y_{i}) * (1)

    theta_t = starting_params
    step_history = [theta_t]
    curr_avg_loss = 5.0
    loss_history = []
    t = 0
    
    while t < n_iters:

        batch_sample = rng.choice(dataset, size=batch, replace=False)
        
        # update params based on batches and our predefined partials
        grad_batch = []
        for point in batch_sample:
            x_j, y_j = point[0], point[1]
            curr_w, curr_b = theta_t[0], theta_t[1]
            # compute w first
            grad_w =  ((curr_w * x_j + curr_b) - y_j) * x_j
            # now compute b
            grad_b = ((curr_w * x_j + curr_b) - y_j)

            grad_batch.append([grad_w, grad_b])

        grad_L = (2 / float(batch)) * np.sum(np.array(grad_batch), axis=0)
        theta_t_1 = theta_t - lr*grad_L

        # now we do updates
        step_history.append(theta_t_1)
        theta_t = theta_t_1
        
        # our condition check is going to be on the current average loss over our batch:
        #curr_avg_loss = eval_loss(theta_t_1, dataset) # if we want smooth convergence curves
        curr_avg_loss = (1 / float(batch)) * sum([((theta_t[0] * batch_point[0] + theta_t[1]) - batch_point[1]) ** 2 for batch_point in batch_sample])
        loss_history.append(curr_avg_loss)
        t += 1
    
    print(f"Mini-Batch SGD (batch = {batch}) completed with final loss of {loss_history[-1]} in {t} iterations. w = {step_history[-1][0]}, b = {step_history[-1][1]}")
    return step_history, loss_history, t

def plot(step_history, loss_history, dataset, ax=None, color="#ff9500", label=None):
    traj = np.array(step_history)
    fx = np.array(loss_history)
    if ax is None: _, ax = plt.subplots()
    ax.semilogy(fx, color=color, label=label)          # <-- semilogy, not plot
    ax.set_xlabel("iteration"); ax.set_ylabel("f(x, y)  (log)")
    ax.legend(); ax.grid(alpha=0.3)
    return ax

def test():

    # generate our dataset of random points
    mean=0.0
    std=0.5
    num_samples = 1000 # starting off with 1000
    x = np.random.normal(mean, std, size=num_samples)
    true_y = sample_func(x)
    dataset = np.column_stack((x, true_y))
    
    init_params = rng.uniform(5.0, 20.0, size=2)
    
    step_history_1, loss_history_1, num_of_iters_1 = SGD(init_params, dataset, batch=1, n_iters=20000)
    step_history_32, loss_history_32, num_of_iters_32 = SGD(init_params, dataset, batch=32, n_iters=20000)
    step_history_64, loss_history_64, num_of_iters_64 = SGD(init_params, dataset, batch=64, n_iters=20000)
    step_history_128, loss_history_128, num_of_iters_128 = SGD(init_params, dataset, batch=128, n_iters=20000)
    step_history_1000, loss_history_1000, num_of_iters_1000 = SGD(init_params, dataset, batch=1000, n_iters=20000)

    ax_loss_plot = plot(step_history_1, loss_history_1, dataset, label=f"SGD Batch 1; Done in {num_of_iters_1}")
    plot(step_history_32, loss_history_32, dataset, ax=ax_loss_plot, color="#ff0000", label=f"SGD Batch 32; Done in {num_of_iters_32}")
    plot(step_history_64, loss_history_64, dataset, ax=ax_loss_plot, color="#ff00ae", label=f"SGD Batch 64; Done in {num_of_iters_64}")
    plot(step_history_128, loss_history_128, dataset, ax=ax_loss_plot, color="#3700ff", label=f"SGD Batch 128; Done in {num_of_iters_128}")
    plot(step_history_1000, loss_history_1000, dataset, ax=ax_loss_plot, color="#090610", label=f"SGD Batch 1000; Done in {num_of_iters_1000}")
    ax_loss_plot.figure.savefig("./figs/D3_SGD_convergence_over_batches_plot.png")



if __name__ == "__main__":
    test()