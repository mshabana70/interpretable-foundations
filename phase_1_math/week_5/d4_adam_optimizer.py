import numpy as np
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

def AdamOptimizer(starting_params, dataset, betas=[0.9, 0.999], alpha=0.001, batch=64, n_iters=10000, epsilon=1e-6):
    
    # we must initialize our momentum and adaptive step terms first
    N = len(dataset)
    beta_m, beta_v = betas[0], betas[1]
    theta_t = starting_params
    curr_m, curr_v = np.zeros_like(theta_t)

    step_history = []
    loss_history = []
    t = 1

    while t < n_iters:
        # lets compute our gradients first, then update our terms
        # we will implement batching again here first
        batch_sample = rng.choice(dataset, size=batch, replace=False)
        
        grad_batch = []
        for point in batch_sample:
            x_i, y_i = point[0], point[1]
            curr_w, curr_b = theta_t[0], theta_t[1]

            grad_w = ((curr_w * x_i + curr_b) - y_i) * x_i
            grad_b = ((curr_w * x_i + curr_b) - y_i)

            grad_batch.append([grad_w, grad_b])

        grad_L = (2 / batch) * np.sum(np.array(grad_batch), axis=0)

        # now we do the momentum and rmsprop term updates
        m_t = (beta_m * curr_m) + ((1 - beta_m) * grad_L)
        v_t = (beta_v * curr_v) + ((1 - beta_v) * (grad_L ** 2))

        # need to do bias correction before we do gradient update
        m_hat_t = m_t / (1 - (beta_m ** t))
        v_hat_t = v_t / (1 - (beta_v ** t))

        # compute grad update
        theta_t -= (alpha * (m_hat_t / (np.sqrt(v_hat_t) + epsilon)))

        # now we update our terms for the next cycle
        curr_m = m_hat_t
        curr_v = v_hat_t

        avg_loss = (1 / batch) * sum([(theta_t[0] * point[0] + theta_t[1]) - point[1] for point in batch_sample])
        step_history.append(theta_t)
        loss_history.append(avg_loss)
        t += 1
    
    print(f"Adam Optimizer (batch = {batch}, t = {t}) converged on {loss_history[-1]}; w = {step_history[-1][0]}, b = {step_history[-1][1]}.")
    return step_history, loss_history, t

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
    adam_step_hist, adam_loss_hist, adam_iters = AdamOptimizer(init_params, dataset, batch=32, n_iters=20000)
    SGD_step_hist, SGD_loss_hist, SGD_iters = SGD(init_params, dataset, batch=32, n_iters=20000)

    # get a plot to compare loss convergence
    ax_conv = plot(adam_step_hist, adam_loss_hist, dataset, color="#1900ff", label=f"Adam (batch = 32; iters = {adam_iters})")
    plot(SGD_step_hist, SGD_loss_hist, dataset, ax=ax_conv, color="#ff0000", label=f"SGD (batch = 32; iters = {adam_iters})")
    ax_conv.figure.savefig("./figs/D4_adam_vs_SGD_loss_plot.png")

if __name__ == "__main__":
    test()