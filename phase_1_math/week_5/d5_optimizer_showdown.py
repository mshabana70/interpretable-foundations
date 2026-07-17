import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# helper funcs
rng = np.random.default_rng()

def get_dataset(path):
    data = pd.read_csv(path)
    # get X set
    X = data.drop(columns=['Performance Index'])
    X['Extracurricular Activities'] = X['Extracurricular Activities'].map({'Yes': 1.0, 'No': 0.0}).astype(float)
    y = data['Performance Index']
    return X, y

def normal_eq(X, y):
    """
    Predicting Performance Index using the normal equation for lin reg.
    """
    theta_pred = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta_pred

def predict(X, theta):
    return X @ theta

def loss(X, y, theta):
    y_hat = predict(X, theta)
    loss = np.mean(((y_hat - y) ** 2))
    return loss

def gradient(X, y, theta):
    residual = predict(X, theta) - y
    return (2 / len(X)) * (X.T @ residual)

# optimizer funcs

def vanilla_GD(X, y, alpha=0.001, n_iters=10000, tol=1e-6):

    theta_t = np.zeros_like(X[0])
    grad = gradient(X, y, theta_t)
    t = 0
    step_history = []
    loss_history = []

    while (abs(np.linalg.norm(grad)) > tol) and t < n_iters:
        
        grad = gradient(X, y, theta_t)
        theta_t_1 = theta_t - alpha*grad

        theta_t = theta_t_1
        curr_loss = loss(X, y, theta_t)
        step_history.append(theta_t)
        loss_history.append(curr_loss)
        t += 1
    
    print(f"[VANILLA] LR = {alpha}; {t} iterations; Final Step: {step_history[-1]}; Final Loss: {loss_history[-1]}")
    return predict(X, step_history[-1])

def momentum_GD(X, y, alpha=0.001, beta=0.9, n_iters=10000, tol=1e-6):
    
    theta_t = np.zeros_like(X[0])
    v_t = np.zeros_like(X[0])
    grad = gradient(X, y, theta_t)
    t = 0
    step_history = []
    loss_history = []

    while (abs(np.linalg.norm(grad)) > tol) and t < n_iters:

        grad = gradient(X, y, theta_t)
        v_t_1 = beta * v_t + alpha*grad
        theta_t_1 = theta_t - v_t_1

        theta_t = theta_t_1
        v_t = v_t_1
        curr_loss = loss(X, y, theta_t)
        step_history.append(theta_t)
        loss_history.append(curr_loss)
        t += 1
    
    print(f"[MOMENTUM] LR = {alpha}; Beta = {beta}; {t} iterations; Final Step: {step_history[-1]}; Final Loss: {loss_history[-1]}")
    return predict(X, step_history[-1])

def nesterov_GD(X, y, alpha=0.001, beta=0.9, n_iters=10000, tol=1e-6):
    theta_t = np.zeros_like(X[0])
    v_t = np.zeros_like(X[0])

    look_ahead = theta_t - beta*v_t
    grad = gradient(X, y, look_ahead)
    t = 0
    step_history = []
    loss_history = []

    while (abs(np.linalg.norm(grad)) > tol) and t < n_iters:

        look_ahead = theta_t - beta*v_t
        grad = gradient(X, y, look_ahead)
        v_t_1 = beta * v_t + alpha*grad
        theta_t_1 = theta_t - v_t_1

        theta_t = theta_t_1
        v_t = v_t_1
        curr_loss = loss(X, y, theta_t)
        step_history.append(theta_t)
        loss_history.append(curr_loss)
        t += 1
    
    print(f"[NESTEROV] LR = {alpha}; Beta = {beta}; {t} iterations; Final Step: {step_history[-1]}; Final Loss: {loss_history[-1]}")
    return predict(X, step_history[-1])

def SGD(X, y, batch=32, alpha=0.001, n_iters=10000):
    
    theta_t = np.zeros_like(X[0])
    step_history = []
    loss_history = []
    t = 0

    while t < n_iters:
        batch_sample_idx = rng.choice(len(X), size=batch, replace=False)
        X_batch = X[batch_sample_idx]
        y_batch = y[batch_sample_idx]
        grad = gradient(X_batch, y_batch, theta_t)

        avg_grad = (1 / len(batch_sample_idx)) * sum(grad)
        theta_t_1 = theta_t - alpha*avg_grad

        theta_t = theta_t_1
        curr_loss = loss(X, y, theta_t) # loss across dataset, not batch
        step_history.append(theta_t)
        loss_history.append(curr_loss)
        t += 1
    
    print(f"[STOCHASTIC] LR = {alpha}; Batch = {batch}; {t} iterations; Final Step: {step_history[-1]}; Final Loss: {loss_history[-1]}")
    return predict(X, step_history[-1])


def Adam(X, y, batch=32, alpha=0.001, beta=[0.9, 0.999], n_iters=10000, epsilon=1e-6):
    
    theta_t = np.zeros_like(X[0])
    curr_m = np.zeros_like(theta_t)
    curr_v = np.zeros_like(theta_t)
    beta_m = beta[0]
    beta_v = beta[1]

    step_history = []
    loss_history = []
    t = 1

    while t < n_iters:
        batch_sample_idx = rng.choice(len(X), size=batch, replace=False)
        X_batch = X[batch_sample_idx]
        y_batch = y[batch_sample_idx]
        grad = gradient(X_batch, y_batch, theta_t)
        avg_grad = (1 / len(batch_sample_idx)) * sum(grad)
        
        m_t = beta_m * curr_m + (1 - beta_m) * avg_grad
        v_t = beta_v * curr_v + (1 - beta_v) * avg_grad
        m_hat_t = m_t / (1 - (beta_m ** t))
        v_hat_t = v_t / (1 - (beta_v ** t))

        theta_t_1 = theta_t - alpha * (m_hat_t / (np.sqrt(v_hat_t) + epsilon))

        theta_t = theta_t_1
        curr_m = m_t
        curr_v = v_t

        curr_loss = loss(X, y, theta_t) # loss across dataset, not batch
        step_history.append(theta_t)
        loss_history.append(curr_loss)
        t += 1
    
    print(f"[ADAM] LR = {alpha}; Beta = {beta}; Batch = {batch}; {t} iterations; Final Step: {step_history[-1]}; Final Loss: {loss_history[-1]}")
    return predict(X, step_history[-1])

def train(optimizer_fn, X, y, **kwargs):
    
    # run the training and return the predicted values
    return optimizer_fn(X, y, **kwargs)


if __name__ == "__main__":
    
    print(f"Loading Student Performance Dataset...")
    path = "./data/Student_performance.csv"
    X, y = get_dataset(path)

    # convert format of data to numpy
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    # prepend X with a column of 1s for our bias term
    X_aug = np.hstack((np.ones((X.shape[0], 1)), X)) # shape => N x (d + 1)

    normal_pred = normal_eq(X_aug, y_np)
    vanilla_GD_preds = train(vanilla_GD, X=X_aug, y=y_np)
    momentum_GD_preds = train(momentum_GD, X=X_aug, y=y_np, beta=0.9)
    nesterov_GD_preds = train(nesterov_GD, X=X_aug, y=y_np, beta=0.9)
    SGD_preds = train(SGD, X=X_aug, y=y_np, batch=32)
    Adam_preds = train(Adam, X=X_aug, y=y_np, batch=32, alpha=0.01)

    y_preds_unnormalized = {
        "vanilla": vanilla_GD_preds,
        "momentum": momentum_GD_preds,
        "nesterov": nesterov_GD_preds,
        "SGD": SGD_preds,
        "Adam": Adam_preds
    }

    # normalize our features then rerun:




    



