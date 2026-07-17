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
    return predict(X, step_history[-1]), loss_history

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
    return predict(X, step_history[-1]), loss_history

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
    return predict(X, step_history[-1]), loss_history

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

        theta_t_1 = theta_t - alpha*grad

        theta_t = theta_t_1
        curr_loss = loss(X, y, theta_t) # loss across dataset, not batch
        step_history.append(theta_t)
        loss_history.append(curr_loss)
        t += 1
    
    print(f"[STOCHASTIC] LR = {alpha}; Batch = {batch}; {t} iterations; Final Step: {step_history[-1]}; Final Loss: {loss_history[-1]}")
    return predict(X, step_history[-1]), loss_history


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
        
        m_t = beta_m * curr_m + (1 - beta_m) * grad
        v_t = beta_v * curr_v + (1 - beta_v) * (grad ** 2)
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
    return predict(X, step_history[-1]), loss_history

def train(optimizer_fn, X, y, **kwargs):
    
    # run the training and return the predicted values
    return optimizer_fn(X, y, **kwargs)


def plot_showdown(preds_raw, preds_norm, normal_raw, normal_norm, y,
                  floor_raw, floor_norm, save_path="./figs/D5_optimizer_showdown.png"):
    """
    2x2 grid:
      left column  = predicted-vs-actual (each optimizer + normal-eq target)
      right column = loss vs iteration (semilogy) with the normal-eq loss floor
      top row = raw features,  bottom row = standardized features
    preds_* are dicts {name: [predictions, loss_history]}.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    colors = {"vanilla": "#ff7f0e", "momentum": "#1f77b4", "nesterov": "#2ca02c",
              "SGD": "#d62728", "Adam": "#9467bd"}
    y = np.asarray(y, dtype=float)
    # scatter is unreadable with 10k x 5 points -> plot a fixed random 300-point sample
    sub = np.random.default_rng(0).choice(len(y), size=min(300, len(y)), replace=False)

    # shared log y-limits so the two loss panels are directly comparable
    all_losses = [v for d in (preds_raw, preds_norm) for (_, lh) in d.values() for v in lh]
    lo = min(min(all_losses), floor_raw, floor_norm) * 0.7
    hi = max(all_losses) * 1.3

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    def fit_panel(ax, preds_dict, normal_preds, title):
        lims = [float(y.min()), float(y.max())]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="perfect (y = x)")
        ax.scatter(y[sub], np.asarray(normal_preds)[sub], s=24, c="black", marker="x",
                   linewidths=1.2, label="Normal Eq (target)", zorder=6)
        for name, (preds, _) in preds_dict.items():
            ax.scatter(y[sub], np.asarray(preds)[sub], s=12, alpha=0.5,
                       color=colors[name], label=name)
        ax.set_xlabel("actual Performance Index"); ax.set_ylabel("predicted")
        ax.set_title(title); ax.legend(fontsize=8, loc="upper left")

    def loss_panel(ax, preds_dict, floor, title):
        for name, (_, lh) in preds_dict.items():
            ax.semilogy(lh, color=colors[name], lw=1.6, label=name)
        ax.axhline(floor, ls="--", c="black", lw=1.2, label=f"Normal Eq loss ≈ {floor:.2f}")
        ax.set_ylim(lo, hi)
        ax.set_xlabel("iteration"); ax.set_ylabel("MSE (log)")
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    fit_panel(axes[0, 0], preds_raw,  normal_raw,  "Model fit — RAW features")
    loss_panel(axes[0, 1], preds_raw,  floor_raw,  "Convergence — RAW features")
    fit_panel(axes[1, 0], preds_norm, normal_norm, "Model fit — STANDARDIZED features")
    loss_panel(axes[1, 1], preds_norm, floor_norm, "Convergence — STANDARDIZED features")

    fig.suptitle("Optimizer Showdown — Student Performance regression", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(save_path, dpi=130)
    print(f"[SAVED] {save_path}")
    return fig


if __name__ == "__main__":
    
    print(f"Loading Student Performance Dataset...")
    path = "./data/Student_performance.csv"
    X, y = get_dataset(path)

    # convert format of data to numpy
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    # prepend X with a column of 1s for our bias term
    X_aug = np.hstack((np.ones((X.shape[0], 1)), X)) # shape => N x (d + 1)

    print(f"[STARTING] Unnormalized run")
    normal_theta_raw = normal_eq(X_aug, y_np)
    normal_preds_raw = predict(X_aug, normal_theta_raw)      # target model's predictions
    floor_raw = loss(X_aug, y_np, normal_theta_raw)          # best achievable MSE (~4.15)
    vanilla_GD_preds, vanilla_GD_loss = train(vanilla_GD, X=X_aug, y=y_np, alpha=0.0001)
    momentum_GD_preds, momentum_GD_loss = train(momentum_GD, X=X_aug, y=y_np, alpha=0.0001, beta=0.9)
    nesterov_GD_preds, nesterov_GD_loss = train(nesterov_GD, X=X_aug, y=y_np, alpha=0.0001, beta=0.9)
    SGD_preds, SGD_loss = train(SGD, X=X_aug, y=y_np, batch=32, alpha=0.0001)
    Adam_preds, Adam_loss = train(Adam, X=X_aug, y=y_np, batch=32, alpha=0.01)

    y_preds_unnormalized = {
        "vanilla": [vanilla_GD_preds, vanilla_GD_loss],
        "momentum": [momentum_GD_preds, momentum_GD_loss],
        "nesterov": [nesterov_GD_preds, nesterov_GD_loss],
        "SGD": [SGD_preds, SGD_loss],
        "Adam": [Adam_preds, Adam_loss]
    }

    print(f"[COMPLETE] Unnormalized run")

    # normalize our features then rerun:
    print(f"[STARTING] Normalized run")
    dataset_mean = X.mean()
    dataset_std = X.std()
    normalized_X = (X - dataset_mean) / dataset_std
    normalized_X_aug = np.hstack((np.ones((normalized_X.shape[0], 1)), normalized_X))

    normal_theta_norm = normal_eq(normalized_X_aug, y_np)
    normal_preds_norm = predict(normalized_X_aug, normal_theta_norm)
    floor_norm = loss(normalized_X_aug, y_np, normal_theta_norm)
    vanilla_GD_preds, vanilla_GD_loss = train(vanilla_GD, X=normalized_X_aug, y=y_np)
    momentum_GD_preds, momentum_GD_loss = train(momentum_GD, X=normalized_X_aug, y=y_np, beta=0.9)
    nesterov_GD_preds, nesterov_GD_loss = train(nesterov_GD, X=normalized_X_aug, y=y_np, beta=0.9)
    SGD_preds, SGD_loss = train(SGD, X=normalized_X_aug, y=y_np, batch=32)
    Adam_preds, Adam_loss = train(Adam, X=normalized_X_aug, y=y_np, batch=32, alpha=0.01)

    y_preds_normalized = {
        "vanilla": [vanilla_GD_preds, vanilla_GD_loss],
        "momentum": [momentum_GD_preds, momentum_GD_loss],
        "nesterov": [nesterov_GD_preds, nesterov_GD_loss],
        "SGD": [SGD_preds, SGD_loss],
        "Adam": [Adam_preds, Adam_loss]
    }
    print(f"[COMPLETE] Normalized run")

    plot_showdown(
        y_preds_unnormalized, y_preds_normalized,
        normal_preds_raw, normal_preds_norm,
        y_np, floor_raw, floor_norm,
    )






    



