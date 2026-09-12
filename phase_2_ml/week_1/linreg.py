import numpy as np


# TODO: going to first start with the three core funcs
# - predict(X, w)
# - loss(X, y, w)
# - gradient(X, y, w)
# The shapes of our inputs are:
# X (n x d matrix)
# w (d x 1 vector)
# y (n x 1 vector)

# adding an rng for weight initialization
rng = np.random.default_rng(seed=42)

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

def apply_standardizer(X, stats: dict):

    X_scaled = np.zeros(X.shape)
    # return X_scaled
    for feature in stats["features"]:
        idx = feature["idx"]
        mean = feature["mean"]
        std = feature["std"]

        # Z-score normalization => z_j = (x_ij - mu_j) / sigma_j for all i in X_j
        X_temp = X.copy()
        if std != 0.0:
            X_scaled[:, idx] = (X_temp[:, idx] - mean) / std
        else:
            X_scaled[:, idx] = 0.0

    # add an intercept column to match the design matrix returned by np.linalg.lstsq
    # useful when verifying
    intercept_col = np.zeros((X.shape[0], 1))
    X_scaled = np.hstack((intercept_col, X_scaled)) 
    return X_scaled

def fit_standardizer(X):
    """
    Func for preprocessing and standardizing our dataset before being fed to the optimizer.
    
    We'll do standardization using the Z-score normalization in the apply_standarizer() func.
    """

    stats_dict = {"features": []}
    # first let's grab the means and standard deviations from each feature of X
    means = np.mean(X, axis=0) # column-wise means
    stand_devs = np.std(X, axis=0)
    for j in range(X.shape[1]):
        print(f"Feature {j}: mean = {means[j]}; std = {stand_devs[j]}")
        stats_dict["features"].append({
            "idx": j,
            "mean": means[j],
            "std": stand_devs[j],
            "max": max(X[:, j]),
            "min": min(X[:, j]),
        })

    X_scaled = apply_standardizer(X, stats_dict)

    return X_scaled, stats_dict

def fit_gradient_descent(X, y, w=None, lr=1e-3, iters=50):
    # now we FINALLY get to the training loop lol (T_T).
    # for this we need to initialize our weights if it's not provided,
    # which is realistic in pretty much all training loops.
    
    if w is None:
        w = rng.random((X.shape[1], 1)) # random init between 0 and 1

    # TODO: do some feature scaling because our training is not converging!!!

    # build our arrays that capture values we care about in the loop
    w_t = w.copy()
    record = {
        "loss": [], 
        "weights": [w_t] # this will be a list of our updated weight vecs
    }
    # now we define the loop
    for t in range(iters):
        curr_grad = gradient(X, y, w_t) # leaving numerical gradient as a verifier

        w_t -= lr * curr_grad
        curr_loss = loss(X, y, w_t) # this isn't necessary, it's really just for stdout. Also, I am measuring loss AFTER the weight update
        record["weights"].append(w_t.copy()) # store COPIES
        record["loss"].append(curr_loss)
        print(f"Iteration {t + 1}: loss = ({curr_loss})") # this should be decreasing every iter

    return (record["weights"][-1], record["loss"]) # return the last weight update, as well as the list of losses during training 