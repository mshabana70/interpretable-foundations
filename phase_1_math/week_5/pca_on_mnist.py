import math
import numpy as np
from sklearn.datasets import fetch_openml


def import_mnist():
    """
    Grab MNIST dataset for our PCA tests.
    """
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    return mnist

def custom_pca(X, num_components=2):

    # we need to compute the mean-centered data matrix B
    col_means = X.mean(axis=0)
    col_means = np.array([col_means]) # (1, 784)
    
    # need to broadcast the row averages so it matches shape of X
    X_bar = np.repeat(col_means, len(X), axis=0)

    B = X - X_bar 
    
    # # quick sanity check
    # print("Original Matrix: ", X[:5])
    # print("Mean-centered Matrix: ", B[:5])

    # get covariance matrix
    C = B.T @ B

    # get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(C)

    # sort eigenvals and eigenvecs in descending order
    desc_idx = np.sort(eigenvalues)[::-1]
    eigenvecs = eigenvectors.T # eigenvecs are returned as columns here
    sorted_eigenvals = np.array([eigenvalues[idx] for idx in desc_idx])
    sorted_eigenvecs = np.array([eigenvecs[idx] for idx in desc_idx])
    V = sorted_eigenvecs.T

    D = np.diag(sorted_eigenvals)

    # now we get the principal components
    T = B @ V
    return T[:, :num_components]


def test():
    # importing dataset here so we only need to do it once
    mnist = import_mnist()
    X, y = mnist["data"], mnist["target"]

    principal_components = custom_pca(X)

    # TODO: now we can plot the PCs



if __name__ == "__main__":
    test()