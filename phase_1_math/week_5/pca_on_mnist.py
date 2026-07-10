import math
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


def import_mnist():
    """
    Grab MNIST dataset for our PCA tests.
    """
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    return mnist

def custom_pca(X, num_components=2):

    # we need to compute the mean-centered data matrix B
    B = X - X.mean(axis=0) 
    
    # # quick sanity check
    # print("Original Matrix: ", X[:5])
    # print("Mean-centered Matrix: ", B[:5])

    # get covariance matrix
    C = B.T @ B

    # get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(C) # switching eigh since it handle real value returns

    # sort eigenvals and eigenvecs in descending order
    desc_idx = np.argsort(eigenvalues)[::-1] # need argsort not sort here, also eigh returns values in ascending already
    eigenvecs = eigenvectors.T # eigenvecs are returned as columns here
    sorted_eigenvals = np.array([eigenvalues[idx] for idx in desc_idx])
    sorted_eigenvecs = np.array([eigenvecs[idx] for idx in desc_idx])

    print(f"First few eigenvalues: {sorted_eigenvals[:10]}")
    V = sorted_eigenvecs.T

    # now we get the principal components
    T = B @ V
    return T[:, :num_components]

def plot_pca(PC, y):
    plt.scatter(PC[:, 0], PC[:, 1], s = 2, c = y.astype(int), cmap="tab10", alpha=0.4) # need to map int target vals to colors
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("Two Principal Components of MNIST")
    plt.colorbar() # so we can tell which label is which color
    plt.savefig("mnist_pca.png")

def test():
    # importing dataset here so we only need to do it once
    mnist = import_mnist()
    X, y = mnist["data"], mnist["target"]

    principal_components = custom_pca(X)

    # TODO: now we can plot the PCs
    plot_pca(principal_components, y)



if __name__ == "__main__":
    test()