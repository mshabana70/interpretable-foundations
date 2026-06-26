import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# we can calculate PCA via eigendecomposition.
# we should first center the data by taking mean of features and subtracting it 
# from the original data matrix.

def plot_pca(T, data, filename="iris_pca_2D_3D.png"):
    fig = plt.figure(figsize=(12, 8))

    # each column of the matrix T is a principal component
    ax1 = fig.add_subplot(1, 2, 1)
    pc1 = T[:, 0]
    pc2 = T[:, 1]
    pc3 = T[:, 2]

    target_names = ['setosa', 'versicolor', 'virginica']
    for k in range(len(target_names)):
        mask = data.target == k
        ax1.scatter(T[mask, 0], T[mask, 1], label=target_names[k], marker='o')

    # ax1.scatter(pc1, pc2, c=data.target, cmap='viridis', marker='o')
    ax1.legend()
    ax1.set_title("2D PCA plot from Iris data")
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    ax1.grid(True)
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    for k in range(len(target_names)):
        mask = data.target == k
        ax2.scatter(T[mask, 0], T[mask, 1], T[mask, 2], label=target_names[k], marker='o')
    
    ax2.legend()
    ax2.set_title("3D PCA plot from Iris data")
    ax2.set_xlabel('PC 1')
    ax2.set_ylabel('PC 2')
    ax2.set_zlabel('PC 3')
    plt.savefig(filename)

def sci_pca(data):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(data.data)
    plot_pca(X_pca, data, filename="iris_pca_sklearn.png")


def myPCA(data):
    # first we compute the mean-centered matrix
    # we need to take the avg of all samples across a feature, so we should parse columns
    X = data.data
    col_means = X.mean(axis=0)

    # creating a (1, 4) matrix of the col_means
    col_means = np.array([col_means])

    # broadcast avg vec to a matrix so we can subtract it from X
    X_bar = np.repeat(col_means, len(X), axis=0)

    B = X - X_bar

    # compute covariance matrix
    C = B.T @ B

    # now we get the eigendecomp of C
    eigenvals, V = np.linalg.eig(C)

    # Should sort these the same way we did for SVD
    descending_idxs = np.argsort(eigenvals)[::-1]
    eigenvecs = V.T # need to transpose because vectors_v is the eigenvectors as columns
    eigenvals = np.array([eigenvals[idx] for idx in descending_idxs])
    V_t = np.array([eigenvecs[idx] for idx in descending_idxs])
    V = V_t.T
    
    # need to create dialgonalized matrix D
    D = np.diag(eigenvals)

    # need to fix this check because it doesn't actually check our PCA computation, just verifies eigendecomp
    # # let's do a quick verification of our C, V, and D
    # # this needs to be true: CV = VD
    # cv_prod = C @ V
    # vd_prod = V @ D

    # let's verify instead with the SVD of B such that T = U \sigma_matrix
    T = B @ V
    B_U, B_S, B_V = np.linalg.svd(B)
    resized_S = np.zeros(X.shape)
    for i in range(len(B_S)):
        resized_S[i][i] = B_S[i]
    
    verify_T = B_U @ resized_S

    if np.allclose(np.abs(T), np.abs(verify_T)):
        print(f"SVD T Verified values!!!")
        plot_pca(T, data)
        
        # verify with sklearn pca
        sci_pca(data)
    else:
        print(f"NOT verified!!!")


def test():
    # grab the iris dataset
    data = load_iris()
    print(data.feature_names)
    myPCA(data)


if __name__ == "__main__":
    test()