import numpy as np
import math


def svd(A):

    # let's first get the eigenvalues and eigenvectors of A^T @ A

    # compute the right singular vectors of A^T A
    right_sing_mat = A.T @ A
    sigmas, vectors_v = np.linalg.eig(right_sing_mat)

    # reorganize the sigmas and eigenvectors
    descending_idxs = np.argsort(sigmas)[::-1]
    eigenvecs = vectors_v.T # need to transpose because vectors_v is the eigenvectors as columns
    sigmas = np.array([sigmas[idx] for idx in descending_idxs])
    vectors_v = np.array([eigenvecs[idx] for idx in descending_idxs])

    # vectors_v is really the matrix P from eigendecomp: PDP^{T}
    V_t = vectors_v
    
    # We can create the sigma matrix, which should be the same shape as A
    sig_matrix = np.zeros(A.shape, dtype=float)
    for i in range(min(sig_matrix.shape)):
        sig_matrix[i][i] = math.sqrt(sigmas[i])
    
    # now we can put together the matrix U using the equation u_i = (1/sig_i)Av_i
    m = min(sig_matrix.shape)
    num_of_rows = sig_matrix.shape[0]
    num_of_cols = sig_matrix.shape[1]
    U = np.zeros((num_of_rows, num_of_rows))
    for i in range(m):
        if sig_matrix[i][i] == 0.0:
            raise ValueError(f"Matrix is rank-deficient: {sig_matrix}")
        u_i = (1 / sig_matrix[i][i]) * A @ V_t[i]
        U[:, i] = u_i

    svd_A = U @ sig_matrix @ V_t
    return svd_A

def numpy_svd_test(A):
    U, S, Vh = np.linalg.svd(A, full_matrices=True, compute_uv=True)
    S_transformed = np.zeros(A.shape)
    for i in range(min(A.shape)):
        S_transformed[i][i] = S[i]
    # print(U, S_transformed, Vh)
    return U @ S_transformed @ Vh

def test():

    A1 = np.array([[3, 2, 2], [2, 3, -2]])
    A2 = np.array([[3, 2], [2, 3], [2, -2]])
    A3 = np.array([[2, 1], [1, 2]])

    svd_A1 = svd(A1)
    numpy_A1 = numpy_svd_test(A1)
    print(f"Original A: {A1}\nSVD A: {svd_A1}\nNumpy SVD A: {numpy_A1}\n")
    
    svd_A2 = svd(A2)
    numpy_A2 = numpy_svd_test(A2)
    print(f"Original A: {A2}\nSVD A: {svd_A2}\nNumpy SVD A: {numpy_A2}\n")
    
    svd_A3 = svd(A3)
    numpy_A3 = numpy_svd_test(A3)
    print(f"Original A: {A3}\nSVD A: {svd_A3}\nNumpy SVD A: {numpy_A3}\n")

if __name__ == "__main__":
    test()