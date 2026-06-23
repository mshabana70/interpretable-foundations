# this code is to compute the eigenvectors and eigenvalues of a matrix
# using the power iteration method

# the power iteration method states that we randomly initialize vector v 
# and keep increasing the power of matrix A, we will land on the dominant 
# eigenvector (the one with the largest eigenvalue in absolute value).

import math
import numpy as np

# need a rng for this
rng = np.random.default_rng(seed=42)

def l2_norm(vector):
    return np.linalg.norm(vector)

def rayleigh_quot(A, vector):
    return ((vector.T @ A @ vector) / (vector.T @ vector)) # denom should cancel out if it's a unit vector

def pow_iter(A, max_iter=100):
    rand_v = rng.standard_normal(len(A))
    v_new = rand_v
    v_old = v_new
    for i in range(max_iter):
        resulting_vec = A @ v_new
        #print(resulting_vec) # this resulting vec is underflowing like crazy, need to normalize

        # we will just extract the direction of the resulting_vec
        # so we normalize the vec to a unit vector
        v_new = resulting_vec / l2_norm(resulting_vec)
        # diff_mag = l2_norm(v_old - v_new) # this works if the vector doesn't flip sign every iter

        eigen_new = rayleigh_quot(A, v_new)
        # more robust check to see if v and \lambda satify the expression Av = \lambda v
        express_check = (A @ v_new) - (eigen_new * v_new)

        # if diff_mag < 1e-18:
        if l2_norm(express_check) < 1e-9:
            print(f"Convergence achieved on step {i}")
            break
        else:
            # we need to set the next iters vector v to the normalize vector
            v_old = v_new
            if i == (max_iter - 1):
                print(f"Failed to converge by step {i+1}")
                break


    # now we need to extract the eigenvalues; we can use rayleigh quotient for this.
    eigenvalue = rayleigh_quot(A, v_old)
    return (v_old, eigenvalue)


def test():

    A1 = np.array([[3, 1], [1, 3]]).astype(float)
    A2 = np.array([[4, 1], [2, 3]]).astype(float)
    A3 = np.array([[6, 2, 1], [2, 3, 1], [1, 1, 1]]).astype(float)
    A4 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, -2]]).astype(float)

    dom_eigenvec, eigenvalue = pow_iter(A1)
    print(f"Matrix A = {A1}\nDominant Eigenvector: {dom_eigenvec}\nDominant Eigenvalue: {eigenvalue}\n")
    dom_eigenvec, eigenvalue = pow_iter(A2)
    print(f"Matrix A = {A2}\nDominant Eigenvector: {dom_eigenvec}\nDominant Eigenvalue: {eigenvalue}\n")
    dom_eigenvec, eigenvalue = pow_iter(A3)
    print(f"Matrix A = {A3}\nDominant Eigenvector: {dom_eigenvec}\nDominant Eigenvalue: {eigenvalue}\n")
    dom_eigenvec, eigenvalue = pow_iter(A4)
    print(f"Matrix A = {A4}\nDominant Eigenvector: {dom_eigenvec}\nDominant Eigenvalue: {eigenvalue}\n")

    # edge cases
    E1 = np.array([[0, -1], [1, 0]])
    E2 = np.array([[-5, 0], [0, 2]])

    print(f"EDGE CASE TEST:")
    dom_eigenvec, eigenvalue = pow_iter(E1)
    print(f"Matrix A = {E1}\nDominant Eigenvector: {dom_eigenvec}\nDominant Eigenvalue: {eigenvalue}\n")
    dom_eigenvec, eigenvalue = pow_iter(E2)
    print(f"Matrix A = {E2}\nDominant Eigenvector: {dom_eigenvec}\nDominant Eigenvalue: {eigenvalue}\n")




if __name__ == "__main__":
    test()






