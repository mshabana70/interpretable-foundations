import numpy as np


def eigen_decomp(A, eig_vals, eig_vecs):

    P = np.asarray(eig_vecs)
    D = np.zeros(A.shape)
    
    for i, eigen_val in enumerate(eig_vals):
        D[i][i] = eigen_val
    
    eigen_decomp_matrix = P @ D @ np.linalg.inv(P)

    # verify that this eigen decomp is correct
    if np.allclose(A, eigen_decomp_matrix):
        print(f"Eigendecomposition is Verified!")
    else:
        print(f"Eigendecomposition Failed!")
    return eigen_decomp_matrix


def test():
    sym_2x2_A = [[2., 1.], [1., 2.]]
    sym_3x3_A = [[2., -1., 0.], [-1., 2., -1.], [0., -1., 2.]]
    non_sym_2x2_A = [[2., 1.], [0., 3.]]
    non_sym_2x2_A2 = [[4., 1.], [2., 3.]]
    defect = [[2., 1.], [0., 2.]]

    A1 = np.array(sym_2x2_A)
    eig_vals, eig_vecs = np.linalg.eig(A1)
    print(f"Matrix A = {A1}")
    eig_mat = eigen_decomp(A1, eig_vals, eig_vecs)
    print(f"Eigendecomp: {eig_mat}\n")
    
    A3 = np.array(non_sym_2x2_A)
    eig_vals, eig_vecs = np.linalg.eig(A3)
    print(f"Matrix A = {A3}")
    eig_mat = eigen_decomp(A3, eig_vals, eig_vecs)
    print(f"Eigendecomp: {eig_mat}\n")
    
    A4 = np.array(non_sym_2x2_A2)
    eig_vals, eig_vecs = np.linalg.eig(A4)
    print(f"Matrix A = {A4}")
    eig_mat = eigen_decomp(A4, eig_vals, eig_vecs)
    print(f"Eigendecomp: {eig_mat}\n")
    
    A5 = np.array(defect)
    eig_vals, eig_vecs = np.linalg.eig(A5)
    print(f"Matrix A = {A5}")
    eig_mat = eigen_decomp(A5, eig_vals, eig_vecs)
    print(f"Eigendecomp: {eig_mat}\n")

    

if __name__ == "__main__":
    test()
    
    