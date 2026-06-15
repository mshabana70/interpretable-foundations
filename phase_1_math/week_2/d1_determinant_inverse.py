import numpy as np

def deter_2_by_2(mat: np.array):
    # for mat = [[a, b], [c, d]]; det mat = a*d - b*c

    a = mat[0][0]
    b = mat[0][1]
    c = mat[1][0]
    d = mat[1][1]

    return (a*d) - (b*c)

def deter_3_by_3(mat: np.array):
    # for mat = [[a, b, c], [d, e, f], [g, h, i]];
    # det(mat) = a * det([[e, f], [h, i]]) - (b* det([[d, f], [g, i]])) + (c * det([[d, e], [g, h]]))
    a = mat[0][0]
    b = mat[0][1]
    c = mat[0][2]

    d = mat[1][0]
    e = mat[1][1]
    f = mat[1][2]

    g = mat[2][0]
    h = mat[2][1]
    i = mat[2][2]

    det_1 = deter_2_by_2(np.array([[e, f], [h, i]]))
    det_2 = deter_2_by_2(np.array([[d, f], [g, i]]))
    det_3 = deter_2_by_2(np.array([[d, e], [g, h]]))

    det = (a * det_1) - (b * det_2) + (c * det_3)

    return det

def inv_2_by_2(mat: np.array):
    # a matrix is invertible if and only if it's determinant is non-zero
    # inv(mat) = (1 / det(mat)) * [[d, -b], [-c, a]]
    if (deter_2_by_2(mat) != 0):
        a = mat[0][0]
        b = mat[0][1]
        c = mat[1][0]
        d = mat[1][1]
        inv_mat = (1 / deter_2_by_2(mat)) * np.array([[d, -b], [-c, a]])
        return inv_mat
    else:
        print(f"[ERROR] Matrix is not invertible: {mat}")
        return None


def test_determinant(mat: np.array):
    if mat.shape[0] == 2:
        det = deter_2_by_2(mat)
    else:
        det = deter_3_by_3(mat)
    print(f"Matrix: {mat}\nDeterminant: {det}")
    print(f"Verifying with np.linalg.det: {np.linalg.det(mat)}\n")

def test_inverse(mat: np.array):
    inverse = inv_2_by_2(mat)
    print(f"Matrix: {mat}\nInverse: {inverse}")
    print(f"Verifying with np.linalg.inv: {np.linalg.inv(mat)}\n")

if __name__ == "__main__":

    matrix_a = np.array([[1, 2], [3, 4]])
    matrix_b = np.array([[8, 11], [4, 19]])
    matrix_c = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    matrix_d = np.array([[3, 9, 2], [11, 0, 13], [14, 8, 7]])
    matrix_e = np.array([[1, 0], [0, 0]])

    # test 2x2
    test_determinant(matrix_a)
    test_determinant(matrix_b)

    # test 3x3
    test_determinant(matrix_c)
    test_determinant(matrix_d)

    # test 2x2 inverse
    test_inverse(matrix_a)
    test_inverse(matrix_b)
    test_inverse(matrix_e)