import numpy as np

def det(matrix):
    # solve for the determinant of an NxN matrix

    # need to define a base case for determinants here
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    elif len(matrix) == 0:
        raise ValueError(f"Matrix is malformed! {matrix}")
    
    total = 0
    for j in range(len(matrix[0])):
        M = [[item for i, item in enumerate(row) if i != j] for row in matrix[1:]]
        #print(M)
        total += pow(-1, j) * matrix[0][j] * det(M)
    
    return total

def minor(matrix, drop_row, drop_col):
    m_row = [row for idx, row in enumerate(matrix) if idx != drop_row]
    minor = [[item for idx, item in enumerate(row) if idx != drop_col] for row in m_row]
    return minor

def cofactor(matrix, i, j):
    return pow(-1, (i+j)) * det(minor(matrix, i, j))

def transpose(matrix):
    temp = [[elem for elem in row] for row in matrix]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            temp[i][j] = matrix[j][i]
    return temp


def inv(matrix):

    # for this, we will go the adjugate route to calculate inverse since
    # we have the determinant function now, as well as separate funcs for
    # cofactor and transpose.

    # so we can just run the inv(matrix) = 1/det(matrix) * adj(matrix) ; adj(matrix) = cofactor^T ; cofactor[i][j] = (-1)^{i+j} det(matrix_ij)
    if det(matrix) == 0.0:
        raise ValueError(f"Invertible Matrix! {matrix}")
    det_frac = 1 / det(matrix)

    # put together the cofactor matrix
    num_of_cols = len(matrix[0])
    num_of_rows = len(matrix)
    cofactor_matrix = [[cofactor(matrix, i, j) for j in range(num_of_cols)] for i in range(num_of_rows)]
    
    adj_matrix = transpose(cofactor_matrix)
    inverse_matrix = [[item * det_frac for item in row] for row in adj_matrix]
    return inverse_matrix

def dot(A, b):
    num_of_rows = len(A)
    num_of_cols = len(A[0])

    dot_product = []
    for i in range(num_of_rows):
        curr_sum = 0
        for j in range(num_of_cols):
            curr_sum += A[i][j] * b[j]
        dot_product.append(curr_sum)
    return dot_product

def solve(A, b):
    # this can be solved with the functions we have currently and
    # we don't need to use gaussian elimination
    # Ax = b => A^{-1}Ax = A^{-1}b => x = A^{-1}b

    # just need dot product func
    return dot(inv(A), b)
    



matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]] # should be 0.0
matrix_2 = [[2.0,1.0,1.0],[-3.0,-1.0,2.0],[-2.0,1.0,2.0]] # should be -11

determinant = det(matrix)

print(f"My determinant: {determinant}")
print(f"Numpy determinant: {np.linalg.det(matrix)}")

determinant = det(matrix_2)

print(f"My determinant: {determinant}")
print(f"Numpy determinant: {np.linalg.det(matrix_2)}")

# testing inverse

print(f"My inverse: {inv(matrix_2)}")
print(f"Numpy inverse: {np.linalg.inv(matrix_2)}")

A1 = [[1.,0.,0.,0.],[0.,2.,0.,0.],[0.,0.,3.,0.],[0.,0.,0.,4.]]
A2 = [[2.,1.,0.,1.],[1.,3.,1.,0.],[0.,1.,2.,1.],[1.,0.,1.,3.]]
A3 = [[4.,-2.,1.,0.],[-1.,3.,0.,2.],[2.,0.,5.,-3.],[0.,1.,-1.,4.]]
A4 = [[1.,2.,3.,4.],[2.,1.,0.,1.],[3.,3.,3.,5.],[0.,0.,1.,1.]]

print(f"My inverse: {inv(A1)}")
print(f"Numpy inverse: {np.linalg.inv(A1)}")
print(f"My inverse: {inv(A2)}")
print(f"Numpy inverse: {np.linalg.inv(A2)}")
print(f"My inverse: {inv(A3)}")
print(f"Numpy inverse: {np.linalg.inv(A3)}")
print(f"My inverse: {inv(A4)}")
print(f"Numpy inverse: {np.linalg.inv(A4)}")