import numpy as np

# for this linear system solver, we can keep the structure of our gaussian elimination code
# but let's do things a bit differently for the "type of solution" detection
# we'll use matrix rank for this.

# then, we can solve the linear system using Cramer's rule:
# x_i = det(coefficient matrix with column i replaced with b) / det(coefficient matrix) where b = vector of linear system results.


def row_swap(A, i, j):
    temp = A.copy()
    temp[i] = A[j] 
    temp[j] = A[i]
    return temp

def row_mul(A, i, c):
    temp = A.copy()
    temp[i] = A[i] * c
    return temp

def row_add(A, i, j, c):
    # returns a matrix with with A[i] + c*A[j]
    temp = A.copy()
    temp[i] = A[i] - (c * A[j])
    return temp

def solve(A, b):
    vector_x = np.array([])
    for i in range(len(A)):
        A_replace = A.copy()
        A_replace[:, i] = b
        variable_solution = np.linalg.det(A_replace) / np.linalg.det(A)
        vector_x = np.append(vector_x, variable_solution)
    
    print("Ax = b")
    print(f"A = {A}\nx = {vector_x}\nb = {b}")



def test():
    A = np.array([[2, 1, 1], [-3, -1, 2], [-2, 1, 2]]).astype(float)
    b = np.array([8, -11, -3]).astype(float)

    print("="*11, "Test 1", "="*11)
    solve(A, b)
    print("="*30)
    
    # A = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]]).astype(float)
    # b = np.array([6, -4, 27]).astype(float)

    # print("="*11, "Test 2", "="*11)
    # solve(A, b)
    # print("="*30)
    
    # A = np.array([[1, 1, 1], [2, 2, 2], [1, 2, 3]]).astype(float)
    # b = np.array([1, 3, 4]).astype(float)

    # print("="*11, "Test 3", "="*11)
    # solve(A, b)
    # print("="*30)
    
    # A = np.array([[1, 1, 1], [1, 2, 3], [2, 3, 4]]).astype(float)
    # b = np.array([6, 14, 20]).astype(float)

    # print("="*11, "Test 4", "="*11)
    # solve(A, b)
    # print("="*30)
            
if __name__ == "__main__":
    test()

