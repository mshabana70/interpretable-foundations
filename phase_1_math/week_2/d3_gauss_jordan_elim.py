import numpy as np
import time

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

def solve(A, I):
    if A.shape != I.shape:
        raise ValueError(f"A and I are not of equal dims!: A = {A.shape}, I = {I.shape}")
    elif np.linalg.det(A) == 0.0:
        raise ValueError(f"A is not invertible!: det(A) = {np.linalg.det(A)}")
    combined_matrix = np.column_stack((A, I))

    for k in range(len(combined_matrix)):
        # k will be our pivot value in an NxN matrix.
        # Limiting k to the range of N reduces the need to avoid the "b" column

        # need to implement swap condition:
        # swap if current pivot = 0.0 with any row below current pivot
        if combined_matrix[k][k] == 0.0 and k < len(combined_matrix) - 1:
            search_idx = k + 1
            while combined_matrix[search_idx][k] == 0.0:
                search_idx += 1
            combined_matrix = row_swap(combined_matrix, k, search_idx)

        # normalize the pivot value so it equals 1
        k_norm = 1 / combined_matrix[k][k]
        combined_matrix = row_mul(combined_matrix, k, k_norm)
        nonzero_idxs = [i for i in range(len(combined_matrix)) if i != k]

        for i in nonzero_idxs:
            constant = (combined_matrix[i][k]) / combined_matrix[k][k]
            combined_matrix = row_add(combined_matrix, i, k, constant)
    
    num_of_rows = len(combined_matrix)
    num_of_columns = num_of_rows # we are assuming square matrix

    # split the combined array into identity and inverted matrix
    new_identity = np.hsplit(combined_matrix, 2)[0]
    inv_matrix = np.hsplit(combined_matrix, 2)[1]
    print(f"Identity Matrix: {new_identity}")
    print(f"Inverse Matrix: {inv_matrix}")

def test():
    A = np.array([[2, 1, 1], [-3, -1, 2], [-2, 1, 2]]).astype(float)
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(float)

    print("="*11, "Test 1", "="*11)
    start_time = time.perf_counter()
    solve(A, I)
    end_time = time.perf_counter()
    my_inverse_time = end_time - start_time

    # verify with numpy inv func
    start_time = time.perf_counter()
    print(f"Numpy Inverse Matrix: {np.linalg.inv(A)}")
    end_time = time.perf_counter()
    np_inverse_time = end_time - start_time
    print(f"My execution time: {my_inverse_time}")
    print(f"NumPy execution time: {np_inverse_time}")
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
    print("="*30)
            
if __name__ == "__main__":
    test()

