import numpy as np

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

A = np.array([[2, 1, 1], [-3, -1, 2], [-2, 1, 2]]).astype(float)
b = np.array([8, -11, -3]).astype(float)

combined_matrix = np.column_stack((A, b.T))
print(combined_matrix)

# Quick func test
swapped_matrix = row_swap(combined_matrix, 0, 1)
mul_matrix = row_mul(combined_matrix, 0, 10)
added_matrix = row_add(combined_matrix, 0, 1, 2)

# print(swapped_matrix)
# print(mul_matrix)
# print(added_matrix)

for k in range(len(combined_matrix)):
    # k will be our pivot value in an NxN matrix.
    # Limiting k to the range of N reduces the need to avoid the "b" column
    print(f"Current k: {k}")

    # need to implement swap condition:
    # swap if current pivot = 0.0 with any row below current pivot
    if combined_matrix[k][k] == 0.0:
        search_idx = k
        while combined_matrix[k][search_idx] != 0.0:
            search_idx += 1
            if search_idx == len(combined_matrix):
                raise ValueError("No solution for Gaussian Elimination")
        combined_matrix = row_swap(combined_matrix, k, search_idx)

    for i in range(k+1, len(combined_matrix)):
        print(f"Current i: {i}")
        constant = (combined_matrix[i][k]) / combined_matrix[k][k]
        combined_matrix = row_add(combined_matrix, i, k, constant)
        print(combined_matrix)
    print(combined_matrix)


        

