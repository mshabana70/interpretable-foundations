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

def solve(A, b):
    if len(A) != len(b):
        raise ValueError(f"A and b are not of equal length!: A = {A.shape}, b = {b.shape}")
    combined_matrix = np.column_stack((A, b.T))

    for k in range(len(combined_matrix)):
        # k will be our pivot value in an NxN matrix.
        # Limiting k to the range of N reduces the need to avoid the "b" column

        # need to implement swap condition:
        # swap if current pivot = 0.0 with any row below current pivot
        if combined_matrix[k][k] == 0.0 and k < len(combined_matrix) - 1:
            print("need to swap")
            search_idx = k + 1
            while combined_matrix[search_idx][k] == 0.0:
                search_idx += 1
                
            combined_matrix = row_swap(combined_matrix, k, search_idx)

        for i in range(k+1, len(combined_matrix)):
            constant = (combined_matrix[i][k]) / combined_matrix[k][k]
            combined_matrix = row_add(combined_matrix, i, k, constant)
    
    # we still need to solve for vector x...
    print(combined_matrix)
    num_of_rows = len(combined_matrix)
    num_of_columns = num_of_rows # we are assuming square matrix

    z_c = combined_matrix[num_of_rows - 1][-2] # solution for z
    inf_sol_check = combined_matrix[-1][-1] # check value of last element of b
    if z_c == 0.0 and inf_sol_check != 0.0:
        print("No Solution for this system of equations!")
    elif z_c == 0 and inf_sol_check == 0.0:
        print("Infinite solutions for this system of equations!")
    else:
        # TODO: compute vector x 
        # this is a hardcoded solution but its fine for now
        z = inf_sol_check / z_c
        y_c = combined_matrix[-2][-3]
        y = (combined_matrix[-2][-1] - (z * combined_matrix[-2][-2])) / y_c
        x_c = combined_matrix[0][0]
        x = (combined_matrix[0][-1] - (y * combined_matrix[0][1]) - (z * combined_matrix[0][2])) / x_c

        vector_x = np.array([x, y, z])
        print(f"Unique Solution: x = {vector_x}")

def test():
    A = np.array([[2, 1, 1], [-3, -1, 2], [-2, 1, 2]]).astype(float)
    b = np.array([8, -11, -3]).astype(float)

    print("="*11, "Test 1", "="*11)
    solve(A, b)
    print("="*30)
    
    A = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]]).astype(float)
    b = np.array([6, -4, 27]).astype(float)

    print("="*11, "Test 2", "="*11)
    solve(A, b)
    print("="*30)
    
    A = np.array([[1, 1, 1], [2, 2, 2], [1, 2, 3]]).astype(float)
    b = np.array([1, 3, 4]).astype(float)

    print("="*11, "Test 3", "="*11)
    solve(A, b)
    print("="*30)
    
    A = np.array([[1, 1, 1], [1, 2, 3], [2, 3, 4]]).astype(float)
    b = np.array([6, 14, 20]).astype(float)

    print("="*11, "Test 4", "="*11)
    solve(A, b)
    print("="*30)
            
if __name__ == "__main__":
    test()

