
class Matrix():

    def __init__(self, rows, cols, data: list[list] = None):
        self.rows = rows
        self.cols = cols
        if data:
            self.data = [row[:] for row in data] 
        else:
            self.data = [[0 for _ in range(cols)] for _ in range(rows)]
    
    def __mul__(self, vector: list):

        if self.cols == len(vector):
            mul_result = []
            for i in range(self.rows):
                sum = 0
                for j in range(self.cols):
                    sum += self.data[i][j] * vector[j]
                mul_result.append(sum)

            return mul_result
        else:
            raise IndexError(f"ERROR: Invalid dimensions for multiplication: Matrix dim A ({self.rows} x {self.cols}), Vector dim ({len(vector)} X 1)")

    def __repr__(self):
        return f"Matrix({self.data})"
    
    def shape(self):
        return f"({self.rows}, {self.cols})"

    def transpose(self):
        new_matrix = []

        for idx in range(self.cols):
            new_matrix.append([])
            for row in self.data:
                new_matrix[idx].append(row[idx])
            
        orig_rows = self.rows
        orig_cols = self.cols
        return Matrix(orig_cols, orig_rows, new_matrix)
    
    @staticmethod
    def identity(n):
        identity_matrix = []

        for row_idx in range(n):
            identity_matrix.append([])
            for col_idx in range(n):
                if row_idx == col_idx:
                    identity_matrix[row_idx].append(1)
                else:
                    identity_matrix[row_idx].append(0)
        
        return Matrix(n, n, identity_matrix)

def test():

    # testing the matrix class definition first
    mat_a = Matrix(3, 4, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    print("\n", 5*"-", "Matrix Class Definition", 5*"-")
    print(f"Definition of Matrix A:")
    print(f"Matrix A rows: {mat_a.rows}")
    print(f"Matrix A columns: {mat_a.cols}")
    print(f"Matrix A data: {mat_a.data}")
    print(f"Matrix A element at (1, 2): {mat_a.data[1][2]}")

    print("\n", 5*"-", "Matrix-Vector Multiplication", 5*"-")
    vec_c = [1, 2]
    mat_b = Matrix(2, 2, [[1, 2], [3, 4]])
    print(f"Multiplication of Matrix A and Vector B:")
    print(f"Matrix B: {mat_b.data}")
    print(f"Vector C: {vec_c}")
    print(f"Product of B x C: {mat_b * vec_c}")

    print("\n", 5*"-", "Matrix Transpose", 5*"-")
    mat_c = Matrix(3, 4, [[2, 4, 6, 8], [10, 12, 14, 16], [18, 20, 22, 24]])
    mat_c_t = mat_c.transpose()
    print(f"Matrix transposition of Matrix C of shape {mat_c.shape()}: {mat_c}")
    print(f"Transposed Matrix C of shape {mat_c_t.shape()}: {mat_c_t}")

    print("\n", 5*"-", "Identity Matrix", 5*"-")
    identity_mat = Matrix.identity(4)
    print(f"Defined identity matrix of size ({identity_mat.rows}, {identity_mat.cols}): {identity_mat}")


if __name__ == "__main__":
    print("---------- Running Matrix tests ----------")
    test()
    print("\n---------- Completed Matrix tests ----------")
