
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


def test():

    # testing the matrix class definition first
    mat_a = Matrix(3, 4, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    print(f"Definition of Matrix A:")
    print(f"Matrix A rows: {mat_a.rows}")
    print(f"Matrix A columns: {mat_a.cols}")
    print(f"Matrix A data: {mat_a.data}")
    print(f"Matrix A element at (1, 2): {mat_a.data[1][2]}")

    vec_c = [1, 2]
    mat_b = Matrix(2, 2, [[1, 2], [3, 4]])
    print(f"Multiplication of Matrix A and Vector B:")
    print(f"Matrix B: {mat_b.data}")
    print(f"Vector C: {vec_c}")
    print(f"Product of B x C: {mat_b * vec_c}")

if __name__ == "__main__":
    print("---------- Running Matrix tests ----------")
    test()
    print("---------- Completed Matrix tests ----------")
