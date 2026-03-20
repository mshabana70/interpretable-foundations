import math

class Vector():

    def __init__(self, *elements):
        self.array = [elem for elem in elements]
    
    def __repr__(self):
        return f"Vector({str(self.array)})"
    
    def __add__(self, other):
        if isinstance(other, Vector):
            new_vec = Vector()
            if len(self.array) == len(other.array):
                new_arr = [self.array[i] + other.array[i] for i in range(len(self.array))]
                new_vec.array = new_arr
                return new_vec
            else:
                return f"Vectors are not of same length: v1 is {len(self.array)} and v2 is {len(other.array)}"
        else:
            return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, Vector):
            new_vec = Vector()
            if len(self.array) == len(other.array):
                new_arr = [self.array[i] - other.array[i] for i in range(len(self.array))]
                new_vec.array = new_arr
                return new_vec
            else:
                return f"Vectors are not of same length: v1 is {len(self.array)} and v2 is {len(other.array)}"
        else:
            return NotImplemented
    
    def __mul__(self, scalar):
        new_vec = Vector()
        new_vec.array = [self.array[i] * scalar for i in range(len(self.array))]
        return new_vec
    
    def shape(self):
        return f"(1, {len(self.array)})"
    
    def dot(self, other):
        if isinstance(other, Vector):
            new_vec = Vector()
            if len(self.array) == len(other.array):
                new_arr = [self.array[i] * other.array[i] for i in range(len(self.array))]
                new_vec.array = new_arr
                return new_vec
            else:
                return f"Vectors are not of same length: v1 is {len(self.array)} and v2 is {len(other.array)}"
        else:
            return NotImplemented
    
    def magnitude(self):
        sum_squares = 0
        for i in range(len(self.array)):
            sum_squares += self.array[i] ** 2
        return math.sqrt(sum_squares)
                
v1 = Vector(1, 2, 3)
v2 = Vector(3, 4, 5)
print(f"Vector v1 shape: {v1.shape()}\nVector v2 shape: {v2.shape()}")

v_add = v1 + v2
print(f"Vector addition of {v1} and {v2} = {v_add}")

v_sub = v1 - v2
print(f"Vector subtraction of {v1} and {v2} = {v_sub}")

v_scalar_mul = v1 * 4
print(f"Vector scalar multiply of {v1} and 4 = {v_scalar_mul}")

v_dot = v1.dot(v2)
print(f"Vector dot produce of {v1} and {v2} = {v_dot}")

print(f"Vector magnitude of v1: {v1.magnitude()}\nVector magnitude of v2: {v2.magnitude()}")
