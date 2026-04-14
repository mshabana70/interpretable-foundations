import numpy as np
from typing import Union

class VectorNP():

    def __init__(self, array: Union[list, int, float]):
        self.array = np.array(array)
    
    def __add__(self, other):
        if isinstance(other, np.ndarray):
            new_vec = self.array + other.array 
            return new_vec
        else:
            return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, np.ndarray):
            new_arr = self.array - other.array 
            return new_arr
        else:
            return NotImplemented
        
    def __mul__(self, scalar):
        new_vec = np.multiply(self.array, scalar)
        return new_vec
    
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
                vec_sum = 0
                for i in range(len(self.array)):
                    vec_sum += self.array[i] * other.array[i]
                return vec_sum
            else:
                return f"Vectors are not of same length: v1 is {len(self.array)} and v2 is {len(other.array)}"
        else:
            return NotImplemented
    
    def magnitude(self):
        sum_squares = 0
        for i in range(len(self.array)):
            sum_squares += self.array[i] ** 2
        return math.sqrt(sum_squares)

import time


def test_classes():
    # testing 10,000 dot product operations
    print("Testing custom vector class execution time...")
    test_vect_a = Vector(1, 2, 3, 4)
    test_vect_b = Vector(7, 8, 9, 10)
    custom_start = time.perf_counter()
    for i in range(10000):
        test_custom_dot = test_vect_a.dot(test_vect_b)
        print(f"Custom Dot Product #{i}")
    custom_stop = time.perf_counter()

    total_exec_time = custom_stop - custom_start
    

    print("Testing numpy vector class execution time...")
    test_vect_a = VectorNP([1, 2, 3, 4])
    test_vect_b = VectorNP([7, 8, 9, 10])
    print(f"Array 1: {test_vect_a}\nArray 2: {test_vect_b}")
    np_start = time.perf_counter()
    for i in range(10000):
        test_np_dot = np.dot(test_vect_a.array, test_vect_b.array)
        print(f"NP Dot Product #{i}")
    np_stop = time.perf_counter()
    total_exec_np = np_stop - np_start
    print(f"Total execution time for NumPY class: {total_exec_np:.6f}")
    print(f"Total execution time for Custom: {total_exec_time:.6f}")

    test_vect_a = Vector(1, 2, 3, 4)
    test_vect_b = Vector(7, 8, 9, 10)
    print(f"Dot product result: {test_vect_a.dot(test_vect_b)}")

if __name__ == "__main__":
    test_classes()
    
