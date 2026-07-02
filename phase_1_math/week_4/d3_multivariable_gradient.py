import math
import numpy as np

def compute_gradient(func, points, h=1e-7):
    grad = np.zeros_like(points, dtype=float)

    for i in range(len(points)):
        # we need to create a pertubation vector here
        e = np.zeros_like(points, dtype=float)
        e[i] = h

        # this is the center diff method
        f_plus = func(points + e)
        f_minus = func(points - e)
        grad[i] = (f_plus - f_minus) / 2 * h
    
    return grad

def test_func_1(points):
    # f(x, y) = x^2 + 3xy + y^2
    x = points[0]
    y = points[1]
    return (x ** 2) + (3 * x * y) + (y ** 2)

def test():
    # need to test f(x, y) = x^2 + 3xy + y^2
    points_1 = np.array([1.0, 2.0])
    gradient_1 = compute_gradient(test_func_1, points_1)
    print(f"The gradient vector of f(x, y) = x^2 + 3xy + y^2 is {gradient_1}")

if __name__ == "__main__":
    test()
