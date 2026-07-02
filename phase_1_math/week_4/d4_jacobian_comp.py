import math
import numpy as np 
from d2_expression_class import (
    Add, Multiply, Power, Constant, Variable, Expression, Sin, Cos, Exp, Ln
)

def compute_gradient_numerical(func, points, h=1e-7):
    # taking what we did on day 3
    grad = np.zeros_like(points, dtype=float)

    for i in range(len(points)):
        e = np.zeros_like(points, dtype=float)
        x_i = points[i]
        # going to try scaling the step size for more approximations 
        e[i] = h * max(abs(x_i), 1)

        # Because we added the value of h to our e vector at pos i, adding e to points will only 
        # perturb the variable at pos i, giving the same thing as x_i + h, with all variables remaining the same. 
        # Wasn't an obvious solution to me in the beginning lol
        f_plus = func(points + e) 
        f_minus = func(points - e)

        grad[i] = (f_plus - f_minus) / (2*e[i])
    
    return grad

def jacobian_numerical(functions, points, h=1e-7):
    """
    Compute Jacobian Matrix using central differentiation method for gradients.
    """
    # we have a vector of functions F and a vector of inputs X.
    # We can get the gradient of F with respect to a variable x_i (one variable of X)
    # by computing the gradient for each func f_i in our vector of funcs F with respect to
    # that one variable x_i.
    # This gives us one column vector of gradients at variable x_i. 
    # We repeat this for every variable x_i in X, appending the new gradient column vector to a matrix
    # until i reaches n => where n is the number of variables in X. 
    # The resulting matrix is the Jacobian Matrix. 

    # Since we have a compute_gradient() function that takes one function and multiple variables,
    # we can just have an outer loop that parses our vector of functions F, pass f_i and all variables,
    # and get back a row vector of f_i for all variables X. 
    # This row vector should be same length as the number of variables in X. This will be our rows in 
    # our Jacobian matrix. For m functions in vector F, we should have an m x n matrix at the end.

    # get our matrix dims first
    m = len(functions)
    n = len(points)

    jacob_mat = np.zeros((m, n), dtype=float)

    # Just parse all the funcs in functions and row replace in jacob matrix. 
    # This is O(m x n) runtime if anyone cared to know lol.
    for i in range(m):
        jacob_mat[i] = compute_gradient_numerical(functions[i], points)
    
    return jacob_mat

def compute_gradient_symbolic(func, points, curr_var):
    # we need to differentiation the function at a given variable x_i.
    # hence why we are passing curr_var
    # we diff at the curr_var but eval at all points. Since diff takes a str, we pass a string for curr_var
    grad_i = func.diff(curr_var).evaluate(points)
    return grad_i


def jacobian_symbolic(functions, points_dict):
    """
    Compute Jacobian Matrix using symbolic differentiation from our day 3 work.

    This is taking advantage of reverse-mode auto-differentiation, which is WAY more
    efficient than our numerical differentiation method.

    Just takes more time writing the test cases :(
    """

    m = len(functions)
    n = max(len(feed_dict) for feed_dict in points_dict)

    jacob_mat = np.zeros((m, n), dtype=float)

    for i, expr in enumerate(functions):
        # grab the points for the curr function
        grad = np.zeros((n, 1), dtype=float)
        for j, (var, value) in enumerate(points_dict[i].items()):
            grad[j] = compute_gradient_symbolic(expr, points_dict[i], var)
            # print(f"j: {j}, Var: {var}, Value: {value}, Expression: {expr}, Grad: {grad[j]}")
        jacob_mat[i] = grad.T
    
    return jacob_mat

    

def test_func_1(points):
    # f(x, y) = x^2 + 3xy + y^2
    x = points[0]
    y = points[1]
    return (x ** 2) + (3 * x * y) + (y ** 2)

def test_func_2(points):
    x = points[0]
    y = points[1]
    return ((x ** 2) * (y ** 2)) + x + y

def test_func_3(points):
    x = points[0]
    y = points[1]
    return x * (y ** 2) + 20.0

def test_func_4(points):
    x = points[0]
    y = points[1]
    return (x ** 4) + ((y ** 3) * (x ** 2)) + y + 8.0

def test():

    # numerical tests
    functions = [test_func_1, test_func_2, test_func_3, test_func_4]
    points = np.array([1.0, 2.0])

    jacobian_num = jacobian_numerical(functions, points)
    print(f"Jacobian Matrix (Numerical Method):\n{jacobian_num}")

    # symbolic tests
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    # Each case targets a different rule. Format: (label, expression, points)
    cases = {
        Add(Add(Power(x, Constant(2.0)), Multiply(Constant(3.0), Multiply(x, y))), Power(y, Constant(2.0))): {"x": 1.0, "y": 2.0},
        Add(Multiply(x, Power(y, Constant(2.0))), Constant(20.0)): {"x": 1.0, "y": 2.0},
        Add(Power(x, Constant(2.0)), x): {"x": 1.3},
        Add(Power(x, Constant(2.0)), y): {"x": 0.7, "y": 1.2},
        Add(Multiply(x, Power(y, Constant(2.0))), Constant(10.0)): {"x": 2.0, "y": 5.0},
        Add(
            Add(Add(Power(x, Constant(4.0)), Multiply(
                Power(y, Constant(2.0)), Power(z, Constant(2.0)))), x), Constant(8.0)): {"x": 1.0, "y": 2.0, "z": 3.0}
    }

    # we need to do some setup here for the symbolic functions and their variables
    functions = list(cases.keys())
    points_dict = list(cases.values()) # list of feed_dicts for our expressions objects

    jacobian_sym = jacobian_symbolic(functions, points_dict)
    print(f"Jacobian Matrix (Symbolic Method):\n{jacobian_sym}")
    


if __name__ == "__main__":
    test()