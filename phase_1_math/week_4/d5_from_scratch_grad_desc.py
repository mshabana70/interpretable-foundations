import math
import numpy as np
import matplotlib.pyplot as plt

from d2_expression_class import (
    Expression, Variable, Constant, Sin, Power, Add, Multiply
)

def grad(func, input_vec):
    grad_vec = np.zeros((len(input_vec),), dtype=float)
    map_list = ["x", "y"]
    input_dict = dict(zip(map_list, input_vec.tolist()))
    
    for i, var in enumerate(input_dict.keys()):
        grad_vec[i] = func.diff(var).evaluate(input_dict)
       
    return grad_vec

def grad_descent(func, initial_point, alpha=0.01):

    # for grad descent, we want to take the initial weight
    # pass it to our function and take the gradient with respect to that initial point.
    # This gives us the slope of our descent (negative * ascent) which we will multiply by
    # a step size (our alpha). We take this product and subtract it from our initial points vector,
    # this gives us the next point to move to and repeat the process until our gradient is close to zero (a local/global minimum)

    # algo: x^{k+1} = x^(k) - alpha * grad(func(x^{k}))

    # need to change feed_dict to numpy row vec so we can handle it easier in our grad_desc.
    # we can just remap in the grad func for now (pos 0 -> x, pos 1 -> y, pos 2 -> z)

    x_k = np.array([initial_point[key] for key in initial_point.keys()]) # this should be a input vector of 1 x num of variables
    x_k_1 = np.zeros_like(x_k, dtype=float)
    distance_traveled = x_k # start with initial point as place-holder here
    threshold_check = 1e-7

    # save update history
    history = [x_k]

    while abs(np.linalg.norm(distance_traveled)) > threshold_check: # basically if the size of our update is small, we stop
        
        x_k_1 = x_k - (alpha * grad(func, x_k))

        # swap old point with new point
        distance_traveled = x_k - x_k_1
        x_k = x_k_1 
        history.append(x_k_1)

    return np.array(history)

def plot_contour(history):

    x_range = np.linspace(-5.0, 5.0, 100)
    y_range = np.linspace(-5.0, 5.0, 100)

    X, Y = np.meshgrid(x_range, y_range)

    # define our test func here
    Z = lambda x, y: ((x - 3.0) ** 2.0) + ((y + 1.0) ** 2.0) + (2 * np.sin(x * y))

    # left plot - standard contour lines
    plt.contour(X, Y, Z(X, Y), levels=20, cmap="viridis")
    plt.plot(history[:, 0], history[:, 1], '-ro', label='Descent Path')
    plt.plot(3, -0.5, '*b', markersize=15, label='Global Minimum') # I believe this is the global minimum of the func?
    plt.title("Contour plot of test function")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gradient_descent_plot.png")

def test():

    x = Variable("x")
    y = Variable("y")

    test_case = ("f(x, y) = (x-3)^2 + (y + 1)^2 + 2sin(xy)", Add(Add(Power(Add(x, Constant(-3.0)), Constant(2.0)) , Power(Add(y, Constant(1.0)), Constant(2.0))) , Multiply(Constant(2.0), Sin(Multiply(x, y)))), {"x": -2.0, "y": 2.0})

    test_label, test_expr, test_input = test_case

    expr_descent = grad_descent(test_expr, test_input)
    print(f"The minimum of {test_label} is {expr_descent[-1]}.")
    plot_contour(expr_descent)

if __name__ == "__main__":
    test()


