import math


def numerical_derivative(f, x, h=1e-7):
    
    # for the approx derivative, we will try using
    # the central diff method: f'(x) = f(x + h) - f(x - h) / 2h

    numerator = f(x + h) - f(x - h)
    return numerator / (2 * h) 

def x_squared(x):
    return x ** 2

def sin_x(x):
    return math.sin(x)

def e_x(x):
    return math.exp(x)

def test():

    # x squared test => verify with value of 2x 
    x1 = 10.0
    test1 = numerical_derivative(x_squared, x1)
    verify1 = 2.0 * x1
    print(f"Test 1 passed! {test1} == {verify1}") if abs(test1 - verify1) < 1e-4 else print(f"Test 1 failed! {test1} != {verify1}")
    
    # sinx test => verify with value of cosx
    x2 = 20.0
    test2 = numerical_derivative(sin_x, x2)
    verify2 = math.cos(x2) # derv of sinx
    print(f"Test 2 passed! {test2} == {verify2}") if abs(test2 - verify2) < 1e-4 else print(f"Test 2 failed! {test2} != {verify2}")

    # e^x test => verify with value of e^x
    x3 = 5.0
    test3 = numerical_derivative(e_x, x3)
    verify3 = math.exp(x3) # derv of e^x
    print(f"Test 3 passed! {test3} == {verify3}") if abs(test3 - verify3) < 1e-4 else print(f"Test 3 failed! {test3} != {verify3}")

if __name__ == "__main__":
    test()