import math
import numpy as np

# reimplementing symbolic differentiation

class Expression():

    def __init__(self, value):
        self.value = value

    def evaluate(self, feed_dict=None):
        raise NotImplementedError

    def diff(self, var):
        raise NotImplementedError
    
    def simply(self):
        return self
    
    def __add__(self, other):
        return Add(self, self._wrap(other))
    
    def __radd__(self, other):
        return Add(self._wrap(other), self)

    def __mul__(self, other):
        return Multiply(self, self._wrap(other))
    
    def __rmul__(self, other):
        return Multiply(self._wrap(other), self)
    
    def __pow__(self, power):
        return Power(self, self._wrap(power))
    
    def _wrap(self, other):
        return other if isinstance(other, Expression) else Constant(other)
    
    def __repr__(self):
        return str(self)
    
class Constant(Expression):
    """Representing a Constant number."""
    def evaluate(self, feed_dict=None):
        return float(self.value)
    
    def diff(self, var):
        return Constant(0.0)
    
    def __str__(self):
        return str(self.value)
    
class Variable(Expression):
    """Representing a variable like 'x', or 'y'"""

    def __init__(self, name):
        super().__init__(None)
        self.name = name
    
    def evaluate(self, feed_dict=None):
        if feed_dict and self.name in feed_dict:
            return feed_dict[self.name]
        raise ValueError(f"Value for variable '{self.name}' not provided.")
    
    def diff(self, var):
        return Constant(1.0) if self.name == var else Constant(0.0)
    
    def __str__(self):
        return self.name

class Sin(Expression):

    def __init__(self, expr):
        super().__init__(None)
        self.expr = expr

    def evaluate(self, feed_dict=None):
        return math.sin(self.expr.evaluate(feed_dict))
    
    def diff(self, var):
        return Multiply(Cos(self.expr), self.expr.diff(var))
    
    def __str__(self):
        return f"sin({self.expr})"
    
class Cos(Expression):
    def __init__(self, expr):
        super().__init__(None)
        self.expr = expr

    def evaluate(self, feed_dict=None):
        return math.cos(self.expr.evaluate(feed_dict))
    
    def diff(self, var):
        return Multiply(Constant(-1.0), Multiply(Sin(self.expr), self.expr.diff(var)))
    
    def __str__(self):
        return f"cos({self.expr})"

class Exp(Expression):

    def __init__(self, expr):
        super().__init__(None)
        self.expr = expr

    def evaluate(self, feed_dict=None):
        return math.exp(self.expr.evaluate(feed_dict))
    
    def diff(self, var):
        return Multiply(Exp(self.expr), self.expr.diff(var))
    
    def __str__(self):
        return f"exp({self.expr})"
    
class Ln(Expression):

    def __init__(self, expr):
        super().__init__(None)
        self.expr = expr

    def evaluate(self, feed_dict=None):
        return math.log(self.expr.evaluate(feed_dict))
    
    def diff(self, var):
        return Multiply(Power(self.expr, Constant(-1.0)), self.expr.diff(var))
    
    def __str__(self):
        return f"ln({self.expr})"
    

### OPS

class Add(Expression):

    def __init__(self, left, right):
        super().__init__(None)
        self.left = left
        self.right = right

    def evaluate(self, feed_dict=None):
        return self.left.evaluate(feed_dict) + self.right.evaluate(feed_dict)
    
    def diff(self, var):
        return Add(self.left.diff(var), self.right.diff(var))
    
    def __str__(self):
        return f"({self.left} + {self.right})"
    
class Multiply(Expression):
    def __init__(self, left, right):
        super().__init__(None)
        self.left = left
        self.right = right

    def evaluate(self, feed_dict=None):
        return self.left.evaluate(feed_dict) * self.right.evaluate(feed_dict)
    
    def diff(self, var):
        return Add(Multiply(self.left, self.right.diff(var)), Multiply(self.left.diff(var), self.right))
    
    def __str__(self):
        return f"({self.left} * {self.right})"
    
class Power(Expression):
    def __init__(self, base, exponent):
        super().__init__(None)
        self.base = base
        self.exponent = exponent

    def evaluate(self, feed_dict=None):
        return self.base.evaluate(feed_dict) ** self.exponent.evaluate(feed_dict)
    
    def diff(self, var):

        if isinstance(self.exponent, Constant):
            exp_val =  self.exponent.value
            return Multiply(
                Constant(exp_val),
                Multiply(
                    Power(self.base, Constant(exp_val - 1)),
                    self.base.diff(var)
                )
            )
        else:
            # log differentiation
            return Exp(Multiply(self.exponent, Ln(self.base))).diff(var)
        
    
    def __str__(self):
        return f"({self.base} ** {self.exponent})"
    

class GradientDescent():

    def __init__(self, function, feed_dict):
        self.function = function
        self.feed_dict = feed_dict
        self.inputs = list(feed_dict.values()) # should be a list of floats
        self.variables = list(feed_dict.keys()) # should be a list of strings
        self.step_history = []

    def derive(self):
        self.partials = []
        for var in self.variables:
            # we are calculating the partial derivatives upfront for every variable in function
            # and storing the partials in a list to eval against in our grad loop
            self.partials.append(self.function.diff(var))
        return self.partials
    
    def run(self, alpha=0.001, beta=0.9, decay=0.0, n_iters=1000, tol=1e-6):
        """
        Returns the trajectory (list of points) so we can plot convergence.
        """

        points = np.array(self.inputs)

        momentum = np.zeros_like(points)
        tol_test = points
        partial_derivatives = self.derive()

        self.step_history.append(points)
        t = 0
        feed_dict_grad = self.feed_dict

        while (abs(np.linalg.norm(tol_test)) > tol) and t < n_iters:
            
            grad = np.array([partial.evaluate(feed_dict_grad) for partial in partial_derivatives])
            alpha_t = alpha / (1 + decay*t)
            momentum = beta*momentum - alpha_t*grad
            points = points + momentum
            
            # updates

            # need to update our feed_dict
            for idx, (k, v) in enumerate(feed_dict_grad.items()):
                feed_dict_grad[k] = points[idx]

            self.step_history.append(points)
            tol_test = self.step_history[-2] - self.step_history[-1]
            t += 1

        check_step_size = abs(np.linalg.norm(tol_test))
        if t < n_iters and check_step_size < tol:
            print(f"Convergence achieved at {self.step_history[-1]} on step {t}")
        elif t == n_iters and check_step_size > tol:
            print(f"GD failed to converge... Stuck at {self.step_history[-1]} at step {t}")
    
    def plot(self):
        pass

if __name__ == "__main__":

    x = Variable("x")
    y = Variable("y")

    a = 2.0
    b = 100.0
    
    # Rosenbrock func => f(x, y) = (a - x)^2 + b(y - x^2)^2
    rosenbrock_func = Add(Power(Add(Constant(a), Multiply(Constant(-1.0), x)) ,Constant(2.0)), Multiply(Constant(b), Power(Add(y, Multiply(Constant(-1.0), Power(x, Constant(2.0)))), Constant(2.0))))

    feed_dict1 = {"x": -1.5, "y": 1.5}
    # define gradient descent object
    gd = GradientDescent(function=rosenbrock_func, feed_dict=feed_dict1)

    # run gradient descent
    gd.run(beta=0.0, n_iters=10000)

    gd.run(beta=0.9, n_iters=10000)


        
