import math
from d1_num_differentiation import numerical_derivative

class Expression():
    
    def __init__(self, value):
        self.value = value
    

    def evaluate(self, feed_dict=None):
        raise NotImplementedError
    
    def diff(self, var):
        raise NotImplementedError
    
    def simplify(self):
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
    """We are repr a constant num here"""
    def evaluate(self, feed_dict=None):
        return float(self.value)
    
    def diff(self, var):
        return Constant(0.0)
    
    def __str__(self):
        return str(self.value)

class Variable(Expression):
    """Representing a variable like 'x'"""
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

# unary ops

class Sin(Expression):
    def __init__(self, expr):
        super().__init__(None)
        self.expr = expr

    def evaluate(self, feed_dict=None):
        return math.sin(self.expr.evaluate(feed_dict))
    
    def diff(self, var):
        # d/dx(sin(u)) = cos(u) * du/dx
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
        # d/dx(cos(u)) = -sin(u) * du/dx
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
        # d/dx(exp(u)) = exp(u) * du/dx
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
        # d/dx(ln(x)) = 1/x * d/dx
        return Multiply(Power(self.expr, Constant(-1.0)), self.expr.diff(var))
    
    def __str__(self):
        return f"ln({self.expr})"

# binary ops

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
        # use prod rule here => u * dv/dx + v * du/dx
        return Add( Multiply(self.left, self.right.diff(var)), Multiply(self.right, self.left.diff(var)) )
    
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
            exp_val = self.exponent.value
            # power rule => d/dx(u^n) = n * u^(n-1) * du/dx 
            return Multiply(
                Constant(exp_val),
                Multiply(
                    Power(self.base, Constant(exp_val - 1)),
                    self.base.diff(var)
                )
            )
        else:
            # handling the case where the exponent is a Variable type
            # we can do that with log differentiation => d/dx(u^v) = d/dx(e^{v * ln(u)})
            return Exp(Multiply(self.exponent, Ln(self.base))).diff(var) 
    
    def __str__(self):
        return f"({self.base} ** {self.exponent})"

def test():
    x = Variable("x")

    # Each case targets a different rule. Format: (label, expression, point)
    cases = [
        ("x^2          (power rule)",  Power(x, Constant(2.0)),                1.3),
        ("sin(x)       (trig)",        Sin(x),                                 0.7),
        ("cos(x)       (trig)",        Cos(x),                                 0.7),
        ("exp(x)       (exp)",         Exp(x),                                 0.5),
        ("x^2 + sin(x) (sum rule)",    Add(Power(x, Constant(2.0)), Sin(x)),   1.1),
        ("x * cos(x)   (product)",     Multiply(x, Cos(x)),                    0.9),
        ("exp(sin(x))  (chain rule)",  Exp(Sin(x)),                            0.4),
        ("x^x  (log diff rule)",       Power(x, x),                            2.0),
        ("x^3x  (log diff rule)",      Power(x, Multiply(Constant(3.0), x)),   0.8),
    ]

    for label, expr, point in cases:
        # symbolic: differentiate to a new tree, then evaluate it at the point
        symbolic = expr.diff("x").evaluate({"x": point})
        # numerical ground truth: wrap the expression as a plain f(value)
        numeric = numerical_derivative(lambda v: expr.evaluate({"x": v}), point)

        ok = abs(symbolic - numeric) < 1e-4
        status = "passed!" if ok else "FAILED!"
        print(f"[{status}] d/dx {label} @ x={point}: "
              f"symbolic={symbolic:.6f}  numeric={numeric:.6f}")

if __name__ == "__main__":
    test()