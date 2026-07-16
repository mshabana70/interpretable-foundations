import math
import numpy as np

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection

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
        self.step_history_momentum = []
        self.step_history_vanilla = []
        self.step_history_nesterov = []

    def derive(self):
        self.partials = []
        for var in self.variables:
            # we are calculating the partial derivatives upfront for every variable in function
            # and storing the partials in a list to eval against in our grad loop
            self.partials.append(self.function.diff(var))
        return self.partials
    
    def run_vanilla(self, alpha=0.001, n_iters=1000, tol=1e-6):
        """
        Vanilla GD with no bells and wistles.

        Returns the trajectory (list of points) so we can plot convergence.
        """

        points = np.array(self.inputs)
        partial_derivatives = self.derive()

        self.step_history_vanilla.append(points)
        t = 0
        feed_dict_grad = dict(self.feed_dict)
        grad = np.array([partial.evaluate(feed_dict_grad) for partial in partial_derivatives])

        while (abs(np.linalg.norm(grad)) > tol) and t < n_iters:
            
            grad = np.array([partial.evaluate(feed_dict_grad) for partial in partial_derivatives])
            update = alpha*grad
            points = points - update
            
            # updates

            # need to update our feed_dict
            for idx, (k, v) in enumerate(feed_dict_grad.items()):
                feed_dict_grad[k] = points[idx]

            self.step_history_vanilla.append(points)
            tol_test = self.step_history_vanilla[-2] - self.step_history_vanilla[-1]
            t += 1

        check_step_size = abs(np.linalg.norm(tol_test))
        if t < n_iters and check_step_size < tol:
            print(f"Vanilla Convergence achieved at {self.step_history_vanilla[-1]} on step {t}")
        elif t == n_iters and check_step_size > tol:
            print(f"Vanilla GD failed to converge... Stuck at {self.step_history_vanilla[-1]} at step {t}")
    
    def run_momentum(self, alpha=0.001, beta=0.9, decay=0.0, n_iters=1000, tol=1e-6):
        """
        GD with momentum and LR decay.

        Returns the trajectory (list of points) so we can plot convergence.
        """

        points = np.array(self.inputs)

        momentum = np.zeros_like(points)
        partial_derivatives = self.derive()

        self.step_history_momentum.append(points)
        t = 0
        feed_dict_grad = dict(self.feed_dict)
        grad = np.array([partial.evaluate(feed_dict_grad) for partial in partial_derivatives])

        while (abs(np.linalg.norm(grad)) > tol) and t < n_iters:
            
            grad = np.array([partial.evaluate(feed_dict_grad) for partial in partial_derivatives])
            alpha_t = alpha / (1 + decay*t)
            momentum = beta*momentum - alpha_t*grad
            points = points + momentum
            
            # updates

            # need to update our feed_dict
            for idx, (k, v) in enumerate(feed_dict_grad.items()):
                feed_dict_grad[k] = points[idx]

            self.step_history_momentum.append(points)
            tol_test = self.step_history_momentum[-2] - self.step_history_momentum[-1]
            t += 1

        check_step_size = abs(np.linalg.norm(tol_test))
        if t < n_iters and check_step_size < tol:
            print(f"Momentum GD Convergence achieved at {self.step_history_momentum[-1]} on step {t}")
        elif t == n_iters and check_step_size > tol:
            print(f"Momentum GD failed to converge... Stuck at {self.step_history_momentum[-1]} at step {t}")
    
    def run_nesterov(self, alpha=0.001, beta=0.9, decay=0.0, n_iters=1000, tol=1e-6):
        """
        GD with nesterov look-ahead using momentum.

        Returns the trajectory (list of points) so we can plot convergence.
        """
        points = np.array(self.inputs)
        curr_momentum = np.zeros_like(points)
        partial_derivatives = self.derive()

        self.step_history_nesterov.append(points)
        t = 0
        feed_dict_grad = dict(self.feed_dict)
        grad_check = np.array([partial.evaluate(feed_dict_grad) for partial in partial_derivatives])

        while (abs(np.linalg.norm(grad_check)) > tol) and t < n_iters:

            # we need to calculate the momentum first, then subtract it from our current position points.
            look_ahead_points = points + (beta*curr_momentum)

            # need to update our feed_dict
            for idx, (k, v) in enumerate(feed_dict_grad.items()):
                feed_dict_grad[k] = look_ahead_points[idx]

            # now take the grad at this look_ahead
            look_ahead_grad = np.array([partial.evaluate(feed_dict_grad) for partial in partial_derivatives])

            # compute our momentum based on our look ahead correction
            alpha_t = alpha / (1 + decay*t)
            corrected_momentum = beta*curr_momentum - alpha_t*look_ahead_grad

            # we update our original position based on this corrected momentum
            points = points + corrected_momentum

            # we have to update our feed_dict again but with our actual update position, not the look_ahead position
            for idx, (k, v) in enumerate(feed_dict_grad.items()):
                feed_dict_grad[k] = points[idx]
            
            # now we save our progression
            grad_check = np.array([partial.evaluate(feed_dict_grad) for partial in partial_derivatives])
            self.step_history_nesterov.append(points)
            curr_momentum = corrected_momentum
            tol_test = self.step_history_nesterov[-2] - self.step_history_nesterov[-1]
            t += 1
        
        check_step_size = abs(np.linalg.norm(tol_test))
        if t < n_iters and check_step_size < tol:
            print(f"Nesterov GD Convergence achieved at {self.step_history_nesterov[-1]} on step {t}")
        elif t == n_iters and check_step_size > tol:
            print(f"Nesterov GD failed to converge... Stuck at {self.step_history_nesterov[-1]} at step {t}")
    
    def plot(self, ax=None, path_color="#ff9500", label=None, draw_field=True):
        traj = np.array(self.step_history)
        xs, ys = traj[:, 0], traj[:, 1]

        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 6))
        
        if draw_field:
            pad = 0.5
            gx = np.linspace(xs.min() - pad, xs.max() + pad, 400)
            gy = np.linspace(ys.min() - pad, ys.max() + pad, 400)
            GX, GY = np.meshgrid(gx, gy)
            a, b = 2.0, 100.0
            Z = ((a - GX) ** 2) + (b * (GY - GX ** 2)**2)
            levels = np.logspace(-1, 3.5, 25)
            ax.contourf(GX, GY, Z, levels=levels, norm=LogNorm(), cmap="viridis", alpha=0.9)
            ax.plot(2.0, 4.0, "*", color="gold", mec="black", ms=18, label="minimum (2, 4)")
        
        pts  = traj.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="autumn", array=np.arange(len(segs)), lw=1.6)
        ax.add_collection(lc)
        ax.plot(xs[0], ys[0], "o", color="white", mec="black", ms=8)   # start
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title("Gradient descent trajectory on Rosenbrock")
        ax.legend(loc="upper left", framealpha=0.9)
        return ax
    
    def plot_convergence(self, ax=None, color="#ff9500", label=None):
        traj = np.array(self.step_history)
        fx = (2.0 - traj[:,0])**2 + 100.0*(traj[:,1] - traj[:,0]**2)**2
        if ax is None: _, ax = plt.subplots()
        ax.semilogy(fx, color=color, label=label)          # <-- semilogy, not plot
        ax.set_xlabel("iteration"); ax.set_ylabel("f(x, y)  (log)")
        ax.legend(); ax.grid(alpha=0.3)
        return ax
        

if __name__ == "__main__":

    x = Variable("x")
    y = Variable("y")

    a = 2.0
    b = 100.0
    
    # Rosenbrock func => f(x, y) = (a - x)^2 + b(y - x^2)^2
    rosenbrock_func = Add(Power(Add(Constant(a), Multiply(Constant(-1.0), x)) ,Constant(2.0)), Multiply(Constant(b), Power(Add(y, Multiply(Constant(-1.0), Power(x, Constant(2.0)))), Constant(2.0))))

    # Test on quadratic bowl function: x^2 + y^2
    quad_bowl_func = Add(Power(x, Constant(2.0)), Power(y, Constant(2.0)))

    
    # define gradient descent objects
    feed_dict1 = {"x": -1.5, "y": 1.5}
    gd = GradientDescent(rosenbrock_func, feed_dict1)
    gd.run_vanilla(n_iters=20000)

    gd.run_momentum(beta=0.9, n_iters=20000)

    gd.run_nesterov(beta=0.9, n_iters=20000)

    # define gradient descent objects
    feed_dict2 = {"x": 9.0, "y": 10.5}
    gd_bowl = GradientDescent(quad_bowl_func, feed_dict2)
    gd_bowl.run_vanilla(n_iters=20000)

    gd_bowl.run_momentum(beta=0.9, n_iters=20000)

    gd_bowl.run_nesterov(beta=0.9, n_iters=20000)

    # # Figure 1 — trajectory (shared spatial axes)
    # ax_traj = plain.plot(path_color="#ff4d4d", label="Beta = 0.0")
    # mom.plot(ax=ax_traj, path_color="#ff9500", label="Beta = 0.9", draw_field=False)
    # ax_traj.figure.savefig("./figs/D2_rosenbrock_GD_trajectory.png", dpi=130)

    # # Figure 2 — convergence (its OWN iteration–loss axes; do NOT pass ax_traj)
    # ax_conv = plain.plot_convergence(color="#ff4d4d", label="Beta = 0.0")
    # mom.plot_convergence(ax=ax_conv, color="#ff9500", label="Beta = 0.9")
    # ax_conv.figure.savefig("./figs/D2_rosenbrock_GD_convergence.png", dpi=130)

        
