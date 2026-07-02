"""Visual accuracy check for d3_multivariable_gradient.compute_gradient.

Plots the gradient field of the 2D test function over its contour lines.
A correct gradient is perpendicular to the contours and points uphill.
The analytical gradient (red) is overlaid so your numerical result (blue)
should land right on top of it.
"""
import numpy as np
import matplotlib.pyplot as plt
from d3_multivariable_gradient import compute_gradient, test_func_1


def analytic_grad_1(x, y):
    # f(x, y) = x^2 + 3xy + y^2  ->  grad = [2x + 3y, 3x + 2y]
    return 2 * x + 3 * y, 3 * x + 2 * y


def visualize_gradient_field(filename="d3_gradient_field.png"):
    # dense grid for the smooth contour surface
    lin = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(lin, lin)
    Z = X**2 + 3 * X * Y + Y**2

    # coarse grid for the arrows (one gradient per node)
    step = np.linspace(-3, 3, 13)
    GX, GY = np.meshgrid(step, step)
    U = np.zeros_like(GX)   # numerical dx component
    V = np.zeros_like(GX)   # numerical dy component
    Ua = np.zeros_like(GX)  # analytical dx component
    Va = np.zeros_like(GX)  # analytical dy component

    for i in range(GX.shape[0]):
        for j in range(GX.shape[1]):
            x, y = GX[i, j], GY[i, j]
            gx, gy = compute_gradient(test_func_1, np.array([x, y]))
            U[i, j], V[i, j] = gx, gy
            Ua[i, j], Va[i, j] = analytic_grad_1(x, y)

    fig, ax = plt.subplots(figsize=(8, 7))
    cs = ax.contour(X, Y, Z, levels=20, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=7)

    # shared scaling so the two fields are directly comparable
    qkw = dict(angles="xy", scale_units="xy", scale=50)
    ax.quiver(GX, GY, Ua, Va, color="red", alpha=0.4, label="analytical", **qkw)
    ax.quiver(GX, GY, U, V, color="blue", alpha=0.85, label="numerical (yours)", **qkw)

    ax.set_title(r"Gradient field of $f(x,y)=x^2+3xy+y^2$ over its contours")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    plt.savefig(filename, dpi=120)
    print(f"saved {filename}")


if __name__ == "__main__":
    visualize_gradient_field()
