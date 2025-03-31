import numpy as np


def inverse_interpolation(xs, ys, y_target, tol=1e-6, max_iter=100):
    """
    Perform iterated inverse interpolation to approximate the solution of x - e^(-x) = 0.

    :param xs: List of x values.
    :param ys: List of corresponding y values.
    :param y_target: The target y value (which is 0 in our case).
    :param tol: Tolerance for convergence.
    :param max_iter: Maximum number of iterations.
    :return: Approximated root.
    """
    for _ in range(max_iter):
        # Perform linear interpolation to estimate the next x
        x_new = np.interp(y_target, ys, xs)

        # Compute new y value for the updated x
        y_new = x_new - np.exp(-x_new)

        # Check for convergence
        if abs(y_new - y_target) < tol:
            return x_new

        # Update data points to include the new estimate
        xs.append(x_new)
        ys.append(y_new)

        # Sort the points based on y values for better interpolation
        sorted_indices = np.argsort(ys)
        xs = [xs[i] for i in sorted_indices]
        ys = [ys[i] for i in sorted_indices]

    return x_new  # Return the last approximation if max_iter is reached


# Given data points
x_values = [0.3, 0.4, 0.5, 0.6]
y_values = [x - np.exp(-x) for x in x_values]

target_y = 0  # We want to solve x - e^(-x) = 0

# Perform inverse interpolation to find the root
solution = inverse_interpolation(x_values, y_values, target_y)
print(f"Approximate solution: x ≈ {solution:.6f}")
