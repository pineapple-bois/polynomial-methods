def newton(f, df, x0, max_it=20, tol=1e-3):
    iter = 0

    # Check if the initial guess is very close to a root
    if abs(f(x0)) < tol:
        return x0, True, iter

    # Continue iterating until the maximum number of iterations is reached
    while iter < max_it:
        f0 = f(x0)
        dfdx0 = df(x0)

        if abs(f0) < tol:
            # Convergence achieved
            return x0, True, iter

        if abs(dfdx0) < tol:
            # Newton's method fails to converge due to zero derivative
            raise RuntimeError("Newton's method failed to converge.")

        x1 = x0 - f0 / dfdx0  # Compute the next guess for the root
        if abs(x1 - x0) < tol:
            # Convergence achieved
            return x1, True, iter

        x0 = x1  # Update the current guess
        iter += 1  # Increment the iteration counter

    return x0, False, iter


def bisection(f, a, b, max_iter=1000, tol=1e-3):
    # Check if the function has no roots or more than one root in the given interval
    if f(a) * f(b) > 0:
        print(f'No roots or more than one root in [{a}, {b}]')
        return None

    # Initialize the midpoint of the interval
    m = (a + b) / 2

    # Perform the bisection method until the desired tolerance or maximum number of iterations is reached
    iterations = 0
    while abs(f(m)) > tol:
        # Break the loop if the method does not converge within max_iter iterations
        if iterations >= max_iter:
            print(f"The method did not converge after {max_iter} iterations.")
            return None

        # Check which half of the interval to update based on the sign of f(a)*f(m)
        if f(a) * f(m) < 0:  # If 'a' and 'm' bracket a root, update 'b' to 'm'
            b = m
        elif f(b) * f(m) < 0:  # If 'b' and 'm' bracket a root, update 'a' to 'm'
            a = m

        # Compute the new midpoint
        m = (a + b) / 2
        iterations += 1

    # Return the approximate root
    return m