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

# Other root finding methods...