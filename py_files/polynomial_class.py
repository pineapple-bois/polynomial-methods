import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import sympy
from py_files import root_finding


class Polynomial:
    """A class to represent a polynomial."""
    def __init__(self, coefficients: dict):
        """
        Initialize the Polynomial class.

        Args:
        coefficients (dict): A dictionary of coefficients, where each key is a tuple of exponents
        corresponding to each variable in the polynomial, and the value is the coefficient.
        """
        if not isinstance(coefficients, dict):
            raise ValueError("Coefficients must be provided as a dictionary.")

        # Ensure all keys are tuples and padded with 0 for missing elements
        self._coeff = {}
        for key, value in coefficients.items():
            if isinstance(key, int):  # If the key is an int (i.e., it's a uni-variate polynomial)
                key = (key, 0, 0)  # Convert it to a tuple and pad with 0s
            elif isinstance(key, tuple) and len(key) == 1:  # If the key is a tuple with 1 element
                key = (key[0], 0, 0)  # Pad with 0s
            self._coeff[key] = value

        self.expression = None
        # Check if it's a multivariate polynomial
        self.is_multivariate = any(exponent[1] != 0 or exponent[2] != 0 for exponent in self._coeff.keys())

    @property
    def coeff(self):
        return self._coeff

    @coeff.setter
    def coeff(self, value):
        self._coeff = value

    def __repr__(self):
        """Return a string representation of the Polynomial instance."""
        return f"Polynomial({self._coeff})"

    def __add__(self, other):
        coeff_sum = self._coeff.copy()  # Start with the coefficients of this polynomial

        # Check if we're adding a univariate to a multivariate
        if self.is_multivariate and not other.is_multivariate:
            # Convert other's keys from int to tuples
            other = Polynomial({(key[0] if isinstance(key, tuple) else key, 0, 0):
                                    coeff for key, coeff in other._coeff.items()})
        elif not self.is_multivariate and other.is_multivariate:
            # Convert self's keys from int to tuples
            coeff_sum = {(key[0] if isinstance(key, tuple) else key, 0, 0):
                                coeff for key, coeff in coeff_sum.items()}

        # Iterate over the coefficients of the other polynomial
        for powers, coeff in other._coeff.items():
            # If powers already in coeff_sum, add the coefficients
            if powers in coeff_sum:
                coeff_sum[powers] += coeff
            else:  # Otherwise, add the new term
                coeff_sum[powers] = coeff

        # Return a new Polynomial instance with the summed coefficients
        return Polynomial(coeff_sum)

    def __sub__(self, other):
        # Start with the coefficients of this polynomial
        coeff_difference = self._coeff.copy()

        # If self is univariate and other is multivariate, change structure of self
        if not self.is_multivariate and other.is_multivariate:
            coeff_difference = {(key[0] if isinstance(key, tuple) else key, 0, 0): coeff for key, coeff in
                                coeff_difference.items()}

        # If self is multivariate and other is univariate, change structure of other
        elif self.is_multivariate and not other.is_multivariate:
            other = Polynomial(
                {(key[0] if isinstance(key, tuple) else key, 0, 0): coeff for key, coeff in other._coeff.items()})

        # Iterate over the coefficients of the other polynomial
        for powers, coeff in other._coeff.items():
            # If powers already in coeff_difference, subtract the coefficients
            if powers in coeff_difference:
                coeff_difference[powers] -= coeff
            else:  # Otherwise, add the new term with negative coeff
                coeff_difference[powers] = -coeff

        # Return a new Polynomial instance with the difference of coefficients
        return Polynomial(coeff_difference)

    def __mul__(self, other):
        # Create a dictionary to hold the coefficients of the result
        coeff_product = {}

        # For each term in the first polynomial
        for powers1, coeff1 in self._coeff.items():
            # For each term in the other polynomial
            for powers2, coeff2 in other._coeff.items():
                # Determine the power and coefficient of the resulting term
                new_powers = tuple(x + y for x, y in zip(powers1, powers2))
                new_coeff = coeff1 * coeff2

                # If the resulting term is already in coeff_product, add the new coefficient
                if new_powers in coeff_product:
                    coeff_product[new_powers] += new_coeff
                else:  # Otherwise, add the new term
                    coeff_product[new_powers] = new_coeff

        # Return a new Polynomial instance with the product of coefficients
        return Polynomial(coeff_product)

    def __rmul__(self, other):
        # Check if other is a scalar (int or float)
        if isinstance(other, (int, float)):
            # Multiply each coefficient by the scalar
            coeff_product = {key: coeff * other for key, coeff in self._coeff.items()}
            # Return a new Polynomial instance with the product of coefficients
            return Polynomial(coeff_product)
        else:
            return NotImplemented

    def __str__(self):
        terms = []
        for powers, coeff in sorted(self._coeff.items(), key=lambda x: sum(x[0]), reverse=True):
            if all(power == 0 for power in powers):
                term = str(coeff)
            else:
                # create variables strings
                vars_str = ['x', 'y', 'z']
                vars_with_power = [f"{vars_str[i]}^{power}" if power > 1 else f"{vars_str[i]}" for i, power in
                                   enumerate(powers) if power != 0]
                if self.expression is None:  # modify the coefficient only if self.expression is None
                    term = (
                        "-{}".format(' * '.join(vars_with_power)) if coeff == -1.0 else
                        "{}".format(' * '.join(vars_with_power)) if coeff == 1.0 else
                        f"{round(coeff, 4) if '.' in str(coeff) and len(str(coeff).split('.')[1]) > 4 else coeff}"
                        f" * {' * '.join(vars_with_power)}"
                    )
                else:  # if self.expression is not None, don't round the coefficient
                    term = f"{coeff} * {' * '.join(vars_with_power)}"
            # Append the term to the list of terms
            terms.append(term)

        string = " + ".join(terms)
        string = string.replace(" + -", " - ")
        for var in ['x', 'y', 'z']:
            string = string.replace(f"{var}^1 ", f"{var} ")
        if string.startswith(" + "):
            string = string[3:]
        elif string.startswith(" - "):
            string = "-" + string[3:]

        return string

    def __call__(self, *args):
        """Evaluate the polynomial at a specific point.

        Args:
        *args: The coordinates at which to evaluate the polynomial.
        x for uni-variate
        x, y, z for multivariate

        Returns:
        float: The value of the polynomial at the specified coordinates.
        """
        if self.is_multivariate:
            return self._eval_multivariate(*args)
        else:
            return self._eval_univariate(*args)

    def _eval_univariate(self, x):
        value = 0
        for powers, coeff in self.coeff.items():
            value += coeff * x ** powers[0]
        return value

    def _eval_multivariate(self, x, y, z):
        value = 0
        for powers, coeff in self.coeff.items():
            value += coeff * (x ** powers[0]) * (y ** powers[1]) * (z ** powers[2])
        return value

    def _convert_to_sympy_expression(self, rational=False, force=False):
        """
        Converts the current polynomial expression into a sympy expression.

        Parameters:
        rational (bool): If True, the coefficients are converted into sympy Rational type. Default is False.
        force (bool): If True, the function forcibly converts to sympy expression even if it already exists.
                      Default is False.

        This function does not return any value. It updates the 'expression' attribute of the Polynomial instance.
        """
        if self.expression is None or force:
            # Create the symbols x, y, z
            x = sympy.Symbol('x')
            y = sympy.Symbol('y')
            z = sympy.Symbol('z')

            if self.is_multivariate:    # different rule for multivariate
                # Build the polynomial
                if rational:
                    polynomial = sum(
                        sympy.Rational(coeff) * x ** powers[0] * y ** powers[1] * z ** powers[2]
                        for powers, coeff in self._coeff.items())
                else:
                    polynomial = sum(
                        coeff * x ** powers[0] * y ** powers[1] * z ** powers[2]
                        for powers, coeff in self._coeff.items())
                # Create a sympy.Poly object
                self.expression = sympy.Poly(polynomial, x, y, z)

            else:
                if rational:
                    polynomial = sum(
                        sympy.Rational(str(coeff)) * x ** powers[0] for powers, coeff in self._coeff.items())
                else:
                    polynomial = sum(
                        coeff * x ** powers[0] for powers, coeff in self._coeff.items())
                # Create a sympy.Poly object
                self.expression = sympy.Poly(polynomial, x)

    def display(self):
        """
        Displays the sympy expression of the polynomial using IPython's display system.

        The method does not take any parameters and does not return any value.
        The method calls the '_convert_to_sympy_expression' function internally.
        Converts the polynomial to a sympy expression.
        """
        self._convert_to_sympy_expression()
        display(self.expression)  # This will use Jupyter's display system.

    def save_as_sympy(self, rational=False):
        """
        Saves the polynomial as a sympy expression and returns it.

        Parameters:
        rational (bool): If True, the coefficients are converted into sympy Rational type. Default is False.

        Returns:
        sympy.core.expr.Expr : A sympy expression representing the polynomial.

        The method calls the '_convert_to_sympy_expression' function.
        Converts the polynomial to a sympy expression.
        """
        self._convert_to_sympy_expression(rational, force=True)
        return self.expression


class UniPoly(Polynomial):
    """A class to represent a uni-variate polynomial. Inherits from the Polynomial class."""
    def __init__(self, coefficients: dict):
        """
        Initialize the UniPoly class.

        Args:
        coefficients (dict): A dictionary of coefficients, where each key is an integer exponent,
        and the value is the coefficient.
        """
        super().__init__(coefficients)
        if self.is_multivariate:
            raise ValueError("The provided coefficients don't represent a uni-variate polynomial.")

    def plot(self, x_min=-10, x_max=10, num_points=1000):
        """Plot the polynomial over a specified range.

        Args:
        x_min (float, optional): The minimum x-value for the plot. Defaults to -10.
        x_max (float, optional): The maximum x-value for the plot. Defaults to 10.
        num_points (int, optional): The number of points to generate for the plot. Defaults to 1000.
        """
        x = np.linspace(x_min, x_max, num_points)
        y = [self._eval_univariate(x_val) for x_val in x]

        plt.plot(x, y)
        plt.axhline(0, color='r', linewidth=0.25)
        formula = self.__str__()
        formula = formula.replace('*', '')
        plt.title(f'${formula}$')
        plt.grid(True)
        plt.show()

    def differentiate(self):
        new_coeff = {}
        for powers, coeff in self.coeff.items():
            if powers[0] > 0:
                new_coeff[(powers[0]-1,)] = coeff * powers[0]
        return UniPoly(new_coeff)

    def integrate(self, constant=0):
        # Create a dictionary to hold the coefficients of the integral
        integral_coeff = {}

        # Iterate over the coefficients of the polynomial
        for powers, coeff in self.coeff.items():
            # Calculate the new power and new coefficient
            new_power = powers[0] + 1
            new_coeff = coeff / new_power

            # Add the new term to the integral
            integral_coeff[(new_power,)] = new_coeff

        # Don't forget about the constant of integration
        if constant != 0:
            integral_coeff[(0,)] = constant

        # Return a new UniPoly instance with the coefficients of the integral
        return UniPoly(integral_coeff)

    def integrate_definite(self, lower_limit, upper_limit):
        antiderivative = self.integrate()
        return antiderivative(upper_limit) - antiderivative(lower_limit)

    def newton(self, x0, max_it=20, tol=1e-3):
        return root_finding.newton(self._eval_univariate, self.differentiate()._eval_univariate, x0, max_it, tol)

    def bisection(self, a, b, tol=1e-7):
        return root_finding.bisection(self._eval_univariate, a, b, tol)

class MultiPoly(Polynomial):
    """A class to represent a multivariate polynomial. Inherits from the Polynomial class."""
    def __init__(self, coefficients: dict):
        """
        Initialize the MultiPoly class.

        Args:
        coefficients (dict): A dictionary of coefficients, where each key is a tuple of exponents
        corresponding to each variable in the polynomial, and the value is the coefficient.
        """
        super().__init__(coefficients)
        if not self.is_multivariate:
            raise ValueError("The provided coefficients don't represent a multivariate polynomial.")


