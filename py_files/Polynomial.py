import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sympy
from IPython.display import display
from typing import Optional, Union
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
            coefficients = {(0, 0, 0): coefficients}

        self._coeff = {}
        for key, value in coefficients.items():
            if isinstance(key, int):  # If the key is an int (i.e., it's a uni-variate polynomial)
                key = (key, 0, 0)  # Convert it to a tuple and pad with 0s
            elif isinstance(key, tuple) and len(key) == 1:  # If the key is a tuple with 1 element
                key = (key[0], 0, 0)  # Pad with 0s
            self._coeff[key] = value

        self.expression = None
        self.num_variables = self._calculate_num_variables()
        self.is_multivariate = self.num_variables > 1

    def _calculate_num_variables(self):
        num_variables = 0
        num_positions = len(next(iter(self._coeff.keys())))
        for position in range(num_positions):
            variables_present = any(position < len(powers) and powers[position] != 0 for powers in self._coeff.keys())
            if variables_present:
                num_variables += 1
        return num_variables

    @property
    def coeff(self):
        return self._coeff

    @coeff.setter
    def coeff(self, value):
        self._coeff = value

    @classmethod
    def create(self, coefficients):
        """Create a new Polynomial instance based on the number of variables in the coefficients."""
        num_variables = self._calculate_num_variables(coefficients)
        if num_variables == 1:
            return UniPoly(coefficients)
        elif num_variables == 2:
            return BiVarPoly(coefficients)
        elif num_variables == 3:
            return TriVarPoly(coefficients)
        else:
            raise ValueError("Too many variables!")

    @staticmethod
    def count_variables(_coeff):
        max_variables = 0
        for powers in _coeff.keys():
            variables = sum(1 for power in powers if power != 0)
            if variables > max_variables:
                max_variables = variables
        return max_variables

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
            return self.__class__(coeff_product)
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
        if len(args) != self.num_variables:
            raise ValueError(f"Expected {self.num_variables} arguments, got {len(args)}")
        value = 0
        for powers, coeff in self.coeff.items():
            term_value = coeff
            for power, arg in zip(powers, args):
                term_value *= arg ** power
            value += term_value
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

            if self.num_variables == 1:  # Uni-variate polynomial
                # Build the polynomial
                if rational:
                    polynomial = sum(
                        sympy.Rational(str(coeff)) * x ** powers[0] for powers, coeff in self._coeff.items())
                else:
                    polynomial = sum(
                        coeff * x ** powers[0] for powers, coeff in self._coeff.items())
                # Create a sympy.Poly object
                self.expression = sympy.Poly(polynomial, x)

            elif self.num_variables == 2:  # Bi-variate polynomial
                # Build the polynomial
                if rational:
                    polynomial = sum(
                        sympy.Rational(coeff) * x ** powers[0] * y ** powers[1]
                        for powers, coeff in self._coeff.items())
                else:
                    polynomial = sum(
                        coeff * x ** powers[0] * y ** powers[1]
                        for powers, coeff in self._coeff.items())
                # Create a sympy.Poly object
                self.expression = sympy.Poly(polynomial, x, y)

            elif self.num_variables == 3:  # Tri-variate polynomial
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

    def display(self):
        """
        Displays the sympy expression of the polynomial using IPython's display system.

        The method does not take any parameters and does not return any value.
        The method calls the '_convert_to_sympy_expression' function internally.
        Converts the polynomial to a sympy expression.
        """
        self._convert_to_sympy_expression()
        display(self.expression)  # This will use Jupyters display system.

    def save_as_sympy(self, rational=False, return_expr=False):
        """
        Saves the polynomial as a sympy expression and returns it.

        Parameters:
        rational (bool): If True, the coefficients are converted into sympy Rational type. Default is False.
        return_expr (bool): If True, returns a sympy expression instead of a sympy.Poly object. Default is False.

        Returns:
        sympy.core.expr.Expr or sympy.Poly : A sympy expression or a sympy.Poly object representing the polynomial.

        The method calls the '_convert_to_sympy_expression' function.
        Converts the polynomial to a sympy expression.
        """
        self._convert_to_sympy_expression(rational, force=True)
        if return_expr:
            return self.expression.as_expr()  # Return as a sympy expression
        else:
            return self.expression  # Return as a sympy.Poly object


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
        self.is_bivariate = False
        self.is_trivariate = False
        self.is_multivariate = False
        if self.num_variables != 1:
            raise ValueError("The provided expression does not represent a uni-variate polynomial.")

    def plot(self, x_min=-10, x_max=10, num_points=1000):
        """Plot the polynomial over a specified range.

        Args:
        x_min (float, optional): The minimum x-value for the plot. Defaults to -10.
        x_max (float, optional): The maximum x-value for the plot. Defaults to 10.
        num_points (int, optional): The number of points to generate for the plot. Defaults to 1000.
        """
        x = np.linspace(x_min, x_max, num_points)
        y = [self.__call__(x_val) for x_val in x]

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
        return root_finding.newton(self.__call__, self.differentiate().__call__, x0, max_it, tol)

    def bisection(self, a, b, tol=1e-7):
        return root_finding.bisection(self.__call__, a, b, tol)



class BiVarPoly(Polynomial):
    """A class to represent a bi-variate polynomial. Inherits from the Polynomial class."""
    def __init__(self, coefficients: dict):
        super().__init__(coefficients)
        if self.num_variables != 2:
            self.is_bivariate = False
        else:
            self.is_bivariate = True

    def validate_bivariate(self):
        if self.num_variables != 2:
            raise ValueError("The provided expression does not represent a bi-variate polynomial.")

    def partial_derivative(self, variable, second_variable=None):
        """
        Compute the partial derivative of the polynomial.

        Args:
            variable (str): The variable to differentiate with respect to ('x' or 'y').
            second_variable (str, optional): The second variable for the second partial derivative.
                Defaults to None.

        Returns:
            Union[Polynomial, 'BiVarPoly']: The partial derivative polynomial.

        Raises:
            ValueError: If invalid variables are specified.
        """
        if variable not in ['x', 'y']:
            raise ValueError("Invalid variable specified.")

        if second_variable is not None and second_variable not in ['x', 'y']:
            raise ValueError("Invalid second variable specified.")

        new_coeff = {}
        for powers, coeff in self._coeff.items():
            if variable == 'x':
                if powers[0] > 0:
                    new_powers = ((powers[0] - 1), powers[1], 0)
                    new_coeff[new_powers] = coeff * powers[0]
            elif variable == 'y':
                if powers[1] > 0:
                    new_powers = (powers[0], (powers[1] - 1), 0)
                    new_coeff[new_powers] = coeff * powers[1]

        if not new_coeff:
            # If no terms are present after differentiation, return a constant (0)
            return BiVarPoly({(0, 0, 0): 0.0})

        # Determine the appropriate class for the derived polynomial
        if self.count_variables(new_coeff) > 1:
            derived_poly = BiVarPoly(new_coeff)
        else:
            derived_poly = BiVarPoly(new_coeff)  # Keep it as BiVarPoly, even if it becomes univariate

        if second_variable is not None:
            derived_poly = derived_poly.partial_derivative(second_variable)

        return derived_poly

    def plot(self, x_range: tuple, y_range: tuple, plot_type: str = 'surface'):
        """
        This method plots the polynomial as a surface, a wireframe, or a contour plot in a 3D space.

        Args:
            x_range (tuple): The range of x-values for the plot. Given as (start, end).
            y_range (tuple): The range of y-values for the plot. Given as (start, end).
            plot_type (str, optional): Defaults to 'surface'.
            The type of plot to create. Options are 'surface', 'wireframe', or 'contour'.

        Raises:
            ValueError: If an invalid plot type is specified.
        """
        # Create a new figure and add a 3D subplot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Generate x and y values in the specified range
        x = np.linspace(x_range[0], x_range[1], num=100)
        y = np.linspace(y_range[0], y_range[1], num=100)

        # Create a meshgrid from x and y arrays
        X, Y = np.meshgrid(x, y)

        # Evaluate the polynomial at each point in the grid to get z values
        Z = np.array([self.__call__(X[i, j], Y[i, j]) for i in range(X.shape[0])
                      for j in range(X.shape[1])]).reshape(X.shape)

        # Normalize the Z array
        norm = mpl.colors.Normalize(-abs(Z).max(), abs(Z).max())

        # Based on the specified plot type, create the appropriate plot
        if plot_type == 'surface':
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0,
                            antialiased=False, norm=norm, cmap=mpl.cm.RdBu)
        elif plot_type == 'wireframe':
            ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color="darkgrey")
        elif plot_type == 'contour':
            ax.contour(X, Y, Z, zdir='z', offset=0, norm=norm, cmap=mpl.cm.cool)
            ax.contour(X, Y, Z, zdir='y', offset=3, norm=norm, cmap=mpl.cm.cool)
        else:
            raise ValueError(f"Invalid plot type specified: {plot_type}")

        # Format the polynomial equation for the title
        formula = self.__str__()
        formula = formula.replace('*', '')

        # Set the title and labels for the axes
        ax.set_title(f"{plot_type.capitalize()} Plot\n${formula}$")
        ax.set_xlabel("$x$", fontsize=16)
        ax.set_ylabel("$y$", fontsize=16)
        ax.set_zlabel("$z$", fontsize=16)

        # Display the plot
        plt.show()

    def find_stationary_points(self, rational=False):
        x, y = sympy.symbols('x y')

        if rational:
            poly_sympy = self.save_as_sympy(rational=True).as_expr()
        else:
            poly_sympy = self.save_as_sympy().as_expr()

        # Compute partial derivatives
        fx = poly_sympy.diff(x)
        fy = poly_sympy.diff(y)

        # Solve for stationary points
        stationary_points = sympy.solve([fx, fy], (x, y))

        # Check if the stationary points are a dict and convert to a list of tuples
        if isinstance(stationary_points, dict):
            stationary_points = [tuple(stationary_points.values())]
        elif isinstance(stationary_points, tuple):  # If the solution is a single tuple, convert it to a list
            stationary_points = [stationary_points]
        return stationary_points

    def Hessian(self, point=None, as_rational=False):
        """Create the Hessian matrix symbolically."""
        x, y = sympy.symbols('x y')

        # Calculate the second order partial derivatives
        fxx = str(self.partial_derivative('x', 'x'))
        fxy = str(self.partial_derivative('x', 'y'))
        fyy = str(self.partial_derivative('y', 'y'))

        # Create sympy expressions from the strings, and convert the coefficients to rationals if as_rational is True
        fxx_expr = sympy.sympify(fxx, rational=as_rational)
        fxy_expr = sympy.sympify(fxy, rational=as_rational)
        fyy_expr = sympy.sympify(fyy, rational=as_rational)

        # Construct Hessian matrix
        H = sympy.Matrix([[fxx_expr, fxy_expr],
                          [fxy_expr, fyy_expr]])

        return H

    def classify_points(self, points: list, as_rational=False):
        """Classify the stationary points."""
        if points is None:
            points = self.find_stationary_points(rational=as_rational)
        if points is not None and not isinstance(points, list):
            raise ValueError("Points must be provided as a list of tuples.")

        H = self.Hessian(as_rational=as_rational)
        results = {}

        # Check if the Hessian matrix is constant (has no variables)
        if len(H.free_symbols) == 0:
            # Apply the determinant test and classify the single point
            if len(points) != 1:
                raise ValueError("Constant Hessian can only have one stationary point.")
            point = points[0]
            D = H.det()
            classification = self._classify_point(D, H[0, 0])
            results[point] = classification
            print(f"The Hessian matrix is constant and its determinant is {D}.")
            print(f"The point {point} is a {classification} by the determinant test.")
            display(H)  # Display the constant Hessian matrix
            return results

        for point in points:
            # Substitute the point into the Hessian matrix and evaluate it
            H_evaluated = H.subs({'x': point[0], 'y': point[1]})

            # Calculate the determinant
            D = H_evaluated.det()

            # Classify the point
            classification = self._classify_point(D, H_evaluated[0, 0])
            results[point] = classification
            print("-----------------")
            print(f"After substituting the point {point} into the Hessian matrix:")
            display(H_evaluated)  # Display the evaluated Hessian matrix
            print(f"The point {point} is a {classification} by the determinant test.")

        return results

    def _classify_point(self, D, fxx):
        """Classify the point based on the determinant and fxx."""
        if D > 0 and fxx > 0:
            return "Local minimum"
        elif D > 0 and fxx < 0:
            return "Local maximum"
        elif D < 0:
            return "Saddle point"
        else:
            return "Test inconclusive"


class TriVarPoly(Polynomial):
    """A class to represent a tri-variate polynomial. Inherits from the Polynomial class."""
    def __init__(self, coefficients: dict):
        super().__init__(coefficients)
        self.is_trivariate = True
        self.is_multivariate = True

    def validate_trivariate(self):
        if self.num_variables != 3:
            raise ValueError("The provided expression does not represent a tri-variate polynomial.")

    def partial_derivative(self, variable: str, second_variable: Optional[str] = None,
                           third_variable: Optional[str] = None,
                           print_progress: bool = True) -> Union[Polynomial, 'TriVarPoly']:
        """
        Compute the partial derivative of the polynomial.

        Args:
            variable (str): The variable to differentiate with respect to ('x', 'y', or 'z').
            second_variable (str, optional): The second variable for the second partial derivative.
                Defaults to None.
            third_variable (str, optional): The third variable for the third partial derivative.
                Defaults to None.
            print_progress (bool, optional): Whether to print the differentiation progress. Defaults to True.

        Returns:
            Union[Polynomial, 'TriVarPoly']: The partial derivative polynomial.

        Raises:
            ValueError: If invalid variables are specified.

        Note:
            The returned partial derivative can be either a Polynomial or a TriVarPoly instance,
            depending on the number of variables present in the resulting polynomial.
        """
        if variable not in ['x', 'y', 'z']:
            raise ValueError("Invalid variable specified.")

        if second_variable is not None and second_variable not in ['x', 'y', 'z']:
            raise ValueError("Invalid second variable specified.")

        if third_variable is not None and third_variable not in ['x', 'y', 'z']:
            raise ValueError("Invalid third variable specified.")

        new_coeff = {}
        for powers, coeff in self._coeff.items():
            if variable == 'x':
                if powers[0] > 0:
                    new_powers = ((powers[0] - 1), powers[1], powers[2])
                    new_coeff[new_powers] = coeff * powers[0]
            elif variable == 'y':
                if powers[1] > 0:
                    new_powers = (powers[0], (powers[1] - 1), powers[2])
                    new_coeff[new_powers] = coeff * powers[1]
            elif variable == 'z':
                if powers[2] > 0:
                    new_powers = (powers[0], powers[1], (powers[2] - 1))
                    new_coeff[new_powers] = coeff * powers[2]

        if print_progress:
            print(f"Partial derivative with respect to {variable}:")
            print(f"{Polynomial(new_coeff)}")

        if not new_coeff:
            # If no terms are present after differentiation, return a constant (0)
            return Polynomial({(0, 0, 0): 0.0})

        derived_poly = TriVarPoly(new_coeff)

        if second_variable is not None:
            if third_variable is not None:
                derived_poly = derived_poly.partial_derivative(second_variable, third_variable, print_progress)
            else:
                derived_poly = derived_poly.partial_derivative(second_variable, print_progress=print_progress)

        if third_variable is not None:
            derived_poly = derived_poly.partial_derivative(third_variable, print_progress)

        return derived_poly
    # Define additional methods for tri-variate polynomials...



