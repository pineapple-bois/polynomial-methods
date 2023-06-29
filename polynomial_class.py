import numpy as np
import matplotlib.pyplot as plt
import root_finding


class Polynomial:
    def __init__(self, coefficients: dict):
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

        # Check if it's a multivariate polynomial
        self.is_multivariate = any(exponent[1] != 0 or exponent[2] != 0 for exponent in self._coeff.keys())

    @property
    def coeff(self):
        return self._coeff

    @coeff.setter
    def coeff(self, value):
        self._coeff = value

    def __repr__(self):
        """" return code for regenerating this instance"""
        return f"Polynomial({self.coeff})"

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
        # Implement multiplication of coefficients
        pass

    def __str__(self):
        terms = []
        for powers, coeff in sorted(self._coeff.items(), key=lambda x: x[0][0], reverse=True):
            if all(power == 0 for power in powers):
                term = str(coeff)
            else:
                # create variables strings
                vars_str = ['x', 'y', 'z']
                vars_with_power = [f"{vars_str[i]}^{power}" if power > 1 else f"{vars_str[i]}" for i, power in
                                   enumerate(powers) if power != 0]
                term = (
                    "-{}".format(' * '.join(vars_with_power)) if coeff == -1.0 else
                    "{}".format(' * '.join(vars_with_power)) if coeff == 1.0 else
                    f"{round(coeff, 4) if '.' in str(coeff) and len(str(coeff).split('.')[1]) > 4 else coeff}"
                    f" * {' * '.join(vars_with_power)}"
                )
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


class UniPoly(Polynomial):
    def __init__(self, coefficients: dict):
        super().__init__(coefficients)
        if self.is_multivariate:
            raise ValueError("The provided coefficients don't represent a uni-variate polynomial.")

    def plot(self, x_min=-10, x_max=10, num_points=1000):
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

    def newton(self, x0, max_it=20, tol=1e-3):
        return root_finding.newton(self._eval_univariate, self.differentiate()._eval_univariate, x0, max_it, tol)


class MultiPoly(Polynomial):
    def __init__(self, coefficients: dict):
        super().__init__(coefficients)
        if not self.is_multivariate:
            raise ValueError("The provided coefficients don't represent a multivariate polynomial.")


