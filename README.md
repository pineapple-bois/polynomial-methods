# `Polynomial` Class in Python

The Polynomial class is a Python implementation designed to represent both univariate and multivariate polynomials and provide various operations on them. This class allows you to easily create, evaluate, manipulate, approximate roots, and plot polynomials.

----

### Key Features

- **Creation of Polynomials**: Allows creating polynomials using a dictionary format for coefficients. Supports both univariate and multivariate polynomials.
- **Evaluation**: Evaluate a polynomial for given values of variables.
- **Operations on Polynomials**: Supports polynomial addition, subtraction, and polynomial and scalar multiplication.
- **Differentiation & Integration**: Ability to compute the derivative of a univariate polynomial and definite and indefinite integrals.
- **Root Finding**: Approximates roots numerically using Newton-Raphson and Bisection methods.
- **Plotting**: Capability to plot univariate polynomials.
- **Conversion to SymPy**: Allows converting the polynomial to a SymPy expression for more advanced operations.
- **Multivariate Polynomials**: (Methods to be added soon)

----

## `decompose_polynomial` function

Takes a string representation of a polynomial as an argument and returns a dictionary. It works for both univariate and multivariate polynomials.

For univariate polynomials, the function returns a dictionary where the keys are the exponents of 'x' and the values are the corresponding coefficients.

For multivariate polynomials, it returns a dictionary where the keys are tuples of exponents (for 'x', 'y', and 'z') and values are the corresponding coefficients.

----

### Quick Start Guide
```python
# Import
from polynomial_class import Polynomial, UniPoly, MultiPoly, 
from poly_dictionary import decompose_polynomial

# Create a uni-variate Polynomial object from string:
p_str = "2*x**3 - x**2 + 1"
p_dict = decompose_polynomial(p_str)
p = UniPoly(p_dict)

# Evaluate the polynomial for x = 2:
result = p(2)

# Add two polynomials:
p1_str = "3*x**2 + x"
p1_dict = decompose_polynomial(p1_str)
p1 = UniPoly(p1_dict)

p2_str = "2*x**3 - x**2 + 1"
p2_dict = decompose_polynomial(p2_str)
p2 = UniPoly(p2_dict)

# Addition of two polynomials:
p3 = p1 + p2

# Subtract two polynomials:
p4 = p1 - p2

# Multiply two polynomials:
p5 = p1 * p2

# Multiplication by a scalar
five_p2 = 5 * p2

# Compute the derivative:
dp = p.differentiate()

# Compute the indefinite integral:
ip = p.integrate()

# Compute the definite integral between two limits:
integral = p.integrate_definite(0, 2)

# Approximate the root using Newton's method:
root = p.newton(1)

# Plot the polynomial:
p.plot()

# Conversion to SymPy object
## option to create with rational coefficients
sym_p = p.save_as_sympy(rational=True)

# Create a multivariate Polynomial object:
mp_str = "x**2*y + -x*y**2 + 1"
mp_dict = decompose_polynomial(mp_str)
mp = MultiPoly(mp_dict)    # Functionality to be added soon

# Print the string representation of the polynomial:
print(p)
```

----

### `symbolic_computation.ipynb`

A guide-book to the basics of the [SymPy](https://docs.sympy.org/latest/index.html) library. 
Transcribed with the aid of [Numerical Python](https://jrjohansson.github.io/numericalpython.html) by Robert Johannson.

----

### Dependencies

This project depends on the following Python libraries:

`numpy`
`matplotlib`
`sympy`
`IPython`

Ensure you have them installed in your environment. If not, you can install the required dependencies from the requirements.txt file:

```bash
pip install -r requirements.txt
```

----

#### Future Enhancements

We're working to include methods and operations on multivariate polynomials, including partial differentiation, surface-plotting, and stationary point finding and analysis.

----