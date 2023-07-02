# A `Polynomial` Class in Python

A Python package that provides a user-friendly and flexible way to work with polynomials. The package is designed for both beginners looking to explore polynomial mathematics and advanced users needing to perform complex operations on univariate and multivariate polynomials. 

----

### Key Features

- **Creation of Polynomials**: Allows creating polynomials using a dictionary format for coefficients. Supports both univariate and multivariate polynomials.
- **Evaluation**: Evaluate a polynomial at a point $(x,y,z)$.
- **Operations on Polynomials**: Supports polynomial addition, subtraction, and polynomial and scalar multiplication.
- **Differentiation & Integration**: Ability to compute the derivative of a univariate polynomial and definite and indefinite integrals.
- **Root Finding**: Approximates roots numerically using [Newton-Raphson] method.
- **Plotting**: Plot a line in 2D space or a surface in 3D space.
- **Conversion to SymPy**: Allows conversion to a SymPy expression for more advanced operations.
----
**Multivariate Polynomials**: 
- `BiVarPoly` class;
**Partial differentiation**, finding of **stationary points**, classifying said points with a **Hessian matrix** and **surface plotting**.

- `TriVarPoly` class;
**Partial differentiation** up to the third degree.

----

### `decompose_polynomial` function

Takes a string representation of a polynomial as an argument and returns a dictionary. It works for both univariate and multivariate polynomials.

For univariate polynomials, the function returns a dictionary where the keys are the exponents of $x$ and the values are the corresponding coefficients.

For multivariate polynomials, the function returns a dictionary where the keys are tuples of exponents ($x$, $y$, $z$) and values are the corresponding coefficients.

----

### Quick Start Guide
```python
# Import
from py_files.Polynomial import Polynomial, UniPoly, BiVarPoly, TriVarPoly
from py_files.poly_dictionary import decompose_polynomial

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

# Create a multivariate Polynomial object in two variables:
bip_str = "x**2*y + -x*y**2 + 1"
bip_dict = decompose_polynomial(bip_str)
bipoly = BivarPoly(bip_dict)    

# Create a multivariate Polynomial object in three variables:
trip_str = "3*x**2*y*z + -x*y**2 + z + 1"
trip_dict = decompose_polynomial(trip_str)
tripoly = TriVarPoly(bip_dict)    

# Print the string representation of the polynomial:
print(p)
```
### Within the `testing` directory, more complex cases are explored. 

Each notebook begins by defining 6 polynomials. These can be redefined and session code used for the different instances.

----

#### `testing.ipynb`

Explores some of the basic functionality inherited by all classes within the module.

Uni-variate polynomials are explored in more depth.

----

#### `partial_testing.ipynb`

Partial differentiation is tested for polynomials $f(x,y)$ in class `BiVarPoly`.

Tested up to the third partial derivative for polynomials $f(x,y,z)$ in class `TriVarPoly`

----

#### `surface_testing.ipynb`

More advanced multi-variate calculus methods are tested for class `BiVarPoly`.

----

#### `symbolic_computation.ipynb`

A guide-book to the basics of the [SymPy](https://docs.sympy.org/latest/index.html) library. 
Transcribed with the aid of [Numerical Python](https://jrjohansson.github.io/numericalpython.html) by Robert Johannson.

----

### Personal Motivations

"Personally, I found the representation of polynomials as lists of coefficients, such as in NumPy's `numpy.polynomial.polynomial.Polynomial` and SciPy's `scipy.interpolate.BarycentricInterpolator`, to be non-intuitive, especially for sparse polynomials with many zero coefficients."

SymPy's `sympy.polys.polytools.Poly` class offers symbolic representation, which can be very powerful but may also be overkill for simpler use-cases. Moreover, when creating a polynomial from a list of coefficients in SymPy, the list represents the polynomial's coefficients in descending order, which can be confusing when compared to other libraries.

Therefore, I opted to use a dictionary data structure for my Polynomial class, where each key-value pair directly maps a polynomial's degree to its coefficient, offering a clear, intuitive, and flexible approach to polynomial representation.

In addition, this project was an opportunity for me to delve deeper into object-oriented programming, with the added challenge of parsing a string representation of a polynomial into a dictionary".

----

"As a newcomer to programming, I found it challenging to understand the underlying functionality of many classes and functions within the `SciPy` stack. My goal was to create a module that, while still a 'black box' in terms of usage (for example, when you call the plot method, a graph appears as if by magic!), has clear and detailed documentation to allow any curious users to understand the 'magic' behind the operations.

Having some experience with the Maxima Computer Algebra System, I wanted to bridge the gap between symbolic computation and Python. Therefore, I've integrated SymPy which allows users to perform complex symbolic computations while maintaining the ease and simplicity of Python.

I hope you find this module useful and intuitive. I welcome any feedback or suggestions for improvement - feel free to open an issue or pull request!"

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

### Installation

Clone this repository to your local machine, navigate to the repository directory, and then install the dependencies:

```bash
https://github.com/pineapple-bois/polynomial-methods.git
cd polynomial-methods
pip install -r requirements.txt
```
Then import the package into your Python project
```python
from polynomial_class import Polynomial, UniPoly, MultiPoly
from poly_dictionary import decompose_polynomial
```

----