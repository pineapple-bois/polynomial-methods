import re


def decompose_polynomial(formula: str):
    """
    The input formula should be a string representation of the polynomial.

    Polynomial exponents are natural numbers.
    If exponents are rationals or negative integers, the formula will not parse

    :return Polynomial dictionary;
    Uni-variable: {exponent: coefficient}
    Multivariable: {(exponent_x, exponent_y, exponent_z): coefficient}
    """

    if not isinstance(formula, str):
        raise TypeError("Input must be a string representation of a polynomial.")

    formula = formula.replace(' ', '').replace('^', '**')
    # Extract all variables in formula
    variables = set(re.findall('[a-z]', formula))

    # Check for uni-variate and multivariate cases
    if len(variables) > 1:
        multivariate = True    # More than one variable, so it's multivariate
        # Check that the variables are only x, y, and z
        if variables.difference({'x', 'y', 'z'}):
            raise ValueError("For multivariate polynomials, variables can only be 'x', 'y', or 'z'")

    elif len(variables) == 1:
        multivariate = False   # One variable, so it's uni-variate
        # Check that the variable is x
        if variables.pop() != 'x':
            raise ValueError("For uni-variate polynomials, variable must be 'x'")

    else:
        # No variables, raise an error
        raise ValueError("A polynomial must contain at least one variable")

    if multivariate:
        terms = re.findall(
            r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\*[a-z](?:\*\*\d+)?)*(?:\*[a-z](?:\*\*\d+)?)*|[+-]?'
            r'(?:[a-z](?:\*\*\d+)?)*(?:\*[a-z](?:\*\*\d+)?)*|[+-]?[a-z]', formula)
        r"""
        [+-]?: Matches an optional "+" or "-" sign. r indicates raw string
        
        1. '(?:\d+(?:\.\d*)?|\.\d+): Matches a number that can be an integer, a decimal, or a fraction. 
        \d+ matches one or more digits, \.\d* matches a decimal point followed by zero or more digits, 
        and | acts as a logical OR operator.
        
        2. '(?:\*[a-z](?:\*\*\d+)?): Matches a term of the form '*x^n', where 'n' is an integer. 
        * is a multiplication symbol, [a-z] matches any lowercase letter (representing a variable), 
        ** represents power operation, and \d+ matches the exponent which is one or more digits.
        *: Matches zero or more of the preceding group. It makes sure to match all terms in the form '*x^n'.
        |: Acts as a logical OR operator in regex.
        
        3. '[+-]?(?:[a-z](?:\*\*\d+)?)*: Matches a term of the form 'x^n' or '+x^n' or '-x^n' where 'n' is an integer. 
        The group can repeat zero or more times to match all such terms.
        [+-]?[a-z]: Matches a variable 'x' with an optional '+' or '-' sign in front of it.
        
        """
        terms = [term for term in terms if term]    # remove any empty strings
        terms = [term.replace('**', '^') for term in terms]     # analysis is simpler with only one iteration of '*'
        # Constructing the dictionary
        poly_dict = {}
        for term in terms:
            if term.startswith('+'):
                term = term[1:]

            if '*' in term:
                parts = term.split('*')

                # Extract and handle the coefficient if it exists
                if parts[0].replace('.', '').isdigit() or parts[0].replace('-', '').replace('.', '').isdigit():
                    coefficient = float(parts.pop(0))
                else:
                    coefficient = -1.0 if parts[0].startswith('-') else 1.0
                    if parts[0].startswith('-'):
                        parts[0] = parts[0][1:]

                # Initialize list of exponents
                exponents = [0, 0, 0]

                for part in parts:
                    # Each part is either of the form 'x', 'x^2', etc.
                    if '^' in part:
                        variable, exponent = part.split('^')
                        exponent = int(exponent)
                    else:
                        variable = part
                        exponent = 1

                    if variable == 'x':
                        exponents[0] = exponent
                    elif variable == 'y':
                        exponents[1] = exponent
                    elif variable == 'z':
                        exponents[2] = exponent

                # The key in the dictionary is a tuple of exponents
                tuple_key = tuple(exponents)
                poly_dict[tuple_key] = coefficient

            elif '^' in term:  # Handle terms like '+x^2', '+y^2', etc.
                parts = term.split('^')
                variable = parts[0]
                if variable.startswith('-'):
                    variable = variable[1:]
                    coefficient = -1.0
                else:
                    coefficient = 1.0
                exponent = int(parts[1])

                exponents = [0, 0, 0]

                if variable == 'x':
                    exponents[0] = exponent
                elif variable == 'y':
                    exponents[1] = exponent
                elif variable == 'z':
                    exponents[2] = exponent

                tuple_key = tuple(exponents)
                poly_dict[tuple_key] = coefficient

            elif term.isalpha() or (len(term) > 1 and term[1:].isalpha()):
                # statement for terms like '+x', '-y' etc ... no explicit coefficient
                variable = term[-1]
                coefficient = 1.0 if term[0] != '-' else -1.0
                exponents = [0, 0, 0]

                if variable == 'x':
                        exponents[0] = 1
                elif variable == 'y':
                        exponents[1] = 1
                elif variable == 'z':
                        exponents[2] = 1

                tuple_key = tuple(exponents)
                if tuple_key not in poly_dict:
                    poly_dict[tuple_key] = coefficient

            else:
                coefficient = float(term)
                poly_dict[(0, 0, 0)] = coefficient

        return poly_dict

    # uni-variate case
    else:
        terms = re.findall(r'([+\-]?\s*(?:\d*\.\d+|\d+\.\d*|\.\d+|\d+(?:\.\d*)?)\*?x\*\*\d+|'
                           r'[-]?\s*(?:\d*\.\d+|\d+\.\d*|\.\d+|\d+(?:\.\d*)?)\*?x?|'
                           r'[+\-]?\s*x(?:\*\*\d+)?)', formula)
        r"""
        The regex pattern consists of three main parts separated by '|', which acts as a logical OR operator.

        1. '([+\-]?\s*(?:\d*\.\d+|\d+\.\d*|\.\d+|\d+(?:\.\d*)?)\*?x\*\*\d+':
        Matches a term with a coefficient and a power, like '+3.5*x**2' or '-x**2'.
        '[+\-]?' matches an optional '+' or '-' sign at the start of the term.
        '\s*' matches any number of whitespace characters.
        '(?:\d*\.\d+|\d+\.\d*|\.\d+|\d+(?:\.\d*)?)' matches any number, integer, decimal, or fraction.
        '\*?x\*\*\d+' matches '*x**n', where '*' is a multiplication symbol, 'x' is the variable, 
        '**' is the power operator, and 'n' is one or more digits (the power).

        2. '[-]?\s*(?:\d*\.\d+|\d+\.\d*|\.\d+|\d+(?:\.\d*)?)\*?x?':
        Matches a term with a possible coefficient and no power or a standalone number, 
        like '-3.5*x' or '+x' or '-3.5' or '3'.
        '[-]?' matches an optional '-' sign at the start of the term.
        '\s*' matches any number of whitespace characters.
        '(?:\d*\.\d+|\d+\.\d*|\.\d+|\d+(?:\.\d*)?)' matches any number, integer, decimal, or fraction.
        '\*?x?' matches an optional '*x', where '*' is a multiplication symbol and 'x' is the variable.

        3. '[+\-]?\s*x(?:\*\*\d+)?':
        Matches a term with a variable and a possible power, like '+x**2' or '-x' or 'x'.
        '[+\-]?' matches an optional '+' or '-' sign at the start of the term.
        '\s*' matches any number of whitespace characters.
        'x(?:\*\*\d+)?' matches 'x**n' or 'x', where 'x' is the variable, '**' is the power operator,
         and 'n' is one or more digits (the power).
         
        """
        terms = [term for term in terms if term]  # remove any empty strings
        powers = []
        coefficients = []
        highest_power = 0

        for term in terms:
            if 'x' in term:
                if '*x**' in term:
                    coefficient, power = term.split('*x**')
                elif 'x**' in term:
                    coefficient, power = term.split('x**')
                    if '+' in term or not coefficient:
                        coefficient = '1.0'
                    elif '-' in term or not coefficient:
                        coefficient = '-1.0'
                else:
                    coefficient, _ = term.split('*x') if term.endswith('*x') else term.split('x')
                    if not coefficient or coefficient == '+':
                        coefficient = '1.0'
                    elif coefficient == '-':
                        coefficient = '-1.0'

                coefficients.append(float(coefficient))
                power = int(power) if '**' in term else 1
                powers.append(power)
                highest_power = max(highest_power, power)

            else:
                coefficients.append(float(term))
                powers.append(0)

        poly_dict = dict(zip(powers, coefficients))
        return poly_dict
