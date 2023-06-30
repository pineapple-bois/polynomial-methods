import pytest
from poly_dictionary import decompose_polynomial

def test_decompose_polynomial():

    assert decompose_polynomial("5*x^8*y^3+3*x^5+3*y^7*z^3-4*z^2*x-x^2-y^2-z^2+z+x-y+27") == \
           {(8, 3, 0): 5.0, (5, 0, 0): 3.0, (0, 7, 3): 3.0, (1, 0, 2): -4.0, (2, 0, 0): -1.0,
            (0, 2, 0): -1.0, (0, 0, 2): -1.0, (0, 0, 1): 1.0, (1, 0, 0): 1.0, (0, 1, 0): -1.0,
            (0, 0, 0): 27.0}

    assert decompose_polynomial("3*x^2+2*x+1") == \
           {2: 3.0, 1: 2.0, 0: 1.0}

    assert decompose_polynomial("4*x^3*y^2*z+2*x^2*y*z-x*y*z+x-z-10") == \
           {(3, 2, 1): 4.0, (2, 1, 1): 2.0, (1, 1, 1): -1.0, (1, 0, 0): 1.0, (0, 0, 1): -1.0,
            (0, 0, 0): -10.0}

    with pytest.raises(ValueError):
        decompose_polynomial("5*x^8*r^3+3*x^5+3*y^7*z^3-4*z^2*x-x^2-y^2-z^2+z+x-y+27")

    with pytest.raises(ValueError):
        decompose_polynomial("5*a^8*b^3+3*a^5+3*b^7*c^3-4*c^2*a-a^2-b^2-c^2+c+a-b+27")
