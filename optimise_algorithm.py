from math import floor, sqrt
from sympy.abc import x, i
from sympy import solveset, binomial, summation, S, Symbol

x = Symbol('x', positive=True)
y = Symbol('y', positive=True)

def get_C(ep, n):
    """
    For the given ep and n, solves the following equation for C, assuming C>1:
        ep = 1/(1+C)^n \sum_{i=0}^{floor(n/2)} \binom(n,i) * C^i
    """
    exp = (1/(1+x))**n * summation(binomial(n, i) * (x**i) , (i, 0, floor(n/2))) - ep
    if not x in exp.free_symbols: # For n=1, the expression is already solved.
        return [exp]
    for C in  solveset(exp, x, domain=S.Reals):
        if C > 1:
            return C
    return None

def get_optimal_C_and_n(ep, R=0, W=0):
    """
    Given a maximum error probaiblity epsilon, computes the optimal setting of
    the constants C and n for Belovs' detection algorithm, assuming we have set
    a=\sqrt{b} and b=1/(2+2C)^2.

    Note that this optimisation is independent of R and W. However, if R and W
    are set to positive integers (and the debug print statement are
    turned on) then the number of queries actually made by the optimal algorithm
    given an isntance with R and W are printed.
    """
    sqrt_RW_factors = []
    Cs = []
    i = 0
    while i < 2 or sqrt_RW_factors[-1] < sqrt_RW_factors[-2]:
        n = 2 * i + 1
        Cs.append(get_C(ep, n))
        sqrt_RW_factors.append(n * sqrt(Cs[-1]*(1+Cs[-1]**2)))
        i += 1
    return round(Cs[-2], 8), n - 2, round(sqrt_RW_factors[-2], 8)
