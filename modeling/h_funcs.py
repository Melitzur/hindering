'''
Notations following Stats paper:  Stats 5, no. 1: 111-127
https://doi.org/10.3390/stats5010008


The hindering functions h(x) are defined implicitely by
(eq. 10 in Stats paper)

        x_from_h(h) = x

where

    x_from_h = ln h + sum (w_k/k)(h**k - 1)   (1)

and where

    sum w_k = 1                               (2)

for a hindering series with w_k the expansion coefficients.
All powers k must be integers >= 1.

The input coefficitns are uncostrained. The normalization
constraint in eq. (2) is imposed internally during the
calculation itself. Denote by a_k the dict of input
coefficients, then the definition can be written as

    F(h) = x_from_h(h, a_k) - x = 0

The solution is done with Newton's method by varying h
with constant x and a_k. The solution tool is Newt() from
utils. Common handling of both single-value x and a
vector of x-values is feasible thanks to numpy.atleast_1d:

https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_1d.html

which converts single-value x to numpy array.

Initial guess is simply exp(x) when x <= 0. For a vector
of monotonically increasing x, each solution is the initial
guess for the next element. When all elements are > 0,
the vector is extended downward with xtend(v) to include
0 as its first element.

Contents:
---------
h
    func
_x_from_h
xtend
h_test
f_hind
x_from_h
ell
sth

'''
#####################################################

import numpy as np
from numpy import exp, log as ln
from modeling.utils import Newt, colprint, headline


def h(X, a_k,
      acc=1.e-4, max_iter=50,
      debug=False, test=False):
    '''
    Universal hindering function with optional testing
    of the results and debug to follow the numerics
    progress of the solution

    X   --- The independent variable; single number or
            monotonically increasing vector
    a_k --- a dictionary of power series expansion coefficients

    returns h(x)
    '''
    def func(h):
        xh, d = _x_from_h(h, a_k)
        return xh - x, d

    X = np.atleast_1d(X).astype(float)
    xtended = xtend(X)
    h = np.empty_like(xtended)
    apr = exp(xtended[0]) #initial guess
    for i, x in enumerate(xtended):
        if debug: print(f'From h: {x = }')
        h[i] = Newt(func, apr,
                    acc=acc, max_iter=max_iter,
                    debug=debug
                   )
        # initial guess for next x; see notebook
        # hinder_problem.ipynb for x < 0 problem
        apr = (h[i] if x > 0 else
               exp(x)
              )

    h = h[-len(X):]  #retain only solution for the original x-vector
    if test: h_test(X, h, a_k, acc)

    return h if len(X) > 1 else h[0]


def _x_from_h(h, a_k, derivative=True):
    '''
    Hindering series; eq(1)
    a_k is the dictionary of expansion coefficients
    returns x_from_h and (optionally) its derivative
    Note: derivative is needed only for internal use
    so this is a private function
    '''
    xh = ln(h)
    if derivative:
        d = 1./h     #derivative of xh
    w = sum(a_k.values())
    for k, a in a_k.items():
        wk = a/w     #normalize the weights (eq 2 above)
        xh += (wk/k)*(h**k - 1)
        if derivative:
            d += wk*h**(k - 1)

    if derivative:
        return xh, d
    else:
        return xh


def xtend(v):
    '''Extend v downward to include 0 whenever v[0] > 0
    '''
    if v[0] <= 0: return v

    n = int(v[0]) + 1
    xtension = np.linspace(0, v[0], n)
    return np.concatenate((xtension[:-1], v))


def h_test(X, h, a_k, acc):
    ''' Test whether solution h obeys x_from_h(h) = x
    Called from function h() when test = True
    '''
    print(f'\n***Testing the hindering h(x) with {a_k = }')

    Xh = np.ones_like(X)
    for i, h_ in enumerate(h):
        Xh[i] = _x_from_h(h_, a_k,
                          derivative=False
                         )

    if np.any(abs(X - Xh) > acc):
        print('>>>>> WARNING: Test failed! <<<<<'.center(32))
    else:
        headline('Test passed', width=32)
    colprint(
              X,    h,   X - Xh + 1.,
    headers=('x', 'h(x)', 'test'   ),
             d=3
            )


################ Utilities for external use #################

def f_hind(x, a_k, derivative=False):
    '''Hindering factor f and, optionally, its derivative

    See sum in denominator of eq. 11, Stats paper
    '''
    f = 0.
    if derivative: d = 0.
    for k, a in a_k.items():
        f += a*x**k
        if derivative:
            d += k*a*x**(k-1)

    if derivative:
        return f, d
    else:
        return f


def x_from_h(h, a_k):
    '''Stand-alone calculation of xh only
    '''
    if not h > 0.:
        s = f'Bad call to x_from_h(h): {h = }'
        s+= 'h must be positive'
        print(s)
        return
    return _x_from_h(h, a_k,
                     derivative=False)


#### Methematical generators of
#### 3-parameter hindering functions
#### normalized to unity at x = 0

def ell(x):
    ''' logistic symmetric about x = 0

    Eq 15 in Stats paper
    '''
    return 2./(1 + exp(-x))


def sth(x, k, debug=False):
    '''kth-order single-term hindering function
       optional debug triggers printing
       from Newton when True

       Eq 12 in Stats paper
    '''
    return h(x, {k: 1}, debug=debug)

###########################################
if __name__ == '__main__':
    import sys

    X = range(1, 11)

    k_list = [1, 2, 5]
    a_list = [.5, .4, .1]
    a = dict(zip(k_list, a_list))
    h(X, a, test=True)

    #sys.exit()

    ak = {7: 1}
    h(X, ak, test=True)

    sys.exit()


###########################################
