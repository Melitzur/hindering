'''
Contents:
---------
headline
header
colprint
zillion
readable
num2readable
fitter
Newt
add_top_axis
get_ufit
the_unit
tag4

'''


######################################################################

import numpy as np
from numpy import log10 as log

import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator)

########################################



def headline(text, border="♦", width=72, space=False):   #chr(9830) = "♦"
    hdl = f" {text} ".center(width, border)
    if space: hdl = '\n' + hdl
    print(hdl)

def header(hdr, w=72):
    """ Print header line(s), width w
    """
    print(w*'_')
    if '\n' in hdr:
        print()
        [print(line.center(w)) for line in hdr.split('\n')]
    else:
        headline(hdr, width=w, space=True)
    print(w*'_'+'\n')

def colprint(*cols, headers=None, d=2):
    """Tabular printing of columns of numbers
       All numbers printed with exponential notation

    Input:
       cols  ---  Some equal-length 1D arrays of numbers

    Optional:
       headers -- Sequence of column headers
                  Default headers are ['col0', 'col1', ...]
       d:    ---  number of digits printed after decimal
    """
    w = 8 + d # column width
    if headers is None:
        headers = [f'col{i}' for i in range(len(cols))]
    else:
        #ensure headers fit into column width
        #headers = [h if len(h) <= w else h[:w] for h in headers]
        headers = [h[:w] for h in headers]
    print(''.join(f'{h:^{w}}' for h in headers))
    for row in zip(*cols):
        print(''.join(f"{f'{r:.{d}E}':^{w}}" for r in row))


def zillion(x):
    """ Get the proper name (million, billion, etc) for x
    """
    names = ['thousand', 'million', 'billion', 'trillion',
             'quadrillion', 'quintillion', 'sextillion']
    n3 = int(log(x)/3.)
    suffix = names[min(n3, len(names)) - 1] #sextillion takes care of > E21
    if x <= 1000: suffix = None
    return suffix, n3

def readable(x, d=2):
    """Convert thousands to proper name
       for example, readable(4.5E6) is 4.5 million
    """
    suffix, n3 = zillion(x)
    d3 = x/(10.**(3*n3))
    num = str(int(d3)) if d == 0 else f'{d3:.{d}f}'
    return f'{num} {suffix}'

def num2readable(x):
    """Thousand separator for numbers smaller than million,
    proper name for numbers larger than million
    Format Specifier for Thousands Separator:
    https://www.python.org/dev/peps/pep-0378/
    https://docs.python.org/3/library/string.html#formatspec
    """
    return f'{x:,.0f}' if x < 1.E6 else readable(x)


from scipy.optimize import curve_fit

def fitter(func, x_data, data,
           x_model=None, p0=None, sigma=None,
           bounds=(-np.inf, np.inf), scale=False):
    """Best-fit for:  data = func(x_data)   using curve_fit
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    When successful, calculates the best-fit model over range x_model,
    which can be different from x_data

    Optional args:
    p0: initial guesses for the parameters
    sigma: None or M-length sequence or MxM array, optional
         Determines the uncertainty in ydata. If we define residuals
         as r = ydata - f(x_data, *popt), then the interpretation of
         sigma depends on its number of dimensions:
         A 1-d sigma should contain values of standard deviations
         of errors in ydata. In this case, the optimized function
         is chisq = sum((r / sigma) ** 2).
    bounds: boundaries for parameter search

    Optionally, scale the data with its median value
    to make as many data points as possible close to unity
    NOTE: This may affect the parameters!
    """
    x_data = np.asarray(x_data).astype(float)
    data = np.asarray(data).astype(float)
    #the scaling factor:
    y_scale = np.median(data) if scale else 1
    y = data/y_scale
    if sigma is not None:
        sigma = np.asarray(sigma).astype(float)/y_scale
    try:
        popt, pcov = curve_fit(func, x_data, y,
                               p0=p0, sigma=sigma, bounds=bounds)
        # standard error of params
        # perr = np.sqrt(np.diag(pcov))
        # The best-fit model
        if x_model is None:
            x_model = x_data
        else:
            x_model = np.asarray(x_model).astype(float)
        model = y_scale*func(x_model, *popt)
    except Exception as ex:
        print(ex)
        popt = pcov = model = None

    return popt, pcov, model, y_scale

def Newt(func, apr,
         acc=1.E-4, max_iter=50,
         debug=False):
    """
    Newton's method to solve for x the equation

         f(x) = 0

    func -- returns tuple (f, f'), the function and its derivative
    apr  -- an initial guess
    acc  -- solution accuracy
    max_iter -- limit on number of iterations
    debug -- for tracing numerical problems;
             print solution progress and failure message
    """
    x = apr # initial guess
    for itr in range(max_iter):
        if debug:
            msg = f'  Newton:  {iter}: {x = :10.5E}'
        f, d = func(x)
        dx = -f/d #= -f/f'
        if debug:
            msg += f'  {dx = :10.5E}'
            print(msg)
        x += dx
        r = dx if x==0 else dx/x #for convergence test
        if abs(f) + abs(r) < acc:
            return x
    if debug:
        print(f'>>>Failure: Newton iterations exceeded max {max_iter}')
    return None



###############################################################
################ Plotting Utils ###############################
###############################################################



def add_top_axis(fig, pars, t):
    x = pars.ru*t
    upper_label = 'g$_u$t'
    if hasattr(pars, 'xh'):
        x -= pars.xh
        upper_label = 'g$_u$(t - t$_h$)'
    ax2 = fig.twiny()
    ax2.set(xlim=(x[0], x[-1]),
            xlabel=upper_label
           )
    ax2.xaxis.set_minor_locator(AutoMinorLocator())


def get_ufit(fit, S):
    if fit.mode.mode_id == 'exp': return None
    r0 = fit.pars.ru
    u_fit = SN(
                model = fit.de_hinder(S.time),
                r_model = np.full_like(fit.model, r0)
              )
    u_fit.dev = 1 - u_fit.model/S.data
    return u_fit


def the_unit(ts):
    unit = ts.info.unit
    if unit == '$': unit = '\$'
    return unit


def tag4(row, col, n_cols, chr1='a'):
    ''' Letter designation for a sublpot
    '''
    num = row*n_cols + col
    tag = chr(ord(chr1) + num)
    return f'({tag})'


