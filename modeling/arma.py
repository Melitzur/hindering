'''

Contents:
---------
ARIMA_name4
best_arma
    pq
'''

#############################################################
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
#############################################################

def ARIMA_name4(pdq):
    ''' Proper name for 3-tuple pdq

    Parameters
    ----------
    pdq : 3-tuple
        The (p, d, q) ARIMA order

    Returns
    -------
    string:
        AR(5) for pdq = (5,0,0)
        ARMA(3, 2) for pdq = (3,0,2), etc
    '''
    name = ''.join([s for s in np.where(pdq,
                                        ['AR','I','MA'],
                                        3*[''])
                   ])
    args = ', '.join(f'{a}' for a in pdq if a > 0)
    return f'{name}({args})'


def best_arma(y, abs_max=5, dates=None, verbose=False,
              **kw_arima
              ):
    ''' Search for ARMA parameters that minimize the fit BIC

    Employs statsmodels.tsa.arima.model.ARIMA:
    https://www.statsmodels.org/devel/generated/statsmodels.tsa.arima.model.ARIMA.html

    A more general implementation with bells and whistles
    is in module Work\fluctuations\tsa.py

    See also
    https://www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.arma_order_select_ic.html
    https://www.statsmodels.org/devel/_modules/statsmodels/tsa/stattools.html#arma_order_select_ic

    Parameters
    ----------
    y : array_like
        The observed time-series process
    abs_max : int
        The maximum ARMA parameters p and q that
        will be searched for minimum BIC.
    dates : arry_like, optional
        When supplied, replaces the dates attribute of the returned
        ARIMA instance
    verbose : int, optional
        Output control.
            0 - no output; default
            1 - print only final result
            2 - print also results for every step
    kw_arima : dict, optional
        Optional parameters to pass to sm.tsa.arima.model.ARIMA
        For example, to suppress the enforcement of stationarity
        and invertibility use
        tweaks = {'enforce_stationarity': False,
                  'enforce_invertibility': False
                 }
        Here's what these two, True by default, are doing:
        enforce_stationarity : bool, optional
            Whether or not to require the autoregressive parameters
            to correspond to a stationarity process.
        enforce_invertibility : bool, optional
            Whether or not to require the moving average parameters
            to correspond to an invertible process.

    Returns
    -------
    bf : sm.tsa.arima.ARIMA instance of the best-fitting ARMA model with
        the following added attributes:

        signature: str
            A string characterizing the solution;
            example:  BIC-selected fit is ARMA(2,4)
        model: np.array
            The fitting model = bf.fittedvalues
        ste: np.array
            The model standard error

        bf.data.dates is replaced by the parameter dates when supplied

    Example
    -------
    The operation
    >>> bf = best_arma(y)
    followed by
    >>> bf.summary()
    will produce tabulation of the fitting results.

    Some other attribues:
    bf.data.endog is the input data y
    bf.resid are the fitting residuals
    bf.specification['order'] is the (p, d, q) order triplet
    bf.bic is the BIC of the fit
    bf.get_prediction().conf_int() is the fit CI
    '''
    import warnings
    warnings.filterwarnings("ignore") # specify to ignore ARIMA warning messages

    if verbose > 1: print('\n(p, q)     BIC') # tabulation header

    def pq(p_max):
        ''' Grid for (p, q) search
        '''
        a = []
        for p in range(p_max+1):
            q0 = (p_max if p < p_max else
                  0
                 )
            for q in range(q0, p_max+1):
                a.append((p, q))
        return a

    bf = None
    bic_min = np.inf
    pmin = qmin = None
    p_max = 0
    while p_max < abs_max:
        p_max += 1
        for pars in pq(p_max):
            (p, q) = pars
            fit = ARIMA(y, order=(p, 0, q)).fit()
            bic = fit.bic
            if verbose > 1: print(f'{pars} {bic:10.2f}')
            if bic < bic_min:
                bic_min = bic
                pmin = p
                qmin = q
                bf = fit
        if pmin < p_max and qmin < p_max: break
    if bf is None: return

    # Add attributes to the sm model instance:
    the_name = ARIMA_name4((pmin, 0, qmin))
    signature = f'BIC-selected fit is {the_name}'
    setattr(bf, 'signature', signature)
    setattr(bf, 'model', bf.fittedvalues)
    pred = bf.get_prediction()
    setattr(bf, 'ste', pred.se_mean)
    if dates is not None:
        setattr(bf.data, 'dates', dates)
    if verbose:
        print(f'\n{the_name} is the fit with minimum BIC = {bic_min:.2f}')
        print(bf.summary().tables[1])
        if (pmin == abs_max) or (qmin == abs_max):
            print(f'\n>>> Warning: Minimal BIC is at search limit of {abs_max}')

    return bf
