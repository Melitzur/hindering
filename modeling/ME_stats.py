'''
Contents:
---------
get_rss
get_fvu
get_bic
sma0
sma
EMWA
lin_reg
    fn
f_stat
trend_test
MK_Zmax
mk_test
MK_test_ME

'''
import numpy as np
from numpy import sqrt, log as ln
import scipy
from scipy.stats import norm, f as fisher_f
from scipy.special import ndtri, ndtr
from types import SimpleNamespace as SN

###################################################################################


def get_rss(data, model, sigma=None):
    '''Residual Sum of Squares, potentailly weighted
    '''
    x = data - model
    if sigma is not None: x /= sigma
    return np.sum(x*x)

def get_fvu(data, model, weights=None):
    '''Coefficient of Determination R^2 for 1d np arrays
    https://en.wikipedia.org/wiki/Coefficient_of_determination
    https://en.wikipedia.org/wiki/Fraction_of_variance_unexplained
    https://en.wikipedia.org/wiki/Explained_sum_of_squares#Partitioning_in_the_general_ordinary_least_squares_model
    https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/metrics/_regression.py#L702
    '''
    weights = (np.ones_like(data) if weights is None else
               np.atleast_1d(weights).astype(float)
              )
    SS_res = np.sum(weights*(data - model)**2)
    SS_tot = np.sum(weights*(data - np.average(data, weights=weights))**2)
    #fraction of variance unexplained; 1 - R2
    fvu = SS_res/SS_tot
    return fvu, 1 - fvu #fvu, R2

def get_bic(rss, n, k):
    '''
    https://en.wikipedia.org/wiki/Bayesian_information_criterion
    On the BIC definition for Gaussian special case:
        Under the assumption that the model errors or disturbances
        are independent and identically distributed according to a
        normal distribution and the boundary condition that the
        derivative of the log likelihood with respect to the true
        variance is zero,..., In terms of the residual sum of squares
        (RSS) the BIC is

            BIC = n*ln(RSS/n) + k*ln(n)

        where n = the number of data points
              k = the number of parameters estimated by the model

    '''
    return n*ln(rss/n) + k*ln(n)



#Moving averages galore:
# https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
# Two versions for calculating simple moving averages:
def sma0(data, w):
    """My own simple method
    """
    return np.array([np.mean(data[i:i+w])
                     for i in range(len(data)+1-w)])

def sma(data_set, frame):
    """ Moving average using convolution with uniform weight 1/frame_size.
    The 'valid' option ensures there are no edge effcts;
    only elements where data_set and frame fully overlap are returned
    """
    window = np.ones(int(frame))/float(frame)
    return np.convolve(data_set, window, 'valid')

#Smoothing with Exponentially Weighted Moving Averages (EMWA)
def EMWA(X, w):
    """
    Given time-series X[t]
    E[t] = exponentially weighted moving average at time t
    definition:

        E[t] = w*X[t] + (1 - w)*E[t-1]

    with E[0] = X[0] and with
        0 <= w <= 1
    the weighting decrease coefficient; a higher w discounts
    older observations faster.
    w = 1 gives the original series, so no smooting
    w = 0 gives E0 = E1 = E2 = .... = X0; so much smoothing
    that the averages are constant at the initial value

    https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/exponentially-weighted-moving-average-ewma/
    https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
    """
    E = np.empty(len(X))
    E[0] = X[0]
    for n in range(1, len(X)):
        E[n] = w*X[n] + (1 - w)*E[n-1]
    return E


def lin_reg(y, *x, sided=2, verbose=False):
    '''My version of Linear Regression
    Created because an innocuous example bombed out with both
    linregress from scipy.stats and with statsmodels:
    see Work/Notes/lin_regress_problem.ipnnb
    Here we just use the trusty curve_fit (fitter) to solve for
    the regression parameters; this works, giving another option.

    See also:
    https://www.statology.org/p-value-from-t-score-python/
    https://keydifferences.com/difference-between-t-test-and-z-test.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html


    Parameters
    ----------
    y : array_like
        The dependent variable
    x : array_like
        The independent variable(s)

    '''
    from modeling.utils import fitter

    y = np.asarray(y).astype(float)
    n = len(y)
    x = [np.asarray(x_).astype(float) for x_ in x]
    for x_ in x:   # make sure all arrays have same length
        if len(x_) != n:
            print(f'Lengths of x ({len(x_)}) and y ({n}) are different')
            return
    nx = len(x)

    def fn(x, c, *a):
        return c + np.array(a).dot(np.array(x))

    pars, pcov, fit, _ = fitter(fn, x, y,
                                p0=(nx + 1)*[0.])
    ste = np.sqrt(np.diag(pcov))


    tiny = np.finfo(float).tiny
    t = np.array(pars)/(ste + tiny)  # t-scores; with division protection
    dof = n - 2                      # 2 degrees of freedom used up for mean and ste
    if sided != 2: sided = 1
    # sf(): Survival function (also defined as 1 - cdf, but sf is sometimes
    #       more accurate).
    p = sided*scipy.stats.t.sf(abs(t), dof)  # p-values

    if verbose:
        print('*** Linear-regression results:\n')
        print(9*' '+'value'+5*' '+'ste'+ 7*' '+'t'+6*' '+'p-value')
        slopes = (['slope'] if nx == 1 else
                  [f'slope{i+1}' for i in range(nx)]
                 )
        names = ['const'] + slopes
        for i, name in enumerate(names):
            s = f'{name:6}'
            s += f'{pars[i]:8.3f} {ste[i]:8.3f} {t[i]:8.3f} {p[i]:10.3E}'
            print(s)

    return SN(pars = pars,
              ste = ste,
              t = t,
              p = p,
              fit = fit
             )






#Calculation of the F-test statistic and p-value
#based on https://en.wikipedia.org/wiki/F-test

def f_stat(dat, model1, p1, model2, p2, sigma, alpha=0.01):
    '''Calculate the statistic for the F-test
       model1 and model2 may exceed the data length
    '''
    nothing = 3*(None,)
    if p1 >= p2:
        print(f'***Error calling f_stat with p1 = {p1}, p2 = {p2}')
        print( '   p2 MUST be larger than p1')
        return nothing

    n = len(dat)
    M1, M2 = model1[:n], model2[:n]
    RSS1 = get_rss(dat, M1, sigma)
    RSS2 = get_rss(dat, M2, sigma)
    df1 = p2 - p1
    df2 = n - p2 - 1
    Fc = fisher_f.ppf(1-alpha, df1, df2) #critical F for alpha
    F = (RSS1/RSS2 - 1)*df2/df1
    p = 1. - fisher_f.cdf(F, df1, df2)
    return F, p, Fc


############## Trend Test with MK ###################################


def trend_test(x, effect, data_id='', alpha=0.01, verbose=False):
    ''' MK-testing for any trend, or specific growth and decline
    Interface to MK-testing function MK_test_ME

    Parameters
    ----------
    x : array_like
        The sequence to test for trend
    effect : str
        Must be 'growth', 'decline' or 'trend'
    data_id : str, optional
        Data identifier for output
    alpha : float, optional
        Significance level of the MK test
    verbose : str or bool, optional
        output control

    Returns
    -------
    result : bool
        The test's conclusion
    z : float
        The test's statistic
    p : float
        The test's p-value
    '''
    nothing = 3*(None,)
    the_trend = {'growth': 'up',
                 'decline': 'down',
                 'trend': 'upordown'
                }
    if not effect in the_trend:
        raise ValueError(f"{effect} is not an allowed 'effect' parameter"
                         " for trend_test; acceptable are only"
                         " 'growth', 'decline' and 'trend'"
                        )
        return nothing

    Ha = the_trend[effect]
    MK, z, p = MK_test_ME(x, Ha=Ha, alpha=alpha)
    result = 'accept' in MK  # True if string MK contains 'accept'
    if verbose:
        evd = 'Evidence' if result else 'No evidence'
        s = f'{z = :.3G}, p-value = {p:.3G}'
        print(f'{evd} for {effect} from {data_id}: {s}')
    return result, z, p


def MK_Zmax(n):
    '''Maximum possible Z for monotonically increasing
       series with n terms:
       max number of pairs is n(n-1)/2
       variance is n*(n-1)*(2*n+5)/18
    '''
    return (n*(n-1)/2 - 1)/sqrt(n*(n-1)*(2*n+5)/18)


################ Two versions of the MK test ##########
'''
Code to conduct the MK-test
author: Michael Schramm
https://github.com/mps9506/Mann-Kendall-Trend
'''
def mk_test(x, alpha=0.05):
    '''
    Input:
        x:   a vector of data
        alpha: significance level (0.05 default)

    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the significance test
        z: normalized test statistics

    '''
    n = len(x)

    # calculate S
    s = 0
    for k in range(n-1):
        for j in range(k+1, n):
            s += np.sign(x[j] - x[k])

    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    var_s = (n*(n-1)*(2*n+5))/18
    if n != g:  # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(x == unique_x[i])
        var_s -= np.sum(tp*(tp-1)*(2*tp+5))/18

    if s > 0:
        z = (s - 1)/np.sqrt(var_s)
    elif s < 0:
        z = (s + 1)/np.sqrt(var_s)
    else: # s == 0:
        z = 0

    # calculate the p_value
    p = 2*(1-norm.cdf(abs(z)))  # two tail test
    h = abs(z) > norm.ppf(1-alpha/2)

    if (z < 0) and h:
        trend = 'decreasing'
    elif (z > 0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'

    return trend, h, p, z

###########################
'''
Code to conduct the MK test with alternative hypothesis
of upward or downward trend
https://up-rs-esp.github.io/mkt/

 Created: Mon Apr 17, 2017  01:18PM
 Last modified: Mon Apr 17, 2017  09:24PM
 Copyright: Bedartha Goswami <goswami@uni-potsdam.de>

Original name:
def test(t, x, eps=None, alpha=None, Ha=None):

ME version:
Removed the bonus calculation of linear correlation line.
This eliminates the need for the variable t

Modified output---added Zmk to return
'''
#def test_ME(t, x, eps=None, alpha=None, Ha=None):
def MK_test_ME(x, eps=np.finfo(float).eps, alpha=None, Ha=None):
    """
    Runs the Mann-Kendall test for trend in time series data.

    Parameters
    ----------
    t : 1D numpy.ndarray        #ME: eliminated
        array of the time points of measurements
    x : 1D numpy.ndarray
        array containing the measurements corresponding to entries of 't'
    eps : scalar, float, greater than zero
        least count error of measurements which help determine ties in the data
    alpha : scalar, float, greater than zero
        significance level of the statistical test (Type I error)
    Ha : string, options include 'up', 'down', 'upordown'
        type of test: one-sided ('up' or 'down') or two-sided ('updown')

    Returns
    -------
    MK : string
        result of the statistical test indicating whether or not to accept hte
        alternative hypothesis 'Ha'
    m : scalar, float               #ME: removed
        slope of the linear fit to the data
    c : scalar, float               #ME: removed
        intercept of the linear fit to the data
    p : scalar, float, greater than zero
        p-value of the obtained Z-score statistic for the Mann-Kendall test

    Raises
    ------
    AssertionError : error
                    least count error of measurements 'eps' is not given
    AssertionError : error
                    significance level of test 'alpha' is not given
    AssertionError : error
                    alternative hypothesis 'Ha' is not given

    """
    # assert a least count for the measurements x
    assert eps, "Please provide least count error for measurements 'x'"
    assert alpha, "Please provide significance level 'alpha' for the test"
    assert Ha, "Please provide the alternative hypothesis 'Ha'"

    # estimate sign of all possible (n(n-1)) / 2 differences
    #n = len(t) --- t eliminated
    n = len(x)
    sgn = np.zeros((n, n), dtype="int")
    for i in range(n):
        tmp = x - x[i]
        tmp[np.where(np.fabs(tmp) <= eps)] = 0.
        sgn[i] = np.sign(tmp)

    # estimate mean of the sign of all possible differences
    S = sgn[np.triu_indices(n, k=1)].sum()

    # estimate variance of the sign of all possible differences
    # 1. Determine no. of tie groups 'p' and no. of ties in each group 'q'
    np.fill_diagonal(sgn, eps * 1E6)
    i, j = np.where(sgn == 0.)
    ties = np.unique(x[i])
    p = len(ties)
    q = np.zeros(len(ties), dtype="int")
    for k in range(p):
        idx = np.where(np.fabs(x - ties[k]) < eps)[0]
        q[k] = len(idx)
    # 2. Determine the two terms in the variance calculation
    term1 = n * (n - 1) * (2 * n + 5)
    term2 = (q * (q - 1) * (2 * q + 5)).sum()
    # 3. estimate variance
    varS = float(term1 - term2) / 18.

    # Compute the Z-score based on above estimated mean and variance
    if S > eps:
        Zmk = (S - 1) / np.sqrt(varS)
    elif np.fabs(S) <= eps:
        Zmk = 0.
    elif S < -eps:
        Zmk = (S + 1) / np.sqrt(varS)

    # compute test based on given 'alpha' and alternative hypothesis
    # note: for all the following cases, the null hypothesis Ho is:
    # Ho := there is no monotonic trend
    #
    # Ha := There is an upward monotonic trend
    if Ha == "up":
        Z_ = ndtri(1. - alpha)
        if Zmk >= Z_:
            MK = "accept Ha := upward trend"
        else:
            MK = "reject Ha := upward trend"
    # Ha := There is a downward monotonic trend
    elif Ha == "down":
        Z_ = ndtri(1. - alpha)
        if Zmk <= -Z_:
            MK = "accept Ha := downward trend"
        else:
            MK = "reject Ha := downward trend"
    # Ha := There is an upward OR downward monotonic trend
    elif Ha == "upordown":
        Z_ = ndtri(1. - alpha / 2.)
        if np.fabs(Zmk) >= Z_:
            MK = "accept Ha := upward OR downward trend"
        else:
            MK = "reject Ha := upward OR downward trend"

    ''' ME: removed
    # ----------
    # AS A BONUS
    # ----------
    # estimate the slope and intercept of the line
    m = np.corrcoef(t, x)[0, 1] * (np.std(x) / np.std(t))
    c = np.mean(x) - m * np.mean(t)
    '''
    # ----------
    # AS A BONUS
    # ----------
    # estimate the p-value for the obtained Z-score Zmk
    if S > eps:
        if Ha == "up":
            p = 1. - ndtr(Zmk)
        elif Ha == "down":
            p = ndtr(Zmk)
        elif Ha == "upordown":
            p = 0.5 * (1. - ndtr(Zmk))
    elif np.fabs(S) <= eps:
        p = 0.5
    elif S < -eps:
        if Ha == "up":
            p = 1. - ndtr(Zmk)
        elif Ha == "down":
            p = ndtr(Zmk)
        elif Ha == "upordown":
            p = 0.5 * (ndtr(Zmk))

    #return MK, m, c, p  -- original
    return MK, Zmk, p


