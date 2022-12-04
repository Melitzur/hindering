'''

Contents:
---------
get_acf
get_ci
add_ci2plot
class Func:
    fn
    desc
    p0
    bounds
    tau2thalf
fit_acf
xcov
get_xcf

'''

#############################################################

import numpy as np
from numpy import exp, log as ln
from dataclasses import dataclass
from types import SimpleNamespace as SN

from .utils import fitter
from .ME_stats import get_rss, get_bic

#############################################################

def get_acf(y, nlags=None, fft=False, bartlett_confint=False):
    ''' Autocorrelation function
    Change the statsmodels defaults for fft and bartlett_confint
    https://www.statsmodels.org/devel/_modules/statsmodels/tsa/stattools.html#acf
    '''
    from statsmodels.tsa.stattools import acf
    if nlags is None: nlags = len(y) - 1
    return acf(y,
               nlags=nlags, fft=fft,
               bartlett_confint=bartlett_confint)


def get_ci(y, a=0.05, s=1, verbose=False):
    '''Confidence interval for sequence y;
    default 95% and sig = 1

    https://en.wikipedia.org/wiki/Confidence_interval
    https://en.wikipedia.org/wiki/1.96
    https://www.bmj.com/content/343/bmj.d2090
    '''
    import scipy.stats
    n = len(y)
    ci = s*scipy.stats.norm.ppf(1 - a/2)/np.sqrt(n)
    if verbose:
        print(
        f'{1 - a:.0%} confidence interval for {n = }, {s = :.2f} is {ci:.2f}'
        )
    return ci


def add_ci2plot(fig, y, a=0.05, s=1, positive_only=False):
    ''' Add confidence interval lines to fig
    '''
    ci = get_ci(y, a=a, s=s) # 1.96/np.sqrt(len(y))
    lims = [ci] if positive_only else [-ci, ci]
    conf = int(100*(1 - a))
    for z in lims:
        fig.axhline(z, color='grey', lw=0.7, dashes=[10,10],
                    label=f'CI({conf})' if z > 0 else ''
                   )

########### Analytic fits to ACF ################################

@dataclass
class Func:
    '''Properties and methods for each fitting function
    '''
    tag: str  # identifier

    def fn(self, t, *args):
        '''The fitting function
        '''
        tag = self.tag
        x = t/args[0]  # t/tau
        return (exp(-x)              if tag == 'exp' else
                exp(-x*x)            if tag == 'gau' else
                1./(1. + x)**args[1] if tag == 'pwr' else
                None
               )

    def desc(self, pars):
        ''' Description of the fitting function
        '''
        tag = self.tag
        s = ('exponential'          if tag == 'exp' else
             'Gaussian'             if tag == 'gau' else
             f'power {pars[1]:.2G}' if tag == 'pwr' else
             ''
            )
        return s

    @property
    def p0(self):
        '''Initial parameters for the fitter
        '''
        return [1., 1.] if self.tag=='pwr' else 1.

    @property
    def bounds(self):
        '''Bounds on parameter search
        '''
        upper = [np.inf, np.inf] if self.tag=='pwr' else np.inf
        return (0., upper)

    def tau2thalf(self, pars):
        ''' Convert tau to t_half
        '''
        tag = self.tag
        c = (ln(2.)               if tag == 'exp' else
             np.sqrt(ln(2.))      if tag == 'gau' else
             2.**(1./pars[1]) - 1 if tag == 'pwr' else
             None
            )
        return c*pars[0]



def fit_acf(acf, a=0.05, verbose=False):
    ''' Analytic fit for autocorrelation function
    Finds the best fit (smaller BIC) among exponential,
    Gaussian and power law to the first lags above the
    random noise level.

    Returns
    -------
    fits : list
        A list of fit objects, one for each function,
        each containing:
            pars -- best fit paramaters
            t_half -- ACF distribution width at half-height
            ste_half -- ste of t_half
            model -- the fit
        The overall best fit is fits[0]
    Example:
    --------
    The best fit can be obtained with:
    bf = fit_acf(acf)[0]
    '''
    # Use only the first lags with ACF above noise:
    ci = get_ci(acf, a=a, verbose=verbose)
    n0 = np.where(acf < ci)[0][0]
    t = np.arange(n0, dtype=float)
    y = acf[:n0]
    if verbose:
        conf = int(100*(1 - a))
        print(f'Lags up to {n0 - 1} have ACF outside the CI({conf}) noise level')

    the_fits = []
    for tag in ['exp', 'gau', 'pwr']:
        func = Func(tag)
        pars, pcov, model, _ = fitter(func.fn, t, y,
                                      p0=func.p0, bounds=func.bounds)
        if pars is None:
            continue
        t_half = func.tau2thalf(pars)
        ste = np.sqrt(np.diag(pcov))
        fit = SN(name=func.desc(pars),
                 pars=pars,
                 t_half=t_half,
                 ste_half=t_half*ste[0]/pars[0],
                 bic=get_bic(get_rss(y, model),
                             n0, len(pars)),
                 model=model,
                )
        the_fits.append(fit)

    if len(the_fits) == 0:
        if verbose: print('>>>> No fits!')
        return
    # sort fits by BIC:
    idx = np.argsort([fit.bic for fit in the_fits])
    fits = [the_fits[i] for i in idx]
    if verbose:
        print('\nAnalytic fits:')
        print(20*' '+'t_half     ste      bic')
        for fit in fits:
            print(f'{fit.name:15} {fit.t_half:9.2f} {fit.ste_half:9.2f} {fit.bic:8.2f}')
    return fits



##################### Plotting ##############################
###import matplotlib.pyplot as plt
###from matplotlib.ticker import AutoMinorLocator
###
###
###def y_and_acf_plots(entries, *,
###                    dates=None, xlabel='', figsize=None, Title='',
###                    bartlett_confint=True, fft=False, lags=None
###                   ):
###    '''Plots of time-series and their autocorrelation functions
###
###    Uses
###    https://www.statsmodels.org/devel/generated/statsmodels.graphics.tsaplots.plot_acf.html
###    https://www.statsmodels.org/devel/_modules/statsmodels/graphics/tsaplots.html#plot_acf
###
###    Parameters
###    ----------
###    entries : list of 2-tuples
###        Each entry in the list is a 2-tuple entry of (str, array_like),
###        plotted in a row of two panels.  The left panel plots the
###        array_like time-series, the string is the y-axis label of
###        the plot. The right panel plots the series autocorrelation
###        function with the aid of plot_acf
###    dates : arry_like, optional
###        When supplied becomes the x-axis of the left column
###    xlabel : str, optional
###        Typically the time unit of the time series. When supplied it
###        is the title of the x-axis of the left column and the unit
###        of the lags in the right-column x-axis label
###    figsize : 2-tuple
###        The figsize. The default is determined from the number of entries
###    Title : str, optional
###        Optional title for the whole figure
###
###    Returns
###    -------
###    Fig
###        Figure instance with the plots
###    '''
###    from statsmodels.graphics.tsaplots import plot_acf
###
###    n_rows = len(entries)
###    if figsize is None:
###        figsize=(12, 4*n_rows)
###    Fig, figs = plt.subplots(n_rows, 2, sharex='col', figsize=figsize)
###    if n_rows == 1: figs = [figs] # make sure figs is always a list
###    if Title: Fig.suptitle(Title)
###    x = (dates if dates is not None else
###         list(range(len(entries[0][1])))
###        )
###
###    for row, entry in enumerate(entries):
###        (ylabel, y) = entry
###        fig, acf = figs[row][0], figs[row][1]
###        # plot the entry
###        fig.axhline(0, color='grey', lw=0.2)
###        fig.plot(x[1:], y[1:])
###        fig.set(ylabel=ylabel)
###        fig.xaxis.set_tick_params(length=10)
###        # and its acf
###        plot_acf(y[1:], ax=acf, title='',
###                 bartlett_confint=bartlett_confint,
###                 fft=fft, lags=lags)
###
###        acf.xaxis.set_tick_params(length=10)
###
###        if row == 0: acf.set(title='ACF')
###        if row == n_rows - 1:
###            fig.set(xlabel=xlabel)
###            fig.xaxis.set_minor_locator(AutoMinorLocator())
###            acf_xlabel = 'lag'
###            if xlabel:
###                acf_xlabel += f' ({xlabel}s)'
###            acf.set(xlabel=acf_xlabel)
###            acf.xaxis.set_minor_locator(AutoMinorLocator())
###
###    plt.tight_layout()
###    Fig.subplots_adjust(hspace=0.06)
###    return Fig
###
###
#################################################




####### Experimental ##########################


def xcov(y1, y2):
    ''' X-covariance <[y1(t) - mean(y1)]*[y2(t + lag) - mean(y2)]>
    https://en.wikipedia.org/wiki/Cross-covariance

    Note:
    When y2 = y1 we get the autocovariance of y1
    The lag = 0 element of xcov(y1, y2) is the off-diagonal
    element of the 2x2 covariance matrix np.cov(y1, y2).
    https://numpy.org/doc/stable/reference/generated/numpy.cov.html
    Normalizing by sigma1*sigma2 gives the Pearson correlation
    coefficient, the off-diagonal element of np.corrcoef(y1, y2)
    https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
    '''
    n = len(y1)
    y1 -= y1.mean()
    y2 -= y2.mean()
    x = np.zeros_like(y1)
    for lag in range(n):
        x[lag] = y1[:n - lag].dot(y2[lag:])
    return x/n

def get_xcf(y1, y2):
    ''' Cross-correlation function, unity at lag = 0
    '''
    x = xcov(y1, y2)
    return x/x[0]


#########################################################################
if __name__ == '__main__':
    import sys

    sys.exit()

