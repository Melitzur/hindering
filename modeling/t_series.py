'''
Contents:
---------
time_series
class T_Series:
    __init__
    data_overvu
        mag
        epoch
    get_trend
class Full_fit:
    model_from
    __init__
    get_ci
    conf_interval
    check4bad
'''

##################################################

import numpy as np
from scipy.interpolate import UnivariateSpline
from types import SimpleNamespace as SN

from .the_globals import *
from .utils import header, headline, num2readable
from .ME_stats import trend_test, MK_Zmax

##################################################


def time_series(info, data, verbose=False,
                kw_args=None, kw_hfit=None,
                kw_arma=None, kw_full=None):
    '''Create and analize time series
    Structure of the produced object ts and the
    relevant attributes of it and its componets:

    ts ---              T_series instance
          |
          info
          data
          time
          dates (optional)
          |
          hfit ---      Fit instance (module fitting)
                  |
                  mode
                  model
                  dev
                  popt
                  pcov
          arma ---      Statsmodels instance (enhanced; module arma)
                  |
                  signature
                  model
                  ste
          fit ---       Full_Fit instance
                 |
                 time
                 hfit
                 arma
                 model
                 ste
                 ci

    Note:
    -----
    ARMA analysis requires single time interval, so keyword 'time'
    is not allowed in kw_args

    In passing multiple keyword sets to the multiple functions,
    avoid empty dictionary as a parameter:
    https://stackoverflow.com/questions/26320899/why-is-the-empty-dictionary-a-dangerous-default-value-in-python
    '''
    ########## Keywords setup ##########

    defaults = dict(alpha_MK=0.01)
    if kw_args is not None: defaults.update(kw_args)
    kw_args = defaults

    defaults = dict(sigma_is='data', max_nk=[10, 10],
                    alpha_F=0.01)
    if kw_hfit is not None: defaults.update(kw_hfit)
    kw_hfit = defaults

    defaults = dict(abs_max=5)
    if kw_arma is not None: defaults.update(kw_arma)
    kw_arma = defaults

    defaults = dict(with_ci=True, sample_size=100,
                    n_iter=2, acc=.01, conf=95)
    if kw_full is not None: defaults.update(kw_full)
    kw_full = defaults
    #####################################

    ts = T_Series(info, data, verbose=verbose, **kw_args)

    if verbose: headline('Growth Trend', space=True, width=30)
    hfit = ts.get_trend(verbose=verbose, **kw_hfit)
    if hfit is None:
        return

    if verbose: headline('ARMA', space=True, width=30)
    # ARMA analysis requires equal time intervals,
    # so make sure:
    single_dt = np.all(np.diff(np.diff(ts.time)) == 0)
    if not single_dt:
        print('>>>> Error: time intervals are not equal')
        return
    from .arma import best_arma
    arma = best_arma(hfit.dev, dates=ts.dates, verbose=verbose,
                     **kw_arma)

    if verbose: headline('Full Fit', space=True, width=30)
    fit = Full_fit(ts.time, hfit, arma, verbose=verbose, **kw_full)


    if verbose: print(2*'\n')
    ts.hfit = hfit
    ts.arma = arma
    ts.fit = fit

    return ts



class T_Series:
    ''' Time series analysis

    Parameters
    ----------
    info : Var_info instance (the_globals.py)
        Information about the data
    data : array_like
        A sequence of data points
    time : array_like, optional
        Index of the data points (default) when all time intervals
        are equal. For unequal intervals, this parameter MUST be
        supplied; it then contains the times of the data points in
        units of info.t_unit starting at 0. When the intervals are a
        mixture of, say, years and decades, better to use decade as
        the unit so that the largest intervals are 1. For example,
        with mixed-interval census data, years and decades, use

            time = 0.1*(years - years[0])

        where years is the sequence of year numbers of the supplied
        data that has both consecutive years and decennial input.
    dates : array_like of strings, optional
        Sequence such as years or actual dates in str format
    verbose : int or bool
        output control:
            0 -- No output (default)
            1 -- Some output messages about the data
            2 -- Information about the fits
    hdr : str
        Header line for output. When None (default),
        a header is generated from the input data when
        verbose > 1
    alpha_MK : float
        Threshold p-value for MK-tests; default is 0.01

    Attributes
    ----------
    The following input parameters are also class attributes
    with the same names:
        info, data, time, dates -- see above

    Additional attribiutes:
        r : object with two attributes:
            r.data : the growth-rate data
            r.spline : spline fit to r.data

    '''

    def __init__(self, info, data, *,
                 time=None, dates=None, hdr=None,
                 alpha_MK=0.01, verbose=False
                ):
        self.info = info
        self.data = np.asarray(data, dtype=float)
        time = (np.arange(len(data)) if time is None else
                time - time[0]
               )
        self.time = time.astype(float)
        self.dates = dates
        if verbose:
            if hdr is None:
                hdr = f'{info.desc}: {len(data)} data points'
                if dates is not None:
                    hdr += f', from {info.t_unit} {dates[0]}--{dates[-1]}'
            header(hdr)
            self.data_overvu()

        # MK-test for growth
        if verbose==2:
            print(f'MK-Tests: Zmax = {MK_Zmax(len(data)):.4G}')
        growth = trend_test(self.data, 'growth', data_id='data',
                            verbose=verbose, alpha=alpha_MK)[0]
        if not growth:
            self.growth = False
            print('***No growth! Execution terminated')
            return

        self.growth = True
        # We have growth so calculate growth rate
        # per unit time and its spline smoothing,
        # needed for initial guess during fitting.
        # Both are collected under sub-class r
        self.r = SN(
                    data = np.gradient(self.data, self.time)/self.data
                   )
        self.r.spline = UnivariateSpline(self.time, self.r.data)(self.time)

        #Exclude accelerated growth:
        accelerated = trend_test(self.r.data, 'growth')[0]
        self.accelerated = accelerated
        if accelerated:
            print('***Accelerated growth! Execution terminated')
            return

        if verbose:
            # info from MK-test for hindering:
            hindering = trend_test(self.r.data, 'decline')[0]
            no = '' if hindering else 'no '
            print(f'Growth rate gives {no}indication for hindering')

            '''
            trend_test(self.r.data, 'decline',
                       data_id='growth rate',
                       verbose=verbose, alpha=alpha_MK)[0]
            trend_test(self.r.spline, 'decline',
                       data_id='growth-rate spline',
                       verbose=verbose, alpha=alpha_MK)[0]
            '''


    def data_overvu(self):
        unit = self.info.unit
        def mag(ind):
            return unit + num2readable(self.data[ind])
        def epoch(ind):
            if self.dates is None:
                epch = f'{self.info.t_unit} {int(self.time[ind])}'
            else:
                epch = self.dates[ind]
            return epch
        s = f'{self.info.desc} grew from {mag(0)} in {epoch(0)} '
        s+= f'to {mag(-1)} in {epoch(-1)}\n'
        s+= f'an increase by factor of {self.data[-1]/self.data[0]:.4G}'
        print(s)


    def get_trend(self, *, sigma_is='data',
                  max_nk=[10, 10], alpha_F=0.01,
                  verbose=False):
        '''
        Hindering fit for time series

        Parameters
        ----------
        See get_fits in module fitting

        Attributes added to self:
        -------------------------
        fits : list of Fit instances; class Fit is
               from module fitting

        returns
        -------
        bf : best fitting Fit instance; 1st member of fits
        '''

        if not self.growth:
            print('***No growth! Execution terminated')
            return

        if self.accelerated:
            print('***Accelerated growth! Execution terminated')
            return

        from .fitting import get_fits
        fits = get_fits(self, sigma_is,
                        max_nk, alpha_F,
                        verbose)

        self.fits = fits
        return fits[0] # best fit


class Full_fit:
    '''
    Overall model prediction for the trend and the arma fit
    for the fluctuations around it:

       model = trend/(1 - arma)

    Parameters
    ----------
    time : array_like
        Time, measured from start of the time series
    hfit : obj, a Fit instance (module fitting)
        The best-fitting trend
    arma : obj, a statsmodels ARMA instance (module arma)
        The best ARMA fit to the fluctuations around
        the trend
    verbose : bool
        Output control
    with_ci : bool
        Add confidence interval (CI) when True (default). All
        other optional parameters are in effect only in
        this case.
    conf : int or float
        The confidence level, in percent, for the CI calculation.
    sample_size : int
        Number of the hindering and arma samples drawn around the
        input fits for CI estimate (see below). Default is 100.
    acc : float
        Accuracy criterion: upper limit on the mean deviation of
        hfit and arma sample means from the input values. Default
        is .01
    n_iter : int
        Number of sampling repetitions allowed to reach the
        prescribed accuracy acc. In each repetition the sample
        size increases by factor 10. Default is 2, for a maximum
        sample size of 10,000

    Attributes
    ----------
    The following input parameters are also class attributes
    with the same names:
        time, hfit, arma -- see above

    Additional attribiutes:
    model : array_like
        The complete model fit, comprised of the hfit trend
        and arma fluctuations
    ste : ndarray (optional)
        The standard error of the model fit
    ci : ndarray (optional)
        The confidence interval; produced when with_ci is True.
        With n the length of the time array, ci is a nX2 array.
        The lower bound of the confidence interval is ci[:,0],
        the upper bound ci[:,1]
    ci_OK : bool (optional)
        Indicates whether the accuracy criterion was met in the
        confidence interval calculation
    conf : int or float
        The confidence level, in percent, of ci
    '''

    @staticmethod
    def model_from(h_model, a_model):
        return h_model/(1 - a_model)


    def __init__(self, t, hfit, arma, with_ci=True,
                 sample_size=100, acc=.01, n_iter=2,
                 conf=95, verbose=False
                 ):
        self.time = t
        self.hfit = hfit
        self.arma = arma
        self.model = self.model_from(hfit.model, arma.model)
        if with_ci:
            ste, ci, ok = self.get_ci(sample_size=sample_size,
                               acc=acc, conf=conf,
                               n_iter=n_iter,
                               verbose=verbose)
            self.conf = conf
            self.acc = acc
            self.ste = ste
            self.ci = ci
            self.ci_OK = ok


    def get_ci(self, sample_size=100, acc=.01, conf=95,
                 n_iter=2, verbose=False
                 ):
        _kw = dict(acc=acc, conf=conf, verbose=verbose)

        size = sample_size
        ok = False
        while not ok and n_iter:
            ste, ci, ok = self.conf_interval(sample_size=size, **_kw)
            n_iter -= 1
            size *= 10
        return ste, ci, ok


    def conf_interval(self,
                      sample_size=100, acc=.01, conf=95,
                      verbose=False
                     ):
        '''
        The arma fit comes from statsmodels with its own error
        estimates, but for the hindering fit for trend we only have
        the error estimates of the parameters. So we have to (1)
        estimate the trend errors from those of its parameters and
        (2) get the error of the full model from those of its two
        components.

        Achieve both with Zeljko's suggestion to determine CI
        through sampling at every time point.  For the trend model,
        Gaussian multivariate sampling centered on the optimal popt
        with covariance matrix pcov. For arma, a Gaussian centered
        on the fit with the arma standard deviation. The model
        sample is generated from the two samples obtained
        independently this way. Checking whether the sampling mean
        agrees with the fitted model is a test of sample size. The
        CI is determined from the model sample with numpy's
        percentile function.

        https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        https://online.stat.psu.edu/stat505/book/export/html/636
        '''
        nothing = 2*(None,)
        if not 0 < conf < 100:
            print(f'>>>> Confidence level ({conf = }) must be in [0, 100]')
            return nothing

        hfit = self.hfit
        arma = self.arma
        t = self.time

        #create sample of h-models
        func = hfit.mode.func
        popt = hfit.popt
        pcov = hfit.pcov
        h_sample = np.array([
            func(t, *pars) for pars in
            np.random.multivariate_normal(popt, pcov, size=sample_size)
            ])

        #create sample of arma models
        a_sample = np.random.normal(
                       loc=arma.model,
                       scale=arma.ste,
                       size=[sample_size, len(arma.model)]
                       )

        #create sample of full models
        sample = self.model_from(h_sample, a_sample)
        err = np.mean(np.abs(
              sample.mean(axis=0)/self.model - 1
              ))
        if err < acc:
            ok = True
            r = '<'
        else:
            ok = False
            r = '>'

        s = f'\n*** CI({conf}) calculation for full fit:\n'
        s+= f'{sample_size = :,}, {err = :.2E} {r} {acc = :.2E}'
        if not ok:
            print(s)
            print('>>>>Warning: need larger sampling')
        elif verbose:
            print(s)

        # Standard error:
        ste = np.std(sample, axis=0)

        # The confidence interval:
        a = 0.5*(100 - conf)
        c_int = np.array([a, 100 - a])
        ci = np.percentile(sample, c_int, axis=0).T

        return ste, ci, ok



    def check4bad(self, data, verbose=False):
        ''' Check for data points outside the CI boundaries of fit
        With a confidence level of 95%, for example, 5% of the points
        can be expected to randomly fall outside the CI boundaries

        Parameters
        ----------
        data : array_like
            The data for testing the full fit
        Returns
        -------
        n_rand : int
            Number of data points expected randomly outside the
            CI boundaries of the fit
        n_bad : int
            Number of data points outside the CI boundaries
            of the fit
        bad : array of Booleans
            False for points in agreement with the fit,
            True for points outside the CI boundaries
        Note:
        -----
        The fit is acceptable when n_bad <= n_rand
        '''
        ci_min, ci_max = self.ci[:,0], self.ci[:,1]
        # Data points that fall outside CI
        data = np.asarray(data, dtype=float)
        bad = (data < ci_min) | (data > ci_max)
        # and their number
        n_bad = bad.sum()

        #Number expected randomly outside CI boundaries
        conf = self.conf
        n_rand = int(len(data)*(100 - conf)/100)

        if verbose:
            CI = f'CI({conf})'
            print(f'\n{n_rand} points expected randomly outside {CI}')
            if n_bad == 0:
                s = 'All data points agree with predictions'
            else:
                ex = 'within' if n_bad <= n_rand else 'above'
                s = f'{n_bad} data points outside {CI} of prediction; {ex} expectations'
            print(s)

        return n_rand, n_bad, bad



##################################################
##################################################
##################################################
