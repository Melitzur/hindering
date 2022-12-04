'''
Follow the convention adopted by statsmodels of in-sample prediction and
out-of-sample forecasting. Our aim is to forecast n_fore years past the
end of data and provide justification by predicting the last n_fore terms
of the time series and checking whether the actual data points are
within the confidence interval of the prediction. To do that we must
truncate the last n_fore terms of the time series, derive the best
hindering and ARMA fits of this partial dataset and use them to
forecast the next n_fore years. So the two concepts are fundamentally
the same in this case: prediction is forecasting for the partial time
series.

Class Forecast does both. Function get_prediction generates the
truncated dataset and its prediction.


Contents:
---------
class Forecast:
    __init__
class Forecast_hind(Fit):
        __init__
get_prediction
forecast_plot

'''

##################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from .t_series import Full_fit, time_series
from .fitting import Fit, t2year
##################################################


class Forecast:
    ''' Forecasting time series

    Example:
    --------
    if verbose: print('\n*** Forecast for entire time series:')
    forecast = Forecast(n_fore, ts.hfit, ts.arma, verbose=verbose, **kw)
    '''

    def __init__(self, n_fore, hfit, arma, verbose=False, **kw):
        if verbose: print(f'Forecasing {n_fore} time steps' )
        n0 = len(hfit.model)
        t_fore = np.arange(n0, n0+n_fore).astype(float)
        # The forecast trend-fit object:
        h_fore = self.Forecast_hind(t_fore, hfit)

        # The forecast ARMA object:
        a_fore = arma.get_forecast(steps=n_fore)
        a_fore.model = a_fore.predicted_mean
        a_fore.ste = a_fore.se_mean

        self.time = t_fore
        self.fit = Full_fit(t_fore, h_fore, a_fore, verbose=verbose, **kw)


    class Forecast_hind(Fit):
        ''' A Fit object with only the attributes needed
            for forecasting: mode, model, popt and pcov
            (no dev). Required for out-of-sample forecasting
            where dev is meaningless
        '''
        def __init__(self, t, hfit):
            for attr in ['mode', 'popt', 'pcov']:
                setattr(self, attr, getattr(hfit, attr))
            self.model = self.model_calc(t)


def get_prediction(ts, n_pred, verbose=False,
                kw_args=None, kw_hfit=None,
                kw_arma=None, kw_full=None):
    ''' Prediction for the last n_pred points of time series

    Parameters
    ----------
    ts : T_Series object
        The time series being analyzed
    n_pred : int
        Number of time steps to predict

    returns
    -------
    partial : a T_Series instance
        The n_pred-truncated time series ts
    prediction : a Forecast instance
        The forecast from the truncated time series
    '''
    ########## Keywords setup ##########
    # All fits must be obtained with the same setup
    # as the input time series, so import from ts its
    # keyword values for sigma_is, alpha_F, acc and
    # conf. alpha_MK can be declared independently
    # since it doesn't do that much anyhow

    defaults = dict(alpha_MK=0.01)
    if kw_args is not None: defaults.update(kw_args)
    kw_args = defaults
    if ts.dates is not None:
        kw_args['dates'] = ts.dates[:-n_pred]

    defaults = dict(max_nk=[10, 10])
    if kw_hfit is not None: defaults.update(kw_hfit)
    kw_hfit = defaults
    kw_hfit['sigma_is'] = ts.hfit.sigma_is
    kw_hfit['alpha_F'] = ts.hfit.alpha_F

    defaults = dict(abs_max=5)
    if kw_arma is not None: defaults.update(kw_arma)
    kw_arma = defaults

    defaults = dict(with_ci=True, sample_size=100,
                    n_iter=2)
    if kw_full is not None: defaults.update(kw_full)
    kw_full = defaults
    kw_full['acc'] = ts.fit.acc
    kw_full['conf'] = ts.fit.conf
    #####################################

    if verbose: print('\n*** Getting partial time series:')
    nothing = 2*(None,)
    partial = time_series(ts.info, ts.data[:-n_pred],
                     verbose=verbose,
                     kw_args=kw_args, kw_hfit=kw_hfit,
                     kw_arma=kw_arma, kw_full=kw_full)
    if partial is None: return nothing

    if verbose: print('\n*** Prediction from partial time series:')
    prediction = Forecast(n_pred, partial.hfit, partial.arma,
                          verbose=verbose, **kw_full)

    return partial, prediction


############################################################

def forecast_plot(base, fit, fore,
                  fig=None, figsize=(7, 5),
                  title='', tag='', vu=None,
                 ):
    '''
    Both predicting and forecasting produce the same relation for
    n_plot but in one case it is the same as n_data, in the other
    it is bigger.

    forecasting:
        n_fit = n_data
        n_plot = n_data + n_fore = n_fit + n_fore > n_data
    predicting:
        n_fit = n_data - n_fore
        n_plot = n_data = n_fit + n_fore
    '''

    if fig is None:
        Fig, fig = plt.subplots(figsize=figsize)

    num2plot = len(fit.time) + len(fore.time)
    #index of 1st point to plot:
    ind0 = (0 if vu is None else
            max(num2plot - int(vu), 0)
           )
    # start and end of fitted data:
    fit0, fit1 =  fit.time[0], fit.time[-1]

    info, data = base.info, base.data
    t_unit = info.t_unit

    y_list = [data[ind0:], fit.fit.model[ind0:], fore.fit.model]
    x_list = [base.time[ind0:], fit.time[ind0:], fore.time]
    # convert time to years when relevant:
    if t2year(base.time, base.dates, t_unit) is not None:
        x_list = [t2year(x, base.dates, t_unit) for x in x_list]
        fit0 = t2year(fit0, base.dates, t_unit)
        fit1 = t2year(fit1, base.dates, t_unit)
    x_fit, x_fore = x_list[1], x_list[2]

    pred = ('prediction' if num2plot == len(data) else
            'forecast'
           )
    if title: title += '\n'
    title += f'fit {fit0}-{fit1}, {pred} to {x_fore[-1]}'
    labels = ['data', 'fit', pred]
    styles = [dict(color='C0', marker='o', lw=0.7, ms=3),
              dict(color='C1', lw=3, alpha=0.5),
              dict(color='red', lw=3, alpha=0.5)
             ]

    for x, y, label, style in zip(x_list, y_list, labels, styles):
        fig.semilogy(x, y, label=label, **style)

    fig.fill_between(x_fit, fit.fit.ci[ind0:,0], fit.fit.ci[ind0:,1],
            label=f'{base.fit.conf}% CI', color='grey', alpha=.2
            )
    fig.fill_between(x_fore, fore.fit.ci[:,0], fore.fit.ci[:,1],
            color='grey', alpha=.2
            )

    fig.xaxis.set_minor_locator(AutoMinorLocator())
    ylabel = f'{info.desc}' + (info.unit == '$')*' ($)'
    fig.set(
            xlabel=t_unit, #'year',
            ylabel=ylabel,
            title=title,
           )
    fig.legend(loc='upper left')
    if tag:
        fig.text(0.9, 0.05, tag, transform=fig.transAxes)




