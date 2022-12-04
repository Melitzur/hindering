"""
August, 2022
------------

Earlier versions relied on the MK-test to detect the presence or absence
of hindering. When hinderning was not detected, only the exponential fit
was attempted. Turns out that relying only on the MK test of the growth
rate as an indicator of hindering is not always sufficient. The Chile
1820--2018 population data from the Madison Project gives strong
indication of hindering --- a 5-sigma result for the MK test (p-value =
2.78E-07). Yet the exponential is actually the best fit. Adding hindeing
hardly does anything to the fit. Logistic modeling fails and both sth
and mth give the exact same ru, tryig to change hindering in the
margins, leading to yh = 2131(!) in both cases.

Conclusion
----------
Must always calculate the exponential fit, use BIC against the logistic
and F-test against the best hindeing fit.

Contents:
---------
get_fits
setup4fits
fit_it
p04
fit4
best_h_term
multi_term
f_test
output_popt
output_wk
bf_output
output_all
t2year
class Fit:
    __post_init__
    popt2pars
    model_calc
    r_model
    de_hinder

"""
##########################################################
import pandas as pd
from IPython.display import display

from .the_globals import *
from .ME_stats import get_rss, get_fvu, get_bic

######################## FITTING #########################

def get_fits(ts, sigma_is, max_nk, alpha_F, verbose):
    '''
    Hindering fits for T_Series ts; called by get_trend()

    Parameters
    ----------
    ts : T_Series object
    sigma_is : str
        Detrmines the data error weights (sigma).
        sigma_is = 'data' implies sigma proportional
        to the data. This is the default in the calling
        function. When None, sigma = 1 for all data points
    max_nk : list of integers
        Limits on the number of k-values searched to get the best
        hindering term. The list length determines the number of
        terms attempted. The highest k in the search for best single
        term is max_nk[0]. When this sth has a smaller RSS than the
        logistic, add a term as follows: get the smallest RSS for
        k = [1, k1], then the best RSS for k = [2, k2] If [2, k2] is
        better, continue to get the best [k1, k2] and F-test against
        the sth solution. If the 2-term fit passes the test,
        continue to a 3rd term and so on. The number of terms in the
        search is limited by the length of max_nk. The length of the
        search in the i-th step is limited by max_nk[i]. The default
        [10, 10] implies a maximum of two hindering terms, each
        limited to k no larger than 10.
    alpha_F : float
        Threshold p-value for F-test; default is 0.01
    verbose : int or bool
        Output control:
           0 -- nothing (default)
           1 -- only summary of best fit
           2 -- also progress on each fitting step
                and comparison of all fits

    returns
    -------

    fits : list of Fit instances; 1st one is the best fit

    Notes
    -----
    Produces a list of all the fits to the data with different
    fitting functions. The fits are obtained during the search
    for best fit. Each list member is an instance of the class
    Fit, defined below.

    The best-fit search starts with the exponential, logistic
    and single-term hindering (sth) fits, finding the sth with
    best-fitting k. The best fitting among these 3 is the one
    with smaller BIC, which for logistic vs sth is simply the
    smaller rss error. When there are fits for both exponential
    and sth, they are checked against each other with the F-test.
    When the overall best fit is sth, the search continues to
    multi-term hindering, with terms added one by one as long as
    they improve the fit according to the F-test.

    Examples
    --------
    Typical usage with fully detailed output:

    >>> ts = T_Series(info, data, verbose=True)
    >>> ts.fits = ts.get_fits(verbose=2)

    The second command adds to ts the list of all fits,
    whose first entry is the best fit.

    '''
    t = ts.time
    data = ts.data
    dates = ts.dates
    info =  ts.info
    r_spline = ts.r.spline

    q, sigma, Qm = setup4fits(data, sigma_is, verbose=verbose)
    fits_prt = verbose > 1 # output control

    fits = []
    # Get fits for exp, logistic
    for mode_id in mode_list[:2]:
        fit = fit4(mode_id, q, t, r_spline,
                     sigma, Qm, verbose=fits_prt)[0]
        if fit: fits.append(fit)
    # and single-term hindering
    fit_h1, popt_dict1 = fit4('sth', q, t, r_spline,
                              sigma, Qm, max_nk[0],
                              verbose=fits_prt)
    if fit_h1: fits.append(fit_h1)
    if fits == []: return

    # Best fit from minimum BIC:
    bf_ind = np.argmin([fit.bic for fit in fits])
    # F-test takes precedence when applicable:
    tags = [fit.mode.mode_id for fit in fits]
    if 'exp' in tags and 'sth' in tags:
        ind_exp = tags.index('exp')
        ind_sth = tags.index('sth')
        if bf_ind in (ind_exp, ind_sth): #F-test them
            bf = f_test(
                       fits[ind_exp],
                       fits[ind_sth],
                       Qm*q, Qm*sigma,
                       verbose=fits_prt
                       )
            bf_ind = tags.index(bf.mode.mode_id)

    if tags[bf_ind] == 'sth': #check multi-term hindering
        fits, bf_ind = multi_term(fits, bf_ind, popt_dict1,
                          q, t, r_spline,
                          sigma, Qm, max_nk,
                          alpha_F, verbose=fits_prt)
    bf = fits[bf_ind]
    if bf != fits[0]: # move it to top of list
        fits.remove(bf)
        fits.insert(0, bf)

    # Need to preserve the info on how the fits
    # were obtained, so add sigma_is and alpha_F
    # to fit properties:
    for i, fit in enumerate(fits):
        fits[i].sigma_is = sigma_is
        fits[i].alpha_F = alpha_F
        # Add also yh when applicable:
        if fit.mode.mode_id == 'exp': continue
        if yh := t2year(fits[i].pars.th, dates, info.t_unit):
            fits[i].pars.yh = yh

    if verbose:
        bf_output(bf, t, data)
        if np.argmin([fit.bic for fit in fits])!= 0:
            print(f'>>> Note: Best fit is not bic minimum')
        if len(fits) > 1 and verbose > 1:
            output_all(fits, t, info)

    return fits


def setup4fits(data, sigma_is, verbose=False):
    ''' Prepare the data for fitting
    To improve the numerics, fitting is done with the
    scaled variable q = data/Qm, where Qm is the data median.
    Fitting methodology and scaling are described in
    Blueprint.pdf in directory Blueprint.

    sigma_is controlls the data error estimates
    '''
    from statistics import median
    Qm = median(data)
    q = data/Qm
    if sigma_is is None:
        sigma = None
        sig = 'sigma = 1'
    elif sigma_is == 'data':
        sigma = q.copy()
        sig = '(1 - model/data)**2'
    if verbose: print(f'\nFitting done minimizing SUM{sig}')

    return q, sigma, Qm


def fit_it(mode, q, t, p0, sigma, verbose=False):
    '''
    Fit the data with the mode function using curve_fit
    Returns the best-fit parameters in popt,
    their covariance matrix in pcov, the model for q
    and its rss, fvu
    '''
    from modeling.utils import fitter
    nothing = 5*(None,)
    popt, pcov, model, _ = fitter(mode.func, t, q,
                                  p0=p0,
                                  bounds=mode.bounds,
                                  sigma=sigma,
                                 )
    if popt is None: return nothing

    rss = get_rss(q, model, sigma)
    fvu = get_fvu(q, model, weights=1./sigma**2)[0]
    if verbose:
        s = f'rss = {rss:.2E}, fvu = {fvu:.2E}; popt: '
        s += ', '.join([f'{p:8.2E}' for p in popt])
        print(s)
    return popt, pcov, model, rss, fvu


def p04(mode_id, q, t, r_spline):
    ''' Initial parameter guesses for exp, logist & sth

    #mode_id = 'Junk' # for testing
    if not check_mode(mode_id, mode_list[:-1]):
        sys.exit('***Execution terminated')
    '''

    r0 = r_spline[0]
    if mode_id == 'exp': return [r0, q[0]]

    mask = np.where(r_spline/r0 <= 0.5)[0]
    ind = mask[0] if mask.any() else -1
    qh = q[ind]
    xh = r0*t[ind]
    return [r0, qh, xh]


def fit4(mode_id, q, t, r_spline, sigma, Qm,
         nk=None,
         k_in=None, popt_in=None,
         verbose=False):
    '''Best fit for mode_id
       returns a Fit object and dict of solution parameters
       used for initial gusses for the subsequent addition of
       more hindering terms
    '''
    nothing = 2*(None, )
    '''
    if not check_mode(mode_id): return nothing
    '''

    mode = Mode(mode_id)
    if verbose: print(f'\nModeling with {mode.desc}, {mode.pars_info}:')
    if mode_id in mode_list[:2]: # straight to the fitter
        popt, pcov, model, rss, fvu = fit_it(mode, q, t,
                                      p04(mode_id, q, t, r_spline),
                                      sigma,
                                      verbose=verbose
                                      )
        if popt is None:  return nothing

        popt_dict = None # nothing more to add
        if verbose: output_popt(mode_id, popt)
    else:
        if mode_id == 'sth':
            p_in = p04(mode_id, q, t, r_spline)
            klist_in = []
        else:
            p_in = popt_in.copy()
            klist_in = ([k_in] if isinstance(k_in, int) else
                        list(k_in)
                       )
        mode, model, popt, pcov, rss, fvu, popt_dict = best_h_term(mode_id,
                                                       p_in, klist_in, nk,
                                                       q, t, r_spline, sigma,
                                                       verbose=verbose)
        if popt is None:  return nothing

    dev = 1 - model/q
    #reinstate scale from q to Q
    popt[1] *= Qm
    for i in range(len(popt)):
        pcov[1,i] *= Qm
        pcov[i,1] *= Qm
    model *= Qm
    return Fit(mode, model, rss, fvu, popt, pcov, dev), popt_dict


def best_h_term(mode_id, p_in, klist_in, nk,
                q, t, r_spline, sigma,
                verbose=False):
    '''Procedure to determine the best-fitting hindering term
    Increase k one-by-one to find minimum RSS
    To avoid runaway, nk is an upper limit to number of
    steps attempted
    '''
    nothing = 7*(None,)

    if mode_id == 'sth':
        k0 = 1
        p0 = p_in
    else:
        k0 = klist_in[-1] + 1
        p0 = list(p_in) + [0]   #add new term with a = 0

    kmax = nk + k0 - 1
    # initial placeholders
    mode = Mode(mode_id)
    kb = k0
    rss = np.inf
    fvu = None
    popt = None
    pcov = None
    model = None
    # storage for what will be initial guesses
    # when adding a hindering term:
    popt_dict = {}
    for k in range(k0, kmax+1):
        if mode_id == 'sth':
            mode_ = Mode(mode_id, k=k)
        else:
            k_list = klist_in + [k] #powers of the new series expansion
            mode_ = Mode(mode_id, k_list=k_list)
        if verbose: print(f'{mode_.k_info}:', end=' ')
        popt_, pcov_, model_, rss_, fvu_ = fit_it(mode_, q, t,
                                                  p0, sigma,
                                                  verbose=verbose)
        if popt_ is None: continue

        if rss_ >= rss:  # error increases; done
            break
        else:            # error decreases; continue to next k
            kb = k
            mode = mode_
            popt = popt_
            pcov = pcov_
            model = model_
            rss = rss_
            fvu = fvu_
            # and store the solution:
            key = (k if mode_id == 'sth' else
                   tuple(k_list)     # dict keys must be immutable
                  )
            val = popt_.copy()
            if mode_id == 'sth': # add initial guess for a_k in 1st mth model
                val = np.append(val, 1)
            popt_dict[key] = val
    if popt is None:
        return nothing

    if verbose:
        print(f'***rss minimum = {rss:.3E}: {mode.k_info}')
        output_popt(mode_id, popt)
        if mode_id == 'mth':
            s = ', '.join([f'a({k}) = {a:6.2E}'
                     for (k, a) in zip(mode.k_list, popt[3:])])
            print(s)
        if kb == kmax:
            print(f'>>>>NOTE: Limit of max_nk = {nk} was reached')

    return mode, model, popt, pcov, rss, fvu, popt_dict


def multi_term(fits, bf_ind, popt_dict_in, q, t, r_spline,
               sigma, Qm, max_nk, alpha_F, verbose=False):
    ''' Multi-term fits

    Invoked only when fits[bf_ind] is single-term hindering
    '''

    fit_h1 = fits[bf_ind]
    popt_dict_list = [popt_dict_in]

    for number_of_terms in range(1, len(max_nk)):
        rss = np.inf
        fit_h2 = None
        popt_dict_list_ = []
        for popt_dict in popt_dict_list:
            for k_in, popt_in in popt_dict.items():
                if verbose: print(f'\nAdding term to k = {k_in}:', end=' ')
                fit_, popt_dict_ = fit4('mth', q, t, r_spline,
                                        sigma, Qm,
                                        nk=max_nk[number_of_terms],
                                        k_in=k_in, popt_in=popt_in,
                                        verbose=verbose
                                       )
                if fit_.rss >= rss: # error increases; done
                    break
                else:               # error decreases; continue to next k_list
                    popt_dict_list_.append(popt_dict_)
                    fit_h2 = fit_
                    rss = fit_.rss
        if not fit_h2:
            if verbose: print('No term added')
            break

        if verbose:
            print(f'\nBest next term is {fit_h2.mode.desc}; rss = {fit_h2.rss:.2E}')
            output_popt(fit_h2.mode.mode_id, fit_h2.popt,
                        Qh_not_qh=True
                       )
            if fit_h2.mode.mode_id == 'mth':
                output_wk(fit_h2)
        fits += [fit_h2]
        # F-testing fit_h2 vs fit_h1
        best_fit = f_test(fit_h1, fit_h2, Qm*q, Qm*sigma,
                          verbose=verbose, alpha_max=alpha_F)
        if best_fit == fit_h1: # no need to add more terms
            if verbose:
                print('No significant improvement from added term')
            break
        else: # continue adding terms
            if verbose:
                print('Significant improvement from added term')
            fit_h1 = fit_h2
            popt_dict_list = popt_dict_list_

    bf_ind = fits.index(best_fit)
    return fits, bf_ind



def f_test(fit1, fit2, data, sigma, alpha_max=0.01, verbose=False):
    '''F-testing two hindering solutions
    '''
    from .ME_stats import f_stat

    fits = [fit1, fit2]
    model = [fit.model for fit in fits]
    p = [fit.mode.n_pars for fit in fits]

    F, alpha, Fcrit = f_stat(data,
                             model[0], p[0], model[1], p[1],
                             sigma, alpha=alpha_max)
    bf = fits[1 if alpha < alpha_max else 0]

    if verbose:
        s = 'F-testing fits with '
        s+= ' and '.join(
                         [f'{fit.mode.label} (p = {p_})'
                          for fit, p_ in zip(fits, p)
                         ]
                        )
        print('\n****'+s+':')
        s = f'{Fcrit = :.4G} for alpha = {alpha_max:.4G}\n'
        s+= f'F = {F:.4G}, alpha = {alpha:.4G}; '
        s+= f'fit with {bf.mode.label} is preferred'
        print(s)
    return bf


def output_popt(mode_id, popt, Qh_not_qh=False):
    '''Output the fit parameters
    '''
    s = f'gu = {popt[0]:.2%}'
    Qh = ('Qh' if Qh_not_qh else
          'qh'
         )
    s += f', {Qh} = {popt[1]:.2E}'
    if mode_id == 'exp':
        print(s)
        return
    s += f', xh = {popt[2]:.2E}'
    print(s)


def output_wk(fit):
    '''output the weights of multi-terms fit

    No error estimates - the covariance matrix
    is meaningless for a_k because rather than
    independent of each other, they are bound
    by the normalization constraint
    '''
    if fit.mode.mode_id != 'mth': return
    a_list = fit.popt[3:] # the a-coefficients
    w = sum(a_list)       # normalization factor
    print(' k   w_k')
    for k, a in zip(fit.mode.k_list, a_list):
        print(f'{k:2} {a/w:8.2E}')


def bf_output(bf, t, Q):
    '''Output for best-fit
    '''
    from .ME_stats import trend_test
    pm = chr(177) # plus/minus symbol

    print(f'\n***** Best-fit model is {bf.mode.desc}')
    bf_id = bf.mode.mode_id
    pars = bf.pars
    perr = np.sqrt(np.diag(bf.pcov))
    ru = pars.ru
    s = f'gu = {ru:.2%}{pm}{perr[0]:.2%}'
    Qh = pars.Qh
    sQh = ('Q1' if bf_id == 'exp' else
           'Qh'
          )
    s += f'; {sQh} = {Qh:.2E}{pm}{perr[1]:.2E}'
    if bf_id == 'exp':
        print(s)
    else:
        th = pars.th
        s+= f'; th = {th:.2f}{pm}{perr[2]/ru:.2E}'
        if hasattr(pars, 'yh'): s+= f', yh = {int(pars.yh)}'
        print(s)
        s = f'gu(t - th) = [{-ru*th:.2f}, {ru*(t[-1] - th):.2f}], '
        s+= f'Q/Qh = [{Q[0]/Qh:.2G}, {Q[-1]/Qh:.2G}]'
        print(s)
    if bf_id == 'mth': output_wk(bf)

    print('Fluctuations = (data - fit)/data:', end=' ')
    m_fluc = f'{np.mean(bf.dev):.2E}'
    s_fluc = f'{np.std(bf.dev):.2E}'
    print(f'mean = {m_fluc}, std = {s_fluc}')
    trend = trend_test(bf.dev, 'trend', data_id='fluctuations',
            verbose=True)


def output_all(the_fits, t, info):
    ''' Summary comparison of the fits
    '''
    print('\nComparing all fits:')
    unit = info.unit
    t_unit = info.t_unit
    # Sort the fits by BIC:
    idx = np.argsort([fit.bic for fit in the_fits])
    fits = [the_fits[i] for i in idx]

    index = [fit.mode.label for fit in fits]
    n = len(index)
    [fluc, bic, gu, Qh, xh, th, yh] = [
        n*[''], n*[''], n*[''], n*[''],
        n*[''], n*[''], n*[''],
        ]
    note = ''
    for i, fit in enumerate(fits):
        pars = fit.pars
        fluc[i] = f'{np.mean(fit.dev):.2E}'
        bic[i] = f'{fit.bic:,.2f}'
        ru = pars.ru
        gu[i] = f'{ru:.2%}'
        if (mode_id:=fit.mode.mode_id) == 'exp':
            continue
        #add hindering properties:
        Qh[i] = f'{pars.Qh:.2E}'
        if mode_id == 'logist':
            K = f'{unit}{pars.K:.2E}'
            note = f'\nlogistic: K = {K}'
        xh[i] = f'{pars.xh:.2f}'
        th[i] = f'{pars.th:.2f}'
        if hasattr(pars, 'yh'): yh[i] = pars.yh

    output_data = {
                   'fluc mean': fluc,
                   'bic': bic,
                   f'gu(/{t_unit})': gu,
                  }
    if any(_ for _ in Qh):
        sQh = 'Qh'
        if unit: sQh += f'({unit})'
        output_data.update(
                  {
                   sQh: Qh,
                   'xh': xh,
                   f'th({t_unit})': th
                  })
        if any(_ for _ in yh):
            output_data['yh'] = yh

    display(pd.DataFrame(index=index,
                         data=output_data)
               )
    print(note)


def t2year(t, years, t_unit):
    '''Convert t to year when relevant
    '''
    if (t_unit not in yr_list or
        years is None or
        np.isnan(t).any()
       ):
       return None
    yrs = np.atleast_1d(years[0] + n_yrs[t_unit]*t).astype(int)
    return yrs if len(yrs) > 1 else yrs[0]

#######################################################################
#                                                                     #
#   Produce a Fit object, containing properties and methods for       #
#   exponential, logistic and hindering fits                          #
#                                                                     #
#######################################################################


@dataclass
class Fit:
    '''Properties and methods of a successful fit

    properties added to the input parameters:
        pars --- object containing the solution parameters
        bic  --- The fit's BIC

    Methods
    -------
        model_calc --- model calculation for the fit properties
        r_model --- calculates the model growth rate
        de_hinder --- model with the hindering effect  removed
    '''
    mode:   object      # mode of the fitting function
    model:  np.ndarray
    rss:    float
    fvu:    float
    popt:   np.ndarray  # parameters of fitting function
    pcov:   np.ndarray  # covariance matrix of the parameters
    dev:    np.ndarray  # fluctuations: 1 - model/data

    def __post_init__(self):
        self.bic = get_bic(self.rss,
                           len(self.model),
                           self.mode.n_pars
                           )
        self.pars = self.popt2pars()

    def popt2pars(self):
        ''' Convert fitting function pars to solution parameters
        '''
        popt, mode = self.popt, self.mode
        ru, Qh = popt[:2]
        pars = SN(
                  ru = ru,
                  Qh = Qh
                 )
        if (mode_id := mode.mode_id) == 'exp':
            return pars

        #additional pars for hindering modes:
        xh, *a = popt[2:]
        pars.xh = xh
        pars.th = xh/ru
        if mode_id == 'logist':
            pars.K = 2*Qh
        if mode_id == 'mth':
            # a_k to w_k dict
            a_list = list(a)
            w = sum(a_list)
            w_list = [_/w for _ in a_list]
            w_k = dict(zip(mode.k_list, w_list))
            pars.w_k = w_k

        return pars


    def model_calc(self, t):
        '''Calculate model results for the fit
        '''
        return self.mode.func(t, *self.popt)


    def r_model(self, Q_model):
        '''Growth rate model calculation
        '''
        from .h_funcs import f_hind

        mode_id = self.mode.mode_id
        '''
        #mode_id = 'Junk' # for testing
        if not check_mode(mode_id):
            print(X+'Could not calculate growth rate')
            return None
        '''

        pars = self.pars
        ru = pars.ru

        if mode_id == 'exp':
            r = (ru if is_number(Q_model) else
                 ru*np.ones_like(Q_model)
                )
        elif mode_id == 'logist':
            r = ru*(1 - Q_model/pars.K)
        elif mode_id == 'sth':
            r = ru/(1 + (Q_model/pars.Qh)**self.mode.k)
        elif mode_id == 'mth':
            r = ru/(1 + f_hind(Q_model/pars.Qh, pars.w_k))

        return r


    def de_hinder(self, t):
        model = self.model
        r0 = self.pars.ru
        return model[0]*exp(r0*t)







#####################################################################################
