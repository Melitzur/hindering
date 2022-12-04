"""
Contents:
---------
class Var_info:
r_conversion
class Mode:
    desc
    label
    k_info
    pars_info
    n_pars
    func
            _h
            _h
    bounds
"""
#######################################################################

import sys
import numpy as np
import pandas as pd
from numpy import exp, log as ln
from types import SimpleNamespace as SN
from collections import namedtuple
from dataclasses import dataclass, field

from .h_funcs import h, sth, ell

#######################################################################
## Currency unit conversions
#
# From the data directory data_dir get conversion factors:
#
# (USD12, MW_GBP2USD15) = from_pickle(data_dir+'USD.pickle')
#
# USD12 is a dictionary of USD deflator normalized to 2012:
#
#        USD12[2012] = 1
#
# MW_GBP2USD15 is the conversion of Measuring Worth UK
# data (UKGDP_1801-2021.csv) in GBP to USD 2015
#
# See USD_conversions.ipynb in data_dir
#
##################################################

### Variables info:

@dataclass
class Var_info:
    var_name: str
    desc: str
    t_unit: str
    unit: str = ''

gdp = Var_info(var_name='GDP', desc='GDP', t_unit='year', unit='$')
popl = Var_info('POP', desc='Population', t_unit='year')
gdppc = Var_info('GDPPC', 'GDP per Capita', 'year', unit='$')

covid_c = Var_info('cases', 'Covid19 Reported Cases', t_unit='day')
covid_d = Var_info('deaths', 'Covid19 Reported Deaths', t_unit='day')


# To convert growth rate to annual when t_unit is not year
n_yrs = {'year': 1, 'decade': 10, 'quarter': 0.25}
yr_list = list(n_yrs.keys())

def r_conversion(t_unit):
    c = 100               #convert growth rate to percents
    if t_unit in yr_list: #convert to growth rate per year
        c /= n_yrs[t_unit]
        t_unit = 'year'
    return c, t_unit


############# Modeling functions and their modes #############
'''
Modeling functions are identified by a mode_id and accompanying info,
all encapsulated in an instant of class Mode. Fitting is done by
curve_fit, which does not offer the option of non-varying parameters;
its first argument is the independent variable and all other positional
arguments are considered the parameters to vary. This creates two types
of modes.

Static modes:
All the parameters of the exponential and logistic vary during
fitting; these functions do not require any additional parameters.

Dynamic modes:
Hindering sums require the indices k of power-law terms, which are held
fixed at any given step; only the term coefficients vary when curve_fit
searches for the optimal parameters. The workaround this limitation is
to define the function for curve_fit in a 2-step process. For each set
of powers and coefficients, a new function and its mode instance are
created dynamically.
'''

mode_info = namedtuple('mode_info', ['description', 'pars_info'])
the_modes = {
             'exp':    mode_info('exponential', '[ru, q0]'),
             'logist': mode_info('logistic', '[ru, qh, xh]'),
             'sth':    mode_info('single-term hindering', '[ru, qh, xh]'),
             'mth':    mode_info('multi-term hindering', '[ru, qh, xh, a_list]')
            }

mode_list = list(the_modes.keys())


@dataclass
class Mode:
    mode_id: str                    # identifier in mode_list
    k:       int = None                         #for sth only
    k_list:  list = field(default_factory=list) #for mth only


    @property
    def desc(self):
        ''' description of the fitting function
        '''
        s = the_modes[self.mode_id].description
        if (k_info := self.k_info):
            s += f', {k_info}'
        return s

    @property
    def label(self):
        ''' short version of description
        same as mode_id for exp and logist
        '''
        s = self.mode_id
        if (k_info := self.k_info):
            s += f', {k_info}'
        return s

    @property
    def k_info(self):
        ''' for sth and mth
        '''
        if (k := self.k):
            s = f'{k = }'
        elif (k_list := self.k_list):
            s = f'k = {k_list}'
        else:
            s = ''
        return s

    @property
    def pars_info(self):
        '''parameter names of the fitting function
        '''
        return the_modes[self.mode_id].pars_info

    @property
    def n_pars(self):
        ''' Number of free parameters
        '''
        n = (2 if self.mode_id == 'exp' else
             3
            )
        if self.mode_id == 'mth':
            # add number of terms minus normalization constraint:
            n += len(self.k_list) - 1
        return n


    def func(self, t, *args):
        '''the fitting function
        '''
        if (mode_id := self.mode_id) == 'exp':
            (r0, q0) = args
            return q0*exp(r0*t)

        elif mode_id == 'logist':
            (ru, qh, xh) = args
            return qh*ell(ru*t - xh)

        elif mode_id == 'sth':
            (ru, qh, xh) = args
            def _h(x):
                return sth(x, self.k)
            return qh*_h(ru*t - xh)

        elif mode_id == 'mth':
            # ak are the unnormalized expansion coefficients
            (ru, qh, xh, *ak) = args
            def _h(x, ak):
                a_dict = dict(zip(self.k_list, list(ak)))
                return h(x, a_dict)
            return qh*_h(ru*t - xh, ak)


    @property
    def bounds(self):
        '''Bounds on parameter search by curve_fit
           Except for xh, all parameters must be positive
        '''
        upper = np.inf
        lower = [0., 0.]    #ru, qh
        if self.mode_id == 'exp':
            return (lower, upper)

        lower += [-np.inf]  #xh can be negative
        if self.mode_id == 'mth':
            #allow only positive a_k:
            lower += len(self.k_list)*[0.]
        return (lower, upper)

