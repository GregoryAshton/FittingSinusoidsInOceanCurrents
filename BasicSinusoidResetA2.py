""" """

import numpy as np
import BDATools as BDA
import pandas as pd

df = pd.read_excel("AMOCdata.xls")
time = df['day no.'].values * 86400.
y = df.AMOC.values

miny = np.min(y)
maxy = np.max(y)
rangey = maxy-miny
ranget = np.max(time)-np.min(time)
params = {'A1': {'prior':
                 {'type': 'unif', 'lower': 0, 'upper': rangey},
                 'symbol': r"$A_1$",
                 'unit': '',
                 },
          'T1': {'prior':
                 {'type': 'unif', 'lower': 0, 'upper': np.max(time)},
                 'symbol': r"$T_1$",
                 'unit': 'rad',
                 'rescale': ((86400)**-1, "days"),
                 },
          'A2': {'prior':
                 {'type': 'unif', 'lower': 0, 'upper': rangey},
                 'symbol': r"$A_2$",
                 'unit': '',
                 },
          'T2': {'prior':
                 {'type': 'unif', 'lower': 0, 'upper': np.max(time)},
                 'symbol': r"$T_2$",
                 'unit': 'rad',
                 'rescale': ((86400)**-1, "days"),
                 },
          }

parent_model_name = "BasicSinusoid"
parent_keys = ['y0', 'yprime0', 'A0', 'P', 'psi0', 'sigma']
ParentParams = BDA.ReadPickle(parent_model_name,
                              dtype="DataDictionary")['params']
for param in parent_keys:
    params[param] = ParentParams[param]

param_keys = ['y0', 'yprime0', 'A0', 'A1', 'T1', 'A2', 'T2', 'P', 'psi0',
              'sigma']
model_name = "BasicSinusoidResetA2"
cargs = BDA.SetupHelper(model_name)

ntemps = 10
nburn0 = 2000
nburn = 5000
nprod = 1000
thin = [1, 10, 1]
scatter_val = 1e-3
nwalkers = 100


def SignalModel(time, y0, yprime0, A0, A1, T1, A2, T2, P, psi0, sigma):

    Ts = np.array([T1, T2])
    if np.array_equal(Ts, np.sort(Ts)) is False:
        return np.zeros(len(time))

    base = y0 + yprime0 * time

    A = np.zeros(len(time)) + A0
    A[time > T1] = A1
    A[time > T2] = A2
    TF = A*np.sin(2*np.pi*time/P + psi0)

    return base + TF

DD = BDA.GetData(
    time, y, SignalModel, model_name=model_name, params=params,
    param_keys=param_keys, nburn0=nburn0,
    ntemps=ntemps, nwalkers=nwalkers, nburn=nburn, nprod=nprod,
    scatter_val=scatter_val, thin=thin)

samples = DD['samples']
lnprobs = DD['lnprobs']
symbols = [params[key]['symbol'] for key in param_keys]
units = [params[key]['unit'] for key in param_keys]

BDA.PlotWithData(time, y, samples, SignalModel,
                 cargs=cargs, MJD_start_date=0,
                 model_name = model_name, noise_index=-1,
                 save=True, tick_size=8, nsamples="MLE",
                 lnprobs=lnprobs, markersize=0.5)
BDA.PlotWithDataAndCorner(time, y, model_name, DD, SignalModel, cargs,
                          markersize=0.5)

BDA.WriteEvidenceToFile(model_name, DD)
