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
params = {'y0': {'prior':
                 {'type': 'unif', 'lower': miny, 'upper': maxy},
                 'symbol': r"$y_0$",
                 'unit': 's',
                 },
          'yprime0': {'prior':
                      {'type': 'norm', 'loc': 0, 'scale': abs(rangey/ranget)},
                      'symbol': r"$y_0$",
                      'unit': 's',
                      },
          'A0': {'prior':
                 {'type': 'unif', 'lower': 0, 'upper': rangey},
                 'symbol': r"$A_0$",
                 'unit': '',
                 },
          'A1': {'prior':
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
          'P': {'prior':
                {'type': 'unif', 'lower': 0, 'upper': 0.2*(ranget)},
                'symbol': r"$P$",
                'rescale': ((86400*356.25)**-1, "yrs"),
                'unit': '',
                },
          'psi0': {'prior':
                   {'type': 'unif', 'lower': 0, 'upper': 4*np.pi},
                   'symbol': r"$\psi_0$",
                   'unit': 'rad'
                   },
          'sigma': {'prior':
                    {'type': 'unif', 'lower': 0, 'upper': rangey},
                    'symbol': r"$\sigma_{\dot{\nu}}$",
                    'unit': '$\mathrm{s}^{-2}$'
                    }}

param_keys = ['y0', 'yprime0', 'A0', 'A1', 'T1', 'P', 'psi0', 'sigma']
model_name = "BasicSinusoidResetA1"
cargs = BDA.SetupHelper(model_name)

ntemps = 10
nburn0 = 1000
nburn = 1000
nprod = 1000

scatter_val = 1e-3
nwalkers = 100


def SignalModel(time, y0, yprime0, A0, A1, T1, P, psi0, sigma):
    base = y0 + yprime0 * time
    A = np.zeros(len(time))
    A = np.zeros(len(time)) + A0
    A[time > T1] = A1
    TF = A*np.sin(2*np.pi*time/P + psi0)
    return base + TF

DD = BDA.GetData(
    time, y, SignalModel, model_name=model_name, params=params,
    param_keys=param_keys, nburn0=nburn0,
    ntemps=ntemps, nwalkers=nwalkers, nburn=nburn, nprod=nprod,
    scatter_val=scatter_val)

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
