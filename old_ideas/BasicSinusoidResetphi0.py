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
          'A': {'prior':
                {'type': 'unif', 'lower': 0, 'upper': rangey},
                'symbol': r"$A$",
                'unit': '',
                },
          'P': {'prior':
                {'type': 'unif', 'lower': 0, 'upper': 0.2*(ranget)},
                'symbol': r"$P$",
                'rescale': ((86400*356.25)**-1, "yrs"),
                'unit': '',
                },
          'psi01': {'prior':
                    {'type': 'unif', 'lower': 0, 'upper': 4*np.pi},
                    'symbol': r"$\psi_0^1$",
                    'unit': 'rad'
                    },
          'psi02': {'prior':
                    {'type': 'unif', 'lower': 0, 'upper': 4*np.pi},
                    'symbol': r"$\psi_0^2$",
                    'unit': 'rad'
                    },
          'T': {'prior':
                {'type': 'unif', 'lower': 0, 'upper': np.max(time)},
                'symbol': r"$T$",
                'unit': 'rad',
                'rescale': ((86400*356.25)**-1, "yrs"),
                },
          'sigma': {'prior':
                    {'type': 'unif', 'lower': 0, 'upper': rangey},
                    'symbol': r"$\sigma_{\dot{\nu}}$",
                    'unit': '$\mathrm{s}^{-2}$'
                    }}

param_keys = ['y0', 'yprime0', 'A', 'P', 'psi01', 'psi02', 'T', 'sigma']
model_name = "BasicSinusoidResetphi0"
cargs = BDA.SetupHelper(model_name)

ntemps = 1
nburn0 = 500
nburn = 500
nprod = 500

scatter_val = 1e-3
nwalkers = 100


def SignalModel(time, y0, yprime0, A, P, psi01, psi02, T, sigma):
    base = y0 + yprime0 * time
    T1 = A*np.sin(2*np.pi*time/P + psi01)
    T2 = A*np.sin(2*np.pi*time/P + psi02)
    TF = np.zeros(len(time))
    TF[time < T] = T1[time < T]
    TF[time >= T] = T2[time >= T]
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
