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
          'A1': {'prior':
                 {'type': 'unif', 'lower': 0, 'upper': rangey},
                 'symbol': r"$A$",
                 'unit': '',
                 },
          'P1': {'prior':
                 {'type': 'unif', 'lower': 0, 'upper': 0.2*(time[-1]-time[0])},
                 'symbol': r"$f$",
                 'unit': '',
                 },
          'psi01': {'prior':
                    {'type': 'unif', 'lower': 0, 'upper': 2*np.pi},
                    'symbol': r"$\psi_0$",
                    'unit': 'rad'
                    },
          'A2': {'prior':
                 {'type': 'unif', 'lower': 0, 'upper': rangey},
                 'symbol': r"$A$",
                 'unit': '',
                 },
          'P2': {'prior':
                 {'type': 'unif', 'lower': 0, 'upper': 0.2*(time[-1]-time[0])},
                 'symbol': r"$f$",
                 'unit': '',
                 },
          'psi02': {'prior':
                    {'type': 'unif', 'lower': 0, 'upper': 2*np.pi},
                    'symbol': r"$\psi_0$",
                    'unit': 'rad'
                    },
          'sigma': {'prior':
                    {'type': 'unif', 'lower': 0, 'upper': rangey},
                    'symbol': r"$\sigma_{\dot{\nu}}$",
                    'unit': '$\mathrm{s}^{-2}$'
                    }}

param_keys = ['y0', 'yprime0', 'A1', 'P1', 'psi01', 'A2', 'P2', 'psi02', 'sigma']
model_name = "DoubleSinusoid"
cargs = BDA.SetupHelper(model_name)

ntemps = 10
nburn0 = 1000
nburn = 4000
nprod = 1000

scatter_val = 1e-3
nwalkers = 100


def SignalModel(time, y0, yprime0, A1, P1, phi01, A2, P2, phi02, sigma):
    return y0 + yprime0*time + (A1*np.sin(2*np.pi*time/P1 + phi01)
                                + A2*np.sin(2*np.pi*time/P2 + phi02))

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
