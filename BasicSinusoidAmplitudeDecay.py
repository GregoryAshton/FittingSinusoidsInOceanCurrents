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

params = {'Aprime': {'prior':
                     {'type': 'norm', 'loc': 0,
                      'scale': 0.01*abs(rangey/ranget)},
                     'symbol': r"$\dot{A}$",
                     'unit': 'Sv/s',
                     },
          }

parent_model_name = "BasicSinusoid"
parent_keys = ['y0', 'yprime0', 'A0', 'P', 'psi0', 'sigma']
ParentParams = BDA.ReadPickle(parent_model_name,
                              dtype="DataDictionary")['params']
for param in parent_keys:
    params[param] = ParentParams[param]

param_keys = ['y0', 'yprime0', 'A0', 'Aprime', 'P', 'psi0', 'sigma']
model_name = "BasicSinusoidAmplitudeDecay"
cargs = BDA.SetupHelper(model_name)

ntemps = 10
nburn0 = 1000
nburn = 1000
nprod = 1000

scatter_val = 1e-3
nwalkers = 100


def SignalModel(time, y0, yprime0, A0, Aprime, P, phi0, sigma):
    return y0 + yprime0*time + (A0 + Aprime*time)*np.sin(2*np.pi*time/P + phi0)

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
                 model_name=model_name, noise_index=-1,
                 save=True, tick_size=8, nsamples="MLE",
                 lnprobs=lnprobs, markersize=0.5)
BDA.PlotWithDataAndCorner(time, y, model_name, DD, SignalModel, cargs,
                          markersize=0.5,  xlabel="days",
                          title=": $y(t)=y_0+(A_0 + \dot{A}t)\sin(2\pi t /P + \psi_0)$")

BDA.WriteEvidenceToFile(model_name, DD)
