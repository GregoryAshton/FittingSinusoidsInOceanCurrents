""" """

import numpy as np
import BDATools as BDA
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("AMOCdata.xls")
time = df['day no.'].values * 86400.
yAMOC = df.AMOC.values
yGulf = df['Gulf Stream'].values
yEkman = df.Ekman.values
yUMO = df.UMO.values

rangeyGulf = max(yGulf)-min(yGulf)
rangeyEkman = max(yEkman)-min(yEkman)
rangeyUMO = max(yUMO)-min(yUMO)

ranget = np.max(time)-np.min(time)

params = {'AprimeGulf': {'prior':
                     {'type': 'norm', 'loc': 0,
                      'scale': 0.01*abs(rangeyGulf/ranget)},
                     'symbol': r"$\dot{A}^\mathrm{G}$",
                     'unit': 'Sv/s',
                     },
          'AprimeEkman': {'prior':
                     {'type': 'norm', 'loc': 0,
                      'scale': 0.01*abs(rangeyEkman/ranget)},
                     'symbol': r"$\dot{A}^\mathrm{E}$",
                     'unit': 'Sv/s',
                     },
          'AprimeUMO': {'prior':
                     {'type': 'norm', 'loc': 0,
                      'scale': 0.01*abs(rangeyUMO/ranget)},
                     'symbol': r"$\dot{A}^\mathrm{U}$",
                     'unit': 'Sv/s',
                     },
          }

parent_model_name = "BasicSinusoid"
parent_keys = ['P',
               'y0Gulf', 'y0Ekman', 'y0UMO',
               'yprime0Gulf', 'yprime0Ekman', 'yprime0UMO',
               'A0Gulf', 'A0Ekman', 'A0UMO',
               'phi0Gulf', 'phi0Ekman', 'phi0UMO',
               'sigmaGulf', 'sigmaEkman', 'sigmaUMO']


ParentParams = BDA.ReadPickle(parent_model_name,
                              dtype="DataDictionary")['params']
for param in parent_keys:
    params[param] = ParentParams[param]

param_keys = ['P',
              'y0Gulf', 'y0Ekman', 'y0UMO',
              'yprime0Gulf', 'yprime0Ekman', 'yprime0UMO',
              'A0Gulf', 'A0Ekman', 'A0UMO',
              'AprimeGulf', 'AprimeEkman', 'AprimeUMO',
              'phi0Gulf', 'phi0Ekman', 'phi0UMO',
              'sigmaGulf', 'sigmaEkman', 'sigmaUMO']
model_name = "BasicSinusoidAmplitudeDecay"
cargs = BDA.SetupHelper(model_name)

ntemps = 10
nburn0 = 10000
nburn = 1000
nprod = 1000
thin = 5

scatter_val = 1e-3
nwalkers = 100


def SignalModelGulf(time, P,
                y0Gulf, y0Ekman, y0UMO,
                yprime0Gulf, yprime0Ekman, yprime0UMO,
                A0Gulf, A0Ekman, A0UMO,
                AprimeGulf, AprimeEkman, AprimeUMO,
                phi0Gulf, phi0Ekman, phi0UMO,
                sigmaGulf, sigmaEkman, sigmaUMO
                ):
    return y0Gulf + yprime0Gulf*time + (A0Gulf+AprimeGulf*time)*np.sin(2*np.pi*time/P + phi0Gulf)

def SignalModelEkman(time, P,
                y0Gulf, y0Ekman, y0UMO,
                yprime0Gulf, yprime0Ekman, yprime0UMO,
                A0Gulf, A0Ekman, A0UMO,
                AprimeGulf, AprimeEkman, AprimeUMO,
                phi0Gulf, phi0Ekman, phi0UMO,
                sigmaGulf, sigmaEkman, sigmaUMO):
    return y0Ekman + yprime0Ekman*time + (A0Ekman+AprimeEkman*time)*np.sin(2*np.pi*time/P + phi0Ekman)

def SignalModelUMO(time, P,
                y0Gulf, y0Ekman, y0UMO,
                yprime0Gulf, yprime0Ekman, yprime0UMO,
                A0Gulf, A0Ekman, A0UMO,
                AprimeGulf, AprimeEkman, AprimeUMO,
                phi0Gulf, phi0Ekman, phi0UMO,
                sigmaGulf, sigmaEkman, sigmaUMO):
    return y0UMO + yprime0UMO*time + (A0UMO+AprimeUMO*time)*np.sin(2*np.pi*time/P + phi0UMO)

x = [time, time, time]
y = [yGulf, yEkman, yUMO]
SignalModel = [SignalModelGulf, SignalModelEkman, SignalModelUMO]

DD = BDA.GetData(
    x, y, SignalModel, model_name=model_name, params=params,
    param_keys=param_keys, nburn0=nburn0, thin=thin,
    ntemps=ntemps, nwalkers=nwalkers, nburn=nburn, nprod=nprod,
    scatter_val=scatter_val)

samples = DD['samples']
lnprobs = DD['lnprobs']
symbols = [params[key]['symbol'] for key in param_keys]
units = [params[key]['unit'] for key in param_keys]

use_only = ['AprimeGulf', 'AprimeEkman', 'AprimeUMO']
BDA.PlotCorner(samples, params, param_keys, model_name, cargs=cargs,
               label_offset=0.1, use_only=use_only)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(3.4, 4.5))
ax1 = BDA.PlotWithData(time, yGulf, samples, SignalModelGulf,
                       cargs=cargs, MJD_start_date=0,
                       ax=ax1, noise_index=-3, markersize=0.5,
                       save=False, tick_size=8, nsamples="MLE",
                       lnprobs=lnprobs)
ax1.set_xlabel("")
ax1.set_ylabel("Gulf")
ax2 = BDA.PlotWithData(time, yEkman, samples, SignalModelEkman,
                       cargs=cargs, MJD_start_date=0,
                       ax=ax2, noise_index=-2, markersize=0.5,
                       save=False, tick_size=8, nsamples="MLE",
                       lnprobs=lnprobs)
ax2.set_xlabel("")
ax2.set_ylabel("Ekman")
ax3 = BDA.PlotWithData(time, yUMO, samples, SignalModelUMO,
                       cargs=cargs, MJD_start_date=0,
                       ax=ax3, noise_index=-1, markersize=0.5,
                       save=False, tick_size=8, nsamples="MLE",
                       lnprobs=lnprobs)
ax3.set_ylabel("UMO")
ax3.set_xlabel("days")
fig.tight_layout()
fig.savefig("img/{}_PosteriorFit.png".format(model_name))

BDA.WriteEvidenceToFile(model_name, DD)

BDA.GenericDistributionTable(params, param_keys, model_name,
                             distribution="prior")
