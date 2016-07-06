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
params = {'y0Gulf': {'prior':
                 {'type': 'unif', 'lower': min(yGulf), 'upper': max(yGulf)},
                 'symbol': r"$y_0^\mathrm{G}$" + '\n',
                 'unit': 'Sv',
                 },
          'yprime0Gulf': {'prior':
                      {'type': 'norm', 'loc': 0, 'scale': rangeyGulf/ranget},
                      'symbol': r"$y_0^\mathrm{G}$" + '\n',
                      'unit': 'Sv/s',
                      },
          'A0Gulf': {'prior':
                 {'type': 'unif', 'lower': 0, 'upper': rangeyGulf},
                 'symbol': r"$A_0^\mathrm{G}$" + '\n',
                 'unit': 'Sv',
                 },
          'sigmaGulf': {'prior':
                    {'type': 'unif', 'lower': 0, 'upper': rangeyGulf},
                    'symbol': r"$\sigma^\mathrm{G}$" + '\n',
                    'unit': 'Sv'
                    },
          'phi0Gulf': {'prior':
                   {'type': 'unif', 'lower': -np.pi, 'upper': np.pi},
                   'symbol': r"$\phi_0^\mathrm{G}$" + '\n',
                   'unit': 'rad'
                   },
          'y0Ekman': {'prior':
                 {'type': 'unif', 'lower': min(yEkman), 'upper': max(yEkman)},
                 'symbol': r"$y_0^\mathrm{E}$" + '\n',
                 'unit': 'Sv',
                 },
          'yprime0Ekman': {'prior':
                      {'type': 'norm', 'loc': 0, 'scale': rangeyEkman/ranget},
                      'symbol': r"$y_0^\mathrm{E}$" + '\n',
                      'unit': 'Sv/s',
                      },
          'A0Ekman': {'prior':
                 {'type': 'unif', 'lower': 0, 'upper': rangeyEkman},
                 'symbol': r"$A_0^\mathrm{E}$" + '\n',
                 'unit': 'Sv',
                 },
          'sigmaEkman': {'prior':
                    {'type': 'unif', 'lower': 0, 'upper': rangeyEkman},
                    'symbol': r"$\sigma^\mathrm{E}$" + '\n',
                    'unit': 'Sv'
                    },
          'phi0Ekman': {'prior':
                   {'type': 'unif', 'lower': -np.pi, 'upper': np.pi},
                   'symbol': r"$\phi_0^\mathrm{E}$" + '\n',
                   'unit': 'rad'
                   },
          'y0UMO': {'prior':
                 {'type': 'unif', 'lower': min(yUMO), 'upper': max(yUMO)},
                 'symbol': r"$y_0^\mathrm{U}$" + '\n',
                 'unit': 'Sv',
                 },
          'yprime0UMO': {'prior':
                      {'type': 'norm', 'loc': 0, 'scale': rangeyUMO/ranget},
                      'symbol': r"$y_0^\mathrm{U}$" + '\n',
                      'unit': 'Sv/s',
                      },
          'A0UMO': {'prior':
                 {'type': 'unif', 'lower': 0, 'upper': rangeyUMO},
                 'symbol': r"$A_0^\mathrm{U}$" + '\n',
                 'unit': 'Sv',
                 },
          'sigmaUMO': {'prior':
                    {'type': 'unif', 'lower': 0, 'upper': rangeyUMO},
                    'symbol': r"$\sigma^\mathrm{U}$" + '\n',
                    'unit': 'Sv'
                    },
          'phi0UMO': {'prior':
                   {'type': 'unif', 'lower': -np.pi, 'upper': np.pi},
                   'symbol': r"$\phi_0^\mathrm{U}$" + '\n',
                   'unit': 'rad'
                   },
          'P': {'prior':
                {'type': 'norm', 'loc': 365*86400, 'scale': 1*86400},
                'symbol': r"$P$" + '\n',
                'rescale': ((86400*365)**-1, "yrs"),
                'unit': 's',
                },
          }

param_keys = ['P',
              'y0Gulf', 'y0Ekman', 'y0UMO',
              'yprime0Gulf', 'yprime0Ekman', 'yprime0UMO',
              'A0Gulf', 'A0Ekman', 'A0UMO',
              'phi0Gulf', 'phi0Ekman', 'phi0UMO',
              'sigmaGulf', 'sigmaEkman', 'sigmaUMO']

model_name = "BasicSinusoid"
cargs = BDA.SetupHelper(model_name)

ntemps = 10
nburn0 = 3000
nburn = 1000
nprod = 1000

scatter_val = 1e-3
nwalkers = 100


def SignalModelGulf(time, P,
                y0Gulf, y0Ekman, y0UMO,
                yprime0Gulf, yprime0Ekman, yprime0UMO,
                A0Gulf, A0Ekman, A0UMO,
                phi0Gulf, phi0Ekman, phi0UMO,
                sigmaGulf, sigmaEkman, sigmaUMO
                ):
    return y0Gulf + yprime0Gulf*time + A0Gulf*np.sin(2*np.pi*time/P + phi0Gulf)

def SignalModelEkman(time, P,
                y0Gulf, y0Ekman, y0UMO,
                yprime0Gulf, yprime0Ekman, yprime0UMO,
                A0Gulf, A0Ekman, A0UMO,
                phi0Gulf, phi0Ekman, phi0UMO,
                sigmaGulf, sigmaEkman, sigmaUMO):
    return y0Ekman + yprime0Ekman*time + A0Ekman*np.sin(2*np.pi*time/P + phi0Ekman)

def SignalModelUMO(time, P,
                y0Gulf, y0Ekman, y0UMO,
                yprime0Gulf, yprime0Ekman, yprime0UMO,
                A0Gulf, A0Ekman, A0UMO,
                phi0Gulf, phi0Ekman, phi0UMO,
                sigmaGulf, sigmaEkman, sigmaUMO):
    return y0UMO + yprime0UMO*time + A0UMO*np.sin(2*np.pi*time/P + phi0UMO)

x = [time, time, time]
y = [yGulf, yEkman, yUMO]
SignalModel = [SignalModelGulf, SignalModelEkman, SignalModelUMO]

DD = BDA.GetData(
    x, y, SignalModel, model_name=model_name, params=params,
    param_keys=param_keys, nburn0=nburn0,
    ntemps=ntemps, nwalkers=nwalkers, nburn=nburn, nprod=nprod,
    scatter_val=scatter_val)

samples = DD['samples']
lnprobs = DD['lnprobs']
symbols = [params[key]['symbol'] for key in param_keys]
units = [params[key]['unit'] for key in param_keys]

BDA.PlotCorner(samples, params, param_keys, model_name, cargs=cargs,
               label_offset=0.75)

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
