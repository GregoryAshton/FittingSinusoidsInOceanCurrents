""" Turn the csv file of logP(data|model) into latex macros """
import numpy as np
import os

src = "../Evidence.dat"
dest = "./Evidence.tex"

if os.path.isfile(dest):
    os.remove(dest)


def clean_up_model_name(m):
    return m.replace("0.1", "")


def get_odds(ma, mb, data):
    odds = data[ma]['val'] - data[mb]['val']
    err = np.sqrt(data[ma]['err']**2 + data[mb]['err']**2)
    return odds, err


def write_macro(ma, mb, data):
    odds, err = get_odds(ma, mb, data)
    ma = clean_up_model_name(ma)
    mb = clean_up_model_name(mb)
    with open(dest, "a+") as f:
        string = ma + mb
        f.write("\def\odds{}{{{:2.1f}}}\n".format(string, odds))
        f.write("\def\err{}{{{:2.1f}}}\n".format(string, err))

# Read in the data
data = {}
with open(src, "r") as f:
    for line in f:
        line = line.rstrip("\n")
        e = line.split(",")
        data[e[0]] = {'val': np.float(e[1]),
                      'err': np.float(e[2])
                      }

for ma in ['BasicSinusoidAmplitudeDecay', 'BasicSinusoidAmplitudeDecayWithTransient']:
    mb = 'BasicSinusoid'
    write_macro(ma, mb, data)

write_macro("BasicSinusoidAmplitudeDecay", "BasicSinusoidAmplitudeDecayWithTransient", data)
