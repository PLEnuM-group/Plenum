import numpy as np
from pickle import dump
from settings import BASEPATH
from os.path import join
import argparse

from daemonflux import Flux

parser = argparse.ArgumentParser(description="Calculate atmospheric fluxes with MCEq/daemonflux.")
parser.add_argument("-s", "--savefile", type=str, default="MCEq_daemonflux.pckl")
args = parser.parse_args()
savepath = join(BASEPATH, "resources", args.savefile)
print("Flux will be saved to:", savepath)

# setup daemonflux with IceCube location/atmosphere
# currently we use this for all detectors, just rotated to the specific latitude
# TODO: use regular atmosphere for the other detectors

fl = Flux(location="IceCube", use_calibration=True, debug=1)
egrid = np.logspace(0, 9)

flux_grid = np.zeros((len(fl.zenith_angles), len(egrid)))
zeniths = np.array(fl.zenith_angles, dtype=float)
for ii, zz in enumerate(fl.zenith_angles):
    # default is flux * E^3, so we need to correct for that
    flux_grid[ii] = fl.flux(egrid, zz, "total_numuflux")  / (egrid**3)
flux_grid = flux_grid.T
savepath = join(BASEPATH, "resources", "MCEq_daemonflux.pckl")

with open(savepath, "wb") as f:
    dump(((egrid, zeniths), flux_grid), f)