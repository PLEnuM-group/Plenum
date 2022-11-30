# Prepares all needed histograms+binnings using mephisto

# Import
import pickle
from os.path import join

import numpy as np
from aeff_calculations import get_aeff_and_binnings
from scipy.interpolate import RegularGridInterpolator
import settings as st
from tools import get_mids

from mephisto import Mephistogram


# # Effective Area
# Calculation can be found in `aeff_calculations.py`

# this is the baseline binning as provided in the data release
aeff_2d_base, logE_bins_old, _, sindec_bins_old, _ = get_aeff_and_binnings("upgoing")
logE_mids_old = get_mids(logE_bins_old)
sindec_mids_old = get_mids(sindec_bins_old)

# provide interpolation function for effective area
aeff_interp = {}
pad_logE = np.concatenate([[logE_bins_old[0]], logE_mids_old, [logE_bins_old[-1]]])
pad_sd = np.concatenate([[-1], sindec_mids_old, [1]])
for k in aeff_2d_base:
    aeff_interp[k] = RegularGridInterpolator(
        (pad_logE, pad_sd),
        np.pad(np.log(aeff_2d_base[k]), 1, mode="edge"),
        bounds_error=False,
        fill_value=1e-16,
    )

# set up new standardized binning
print(len(st.emids), "log_10(energy) bins")
print(len(st.sindec_mids), "declination bins")
# evaluate the interpolation and make mephistograms
aeff_2d = {}
ss, ll = np.meshgrid(st.sindec_mids, st.logE_mids)
for k in aeff_2d_base:
    aeff_tmp = np.exp(aeff_interp[k]((ll, ss)))
    aeff_tmp[np.isnan(aeff_tmp)] = 0

    aeff_2d[k] = Mephistogram(
        aeff_tmp,
        (st.logE_bins, st.sindec_bins),
        ("log(E/GeV)", "sin(dec)"),
        make_hist=False,
    )

with open(join(st.LOCALPATH, "effective_area_MH_upgoing.pckl"), "wb") as f:
    pickle.dump(aeff_2d, f)

# Calculation can be found in `atmospheric_background.py`
# MCEQ
with open(join(st.BASEPATH, "resources/MCEq_flux.pckl"), "rb") as f:
    (e_grid, zen), flux_def = pickle.load(f)
# re-bin the atmospheric background flux
rgi = RegularGridInterpolator(
    (e_grid, -np.cos(np.deg2rad(zen))), np.log(flux_def["numu_total"])
)
# baseline evaluation grid
ss, em = np.meshgrid(st.sindec_mids, st.emids)
bckg_histo = Mephistogram(
    np.exp(rgi((em, ss))),
    (st.logE_bins, st.sindec_bins),
    ("log(E/GeV)", "sin(dec)"),
    make_hist=False,
)

# check if histos are matching
print(bckg_histo.match(aeff_2d["IceCube"], verbose=True))

with open(join(st.LOCALPATH, "atmospheric_background_MH.pckl"), "wb") as f:
    pickle.dump(bckg_histo, f)

# # Energy resolution function
# Calculation can be found in `resolution.py`
#
# ` %run ../../core/resolution.py`
# -> only run this if you need to update the histograms


# baseline resolution
with open(join(st.LOCALPATH, "energy_smearing_MH_up.pckl"), "rb") as f:
    baseline_eres = pickle.load(f)
baseline_eres.normalize()

# resolution improved by 50%
impro_factor = 0.5
filename = join(
    st.LOCALPATH,
    f"improved_{impro_factor}_artificial_energy_smearing_MH_up.pckl",
)
with open(filename, "rb") as f:
    improved_eres = pickle.load(f)
improved_eres.normalize()


# # Psi²
# Calculation can be found in `resolution.py`

filename = join(
    st.LOCALPATH,
    f"e_psf_grid_psimax-{st.delta_psi_max}_bins-{st.bins_per_psi2}.pckl",
)

# angular resolution
with open(filename, "rb") as f:
    all_psi = pickle.load(f)
e_psi2_grid = all_psi["dec-0.0"]
e_psi2_grid.normalize()
