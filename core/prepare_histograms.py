# Prepares all needed histograms+binnings using mephisto

# Import
import pickle
from os.path import join

import numpy as np
from aeff_calculations import get_aeff_and_binnings, setup_aeff_grid, aeff_rotation
from scipy.interpolate import RegularGridInterpolator
import settings as st
from tools import get_mids

from mephisto import Mephistogram


# # Effective Area
# Calculation can be found in `aeff_calculations.py`
keys = ["upgoing", "full"]
for hemi in keys:
    # this is the baseline binning as provided in the data release
    aeff_2d_base, logE_bins_old, _, sindec_bins_old = get_aeff_and_binnings(hemi)
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
            aeff_tmp.T,
            (st.sindec_bins, st.logE_bins),
            ("sin(dec)", "log(E/GeV)"),
            make_hist=False,
        )

    with open(join(st.LOCALPATH, f"effective_area_MH_{hemi}.pckl"), "wb") as f:
        pickle.dump(aeff_2d, f)

# atmospheric neutrino background
# Calculation can be found in `atmospheric_background.py`
# MCEQ
with open(join(st.BASEPATH, "resources/MCEq_flux.pckl"), "rb") as f:
    (e_grid, zen), flux_def = pickle.load(f)
# set up the interpolation function
sindec_mids_bg = -np.cos(np.deg2rad(zen))
rgi = RegularGridInterpolator(
    (e_grid, sindec_mids_bg), np.log(flux_def["numu_total"])
)

# finer interpolation for further steps
ss, em = np.meshgrid(st.sindec_mids, st.emids)
numu_bg = np.exp(rgi((em, ss)))

grid2d, eq_coords = setup_aeff_grid(
    numu_bg, st.sindec_mids, st.ra_mids, st.ra_width, log_int=True
)

# loop over detectors and rotate the local background flux to equatorial coordinates
# i.e. calculate the average bg flux per day in equatorial sin(dec)
bg_i = {}
det_list = ["IceCube", "P-ONE", "KM3NeT", "Baikal-GVD"]
for k in det_list:
    bg_i[k] = Mephistogram(
        aeff_rotation(
            st.poles[k]["lat"], st.poles[k]["lon"], eq_coords, grid2d, st.ra_width, log_aeff=True).T,
        (st.sindec_bins, st.logE_bins),
        ("sin(dec)", "log(E/GeV)"),
        make_hist=False,
    )

# check if histos are matching
print(bg_i["IceCube"].match(aeff_2d["IceCube"], verbose=True))

with open(join(st.LOCALPATH, "atmospheric_background_MH.pckl"), "wb") as f:
    pickle.dump(bg_i, f)

# # Energy resolution function
# Calculation can be found in `resolution.py`
#
# ` %run ../../core/resolution.py`
# -> only run this if you need to update the histograms

##  These are already mephistograms ! Only need to run this if you want to import them
if False:
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
    # angular resolution
    with open(join(st.LOCALPATH, "Psi2_res_mephistograms.pckl"), "rb") as f:
        all_psi = pickle.load(f)
    e_psi2_grid = all_psi["dec-0.0"]
    e_psi2_grid.normalize()
