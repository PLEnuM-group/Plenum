# Prepares all needed histograms+binnings using mephisto

# Import
import pickle
from os.path import join

import numpy as np
from aeff_calculations import (
    get_aeff_and_binnings,
    setup_aeff_grid,
    aeff_rotation,
    padded_interpolation,
)
from scipy.interpolate import RegularGridInterpolator
import settings as st
from settings import interpolation_method
from tools import get_mids

from mephisto import Mephistogram


# # Effective Area
# Calculation can be found in `aeff_calculations.py`
keys = ["upgoing", "full"]
for hemi in keys:
    aeff_2d_base, logE_bins_old, _, sindec_bins_old = get_aeff_and_binnings(hemi)

    need_to_update_binning = np.any(logE_bins_old != st.logE_bins) or np.any(
        sindec_bins_old != st.sindec_bins
    )
    if need_to_update_binning:
        # provide interpolation function for effective area
        aeff_interp = {}
        for k in aeff_2d_base:
            aeff_interp[k] = padded_interpolation(
                np.log(aeff_2d_base[k]),
                logE_bins_old,
                sindec_bins_old,
                bounds_error=True,
                method=interpolation_method,
            )

    # set up new standardized binning
    print(len(st.emids), "log_10(energy) bins")
    print(len(st.sindec_mids), "declination bins")
    # evaluate the interpolation and make mephistograms
    aeff_2d = {}
    if need_to_update_binning:
        ss, ll = np.meshgrid(st.sindec_mids, st.logE_mids)
    for k in aeff_2d_base:
        if need_to_update_binning:
            aeff_tmp = np.exp(aeff_interp[k]((ll, ss)))
            aeff_tmp[np.isnan(aeff_tmp)] = 0

        aeff_2d[k] = Mephistogram(
            aeff_tmp.T if need_to_update_binning else aeff_2d_base[k].T,
            (st.sindec_bins, st.logE_bins),
            ("sin(dec)", "log(E/GeV)"),
            make_hist=False,
        )

    with open(join(st.LOCALPATH, f"effective_area_MH_{hemi}.pckl"), "wb") as f:
        pickle.dump(aeff_2d, f)

# atmospheric neutrino background
# Calculation can be found in `atmospheric_background.py`
# MCEQ
with open(join(st.BASEPATH, "resources/MCEq_daemonflux.pckl"), "rb") as f:
    (e_grid, zen), bckg_flux = pickle.load(f)

# set up the interpolation function
sindec_mids_bg = -np.cos(np.deg2rad(zen))
rgi = RegularGridInterpolator(
    (e_grid, sindec_mids_bg),
    np.log(bckg_flux ),
    method=interpolation_method,
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
            st.poles[k]["lat"],
            st.poles[k]["lon"],
            eq_coords,
            grid2d,
            st.ra_width,
            log_aeff=True,
        ).T,
        (st.sindec_bins, st.logE_bins),
        ("sin(dec)", "log(E/GeV)"),
        make_hist=False,
    )

# check if histos are matching
#print(bg_i["IceCube"].match(aeff_2d["IceCube"], verbose=True))

with open(join(st.LOCALPATH, "atmospheric_background_daemonflux_MH.pckl"), "wb") as f:
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
    with open(join(st.LOCALPATH, f"Psi2-{st.delta_psi_max}_res_mephistograms.pckl"), "rb") as f:
        all_psi = pickle.load(f)
    e_psi2_grid = all_psi["dec-0.0"]
    e_psi2_grid.normalize()
