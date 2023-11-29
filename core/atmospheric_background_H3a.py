# old background flux

from settings import *
from os.path import join
import mceq_config as config
from MCEq.core import MCEqRun
import crflux.models as pm
import pickle
import argparse
from scipy.interpolate import RegularGridInterpolator
import settings as st
from settings import interpolation_method
from aeff_calculations import (
    setup_aeff_grid,
    aeff_rotation,
)

from mephisto import Mephistogram

parser = argparse.ArgumentParser(description="Calculate atmospheric fluxes with MCEq.")
parser.add_argument("-s", "--savefile", type=str, default="MCEq_flux_H3a.pckl")
args = parser.parse_args()
savepath = join(BASEPATH, "resources", args.savefile)
print("Flux will be saved to:", savepath)


# setup
config.e_min = 1e-1
config.e_max = 1e12

mceq = MCEqRun(
    interaction_model="SIBYLL2.3c",
    primary_model=(pm.HillasGaisser2012, "H3a"),
    theta_deg=0.0,
    density_model=("MSIS00_IC", ("SouthPole", "January")),
)


mag = 0
# Define equidistant grid in cos(theta)
zen = np.rad2deg(np.arccos(np.linspace(1, -1, 21)))
flux_def = dict()

all_component_names = [
    "numu_conv",
    "numu_pr",
    "numu_total",
    "mu_conv",
    "mu_pr",
    "mu_total",
    "nue_conv",
    "nue_pr",
    "nue_total",
    "nutau_pr",
]

# Initialize empty grid
for frac in all_component_names:
    flux_def[frac] = np.zeros((len(mceq.e_grid), len(zen)))

# fluxes calculated for different angles
for ti, theta in enumerate(zen):
    mceq.set_theta_deg(theta)
    mceq.solve()

    # same meaning of prefixes for muon neutrinos as for muons
    flux_def["mu_conv"][:, ti] = mceq.get_solution("conv_mu+", mag) + mceq.get_solution(
        "conv_mu-", mag
    )

    flux_def["mu_pr"][:, ti] = mceq.get_solution("pr_mu+", mag) + mceq.get_solution(
        "pr_mu-", mag
    )

    flux_def["mu_total"][:, ti] = mceq.get_solution(
        "total_mu+", mag
    ) + mceq.get_solution("total_mu-", mag)

    # same meaning of prefixes for muon neutrinos as for muons
    flux_def["numu_conv"][:, ti] = mceq.get_solution(
        "conv_numu", mag
    ) + mceq.get_solution("conv_antinumu", mag)

    flux_def["numu_pr"][:, ti] = mceq.get_solution("pr_numu", mag) + mceq.get_solution(
        "pr_antinumu", mag
    )

    flux_def["numu_total"][:, ti] = mceq.get_solution(
        "total_numu", mag
    ) + mceq.get_solution("total_antinumu", mag)

    # same meaning of prefixes for electron neutrinos as for muons
    flux_def["nue_conv"][:, ti] = mceq.get_solution(
        "conv_nue", mag
    ) + mceq.get_solution("conv_antinue", mag)

    flux_def["nue_pr"][:, ti] = mceq.get_solution("pr_nue", mag) + mceq.get_solution(
        "pr_antinue", mag
    )

    flux_def["nue_total"][:, ti] = mceq.get_solution(
        "total_nue", mag
    ) + mceq.get_solution("total_antinue", mag)

    # since there are no conventional tau neutrinos, prompt=total
    flux_def["nutau_pr"][:, ti] = mceq.get_solution(
        "total_nutau", mag
    ) + mceq.get_solution("total_antinutau", mag)
print("\U0001F973")


## save the result
with open(savepath, "wb") as f:
    pickle.dump(((mceq.e_grid, zen), flux_def), f)


# set up the interpolation function
sindec_mids_bg = -np.cos(np.deg2rad(zen))
rgi = RegularGridInterpolator(
    (mceq.e_grid, sindec_mids_bg),
    np.log(flux_def["numu_total"]),
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

with open(join(st.LOCALPATH, "atmospheric_background_MH.pckl"), "wb") as f:
    pickle.dump(bg_i, f)
