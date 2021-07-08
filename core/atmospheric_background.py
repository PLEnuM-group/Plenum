#!/usr/bin/env python3

import sys
sys.path.append('../core')
from settings import *
import mceq_config as config
from MCEq.core import MCEqRun
import crflux.models as pm
import pickle
from tqdm import tqdm
from scipy.interpolate import splrep, splev, RegularGridInterpolator

# setup
config.e_min = 0.1
config.e_max = 1E10

mceq = MCEqRun(
    interaction_model='SIBYLL2.3c',
    primary_model = (pm.HillasGaisser2012, "H3a"),
    theta_deg=0.0,
    density_model=('MSIS00_IC',('SouthPole','January'))
)


mag = 0
# Define equidistant grid in cos(theta)
angles = np.rad2deg(np.arccos(np.linspace(1, -1, 21)))
flux_def = dict()

all_component_names = [
    'numu_conv','numu_pr','numu_total',
    'mu_conv','mu_pr','mu_total',
    'nue_conv','nue_pr','nue_total','nutau_pr'
]

# Initialize empty grid
for frac in all_component_names:
    flux_def[frac] = np.zeros((len(mceq.e_grid), len(angles)))

# fluxes calculated for different angles
for ti, theta in tqdm(enumerate(angles)):
    mceq.set_theta_deg(theta)
    mceq.solve()
    
    # same meaning of prefixes for muon neutrinos as for muons
    flux_def['mu_conv'][:,ti] = (mceq.get_solution('conv_mu+', mag)
                         + mceq.get_solution('conv_mu-', mag))

    flux_def['mu_pr'][:,ti] = (mceq.get_solution('pr_mu+', mag)
                       + mceq.get_solution('pr_mu-', mag))

    flux_def['mu_total'][:,ti] = (mceq.get_solution('total_mu+', mag)
                          + mceq.get_solution('total_mu-', mag))
    
    
    # same meaning of prefixes for muon neutrinos as for muons
    flux_def['numu_conv'][:,ti] = (mceq.get_solution('conv_numu', mag)
                         + mceq.get_solution('conv_antinumu', mag))

    flux_def['numu_pr'][:,ti] = (mceq.get_solution('pr_numu', mag)
                       + mceq.get_solution('pr_antinumu', mag))

    flux_def['numu_total'][:,ti] = (mceq.get_solution('total_numu', mag)
                          + mceq.get_solution('total_antinumu', mag))

    # same meaning of prefixes for electron neutrinos as for muons
    flux_def['nue_conv'][:,ti] = (mceq.get_solution('conv_nue', mag)
                        + mceq.get_solution('conv_antinue', mag))

    flux_def['nue_pr'][:,ti] = (mceq.get_solution('pr_nue', mag)
                      + mceq.get_solution('pr_antinue', mag))

    flux_def['nue_total'][:,ti] = (mceq.get_solution('total_nue', mag)
                         + mceq.get_solution('total_antinue', mag))


    # since there are no conventional tau neutrinos, prompt=total
    flux_def['nutau_pr'][:,ti] = (mceq.get_solution('total_nutau', mag)
                        + mceq.get_solution('total_antinutau', mag))
print("\U0001F973")


## save the result
with open("../resources/MCEq_flux.pckl", "wb") as f:
    pickle.dump(((mceq.e_grid, angles), flux_def), f)
    
# plotting can be found in background_flux.ipynb