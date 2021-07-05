#!/usr/bin/env python3

import numpy as np
import pickle
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from scipy.interpolate import RegularGridInterpolator, splrep, splev
from settings import LIVETIME, GAMMA_ASTRO, PHI_ASTRO
from tools import get_mids

# local coordinates of experiments
coords = {
    'IceCube': {"lon": 0*u.deg, "lat": -90*u.deg},
    'P-ONE': {"lon": -123.3656*u.deg, "lat": 48.4284*u.deg},
    'KM3NeT': {"lon": (16 + 6/60)*u.deg, "lat": (36 + 16/60)*u.deg},
    'Baikal-GVD': {"lon": 108.1650*u.deg, "lat": 53.5587*u.deg}
}

d_public = np.genfromtxt(
    '../icecube_10year_ps/irfs/IC86_II_effectiveArea.csv',
    skip_header=1
)
# log10(E_nu/GeV)_min log10(E_nu/GeV)_max
# Dec_nu_min[deg] Dec_nu_max[deg]
# A_Eff[cm^2]
emin, emax = np.power(10, [d_public[:,0], d_public[:,1]])
sindec_min, sindec_max = np.sin(np.deg2rad([d_public[:,2], d_public[:,3]]))
# actually it's cos(zen), but we'll just switch the eff area ordering
aeff = d_public[:,4]

emin = np.unique(emin) # all lower energy bounds
emax = np.unique(emax) # all upper energy bounds
ebins = np.unique([emin, emax]) # all bin edges in order
sindec_min = np.unique(sindec_min) # all lower sindec bounds
sindec_max = np.unique(sindec_max) # all upper sindec bounds
sindec_bins = np.unique([sindec_min, sindec_max]) # all bin edges in order
ra_bins = np.linspace(0, np.pi*2, num=101)
ra_mids = get_mids(ra_bins)
ra_width = ra_bins[1:] - ra_bins[:-1]

aeff_2d = dict()
# re-shape into 2D array with (A(E) x A(delta))
# and switch the eff area ordering
aeff_2d["icecube_full"] = aeff.reshape(len(sindec_min), len(emin)).T

emids = get_mids(ebins)
ewidth = emax - emin
sindec_mids = get_mids(sindec_bins)
sindec_width = sindec_max - sindec_min

print(len(emin), "log_10(energy) bins")
print(len(sindec_min), "declination bins")

# cut at delta > -5deg
min_idx = np.searchsorted(sindec_mids, np.sin(np.deg2rad(-5)))
print(f"Below {np.rad2deg(np.arcsin(sindec_min[min_idx])):1.2f}deg aeff is 0")
aeff_2d["icecube"] = np.copy(aeff_2d["icecube_full"])
aeff_2d["icecube"][:,:min_idx] = 0

# some event numbers for checking
det = "icecube"
aeff_factor = (aeff_2d[det] * sindec_width).T * ewidth * LIVETIME * np.sum(ra_width)
astro_ev = aeff_factor * (emids/1E5)**(-GAMMA_ASTRO) * PHI_ASTRO
print(det)
print("astro events:", np.sum(astro_ev))


print("starting aeff rotations")
## Idea: transform the integration over R.A. per sin(dec) into local coordinates
# **Note:** here, equal contributions of each detector are assumed. 
# In case one wants to use different livetimes, the effective areas have to multiplied 
# individually before calculating e.g. the expected number of astrophysical events

# Interpolated grid of the effective area in "local" coordinates
# (= icecube's native coordinates)
grid2d = [RegularGridInterpolator(
    (np.arcsin(sindec_mids), ra_mids), # transform dec to local theta
    # switch back to local zenith, add ra as new axis and normalize accordingly
    aeff_2d["icecube"][i][::-1,np.newaxis] / np.atleast_2d(ra_width)/ len(ra_mids),
    method='linear',
    bounds_error=False,
    fill_value=0.
) for i in range(len(emids))]
# grid elements are calculated for each energy bin, grid is theta x phi

# coordinate grid in equatorial coordinates (icrs)
# these will be the integration coordinates
pp, tt = np.meshgrid(ra_mids, np.arcsin(sindec_mids))
eq_coords = SkyCoord(
    pp * u.radian,
    tt * u.radian,
    frame="icrs"
)

aeff_i = {}
factor = 10 # for plotting energy slices
aeff_i["Plenum-1"] = np.zeros_like(aeff_2d["icecube"])

# loop over detectors
for k, coord in coords.items():
    # local detector
    loc = EarthLocation(
        lat=coord["lat"],
        lon=coord["lon"],
    )
    # arbitrary time, doesnt matter here
    time = Time('2021-6-21 00:00:00')
    # transform integration coordinates to local frame
    local_coords = eq_coords.transform_to(AltAz(obstime=time, location=loc))
    # sum up the contributions over the transformed RA axis per declination 
    
    # loop over the energy bins to get the same shape of aeff as before
    # sum along transformed ra coordinates
    aeff_i[k] = np.array([np.sum(
        grid2d[i]((local_coords.alt.rad, local_coords.az.rad)) * ra_width, # integrate over RA
        axis=1) for i in range(len(emids))])
    aeff_i["Plenum-1"] += aeff_i[k]
    
## GEN-2 will have ~7.5x effective area ==> 5times better discovery potential
aeff_i["Gen-2"] = aeff_i["IceCube"] * 5 ** (1 / 0.8)
aeff_i["Plenum-2"] = aeff_i["Plenum-1"] - aeff_i["IceCube"] + aeff_i["Gen-2"]

## save to disc
savefile = "../resources/tabulated_logE_sindec_aeff_upgoing.pckl"
print("Saving up-going effective areas to", savefile)
with open(savefile, "wb") as f:
    pickle.dump((np.log10(ebins), sindec_bins, aeff_i), f)
    
    
# same but wit FULL icecube effective area
print("starting full effective area calculation...")
# Interpolated grid of the effective area in "local" coordinates
# (= icecube's native coordinates)
grid2d = [RegularGridInterpolator(
    (np.arcsin(sindec_mids), ra_mids), # transform dec to local theta
    # switch back to local zenith, add ra as new axis and normalize accordingly
    aeff_2d["icecube_full"][i][::-1,np.newaxis] / np.atleast_2d(ra_width)/ len(ra_mids),
    method='linear',
    bounds_error=False,
    fill_value=0.
) for i in range(len(emids))]
# grid elements are calculated for each energy bin, grid is theta x phi

# coordinate grid in equatorial coordinates (icrs)
# these will be the integration coordinates
pp, tt = np.meshgrid(ra_mids, np.arcsin(sindec_mids))
eq_coords = SkyCoord(
    pp * u.radian,
    tt * u.radian,
    frame="icrs"
)

aeff_i_full = {}
aeff_i_full["Plenum-1"] = np.zeros_like(aeff_2d["icecube_full"])

# loop over detectors
for k, coord in coords.items():
    # local detector
    loc = EarthLocation(
        lat=coord["lat"],
        lon=coord["lon"],
    )
    # arbitrary time, doesnt matter here
    time = Time('2021-6-21 00:00:00')
    # transform integration coordinates to local frame
    local_coords = eq_coords.transform_to(AltAz(obstime=time, location=loc))
    # sum up the contributions over the transformed RA axis per declination 
    
    # loop over the energy bins to get the same shape of aeff as before
    # sum along transformed ra coordinates
    aeff_i_full[k] = np.array([np.sum(
        grid2d[i]((local_coords.alt.rad, local_coords.az.rad)) * ra_width, # integrate over RA
        axis=1) for i in range(len(emids))])
    aeff_i_full["Plenum-1"] += aeff_i_full[k]
    
# GEN-2 will have ~7.5x effective area ==> 5times better discovery potential
aeff_i_full["Gen-2"] = aeff_i_full["IceCube"] * 5 ** (1 / 0.8)
aeff_i_full["Plenum-2"] = aeff_i_full["Plenum-1"] - aeff_i_full["IceCube"] + aeff_i_full["Gen-2"]

# save
savefile = "../resources/tabulated_logE_sindec_aeff_full.pckl"
print("Saving full effective areas to", savefile)
with open(savefile, "wb") as f:
    pickle.dump((np.log10(ebins), sindec_bins, aeff_i_full), f)
    
    
print("Calculate energy smearing...")
from scipy.stats import gaussian_kde

# Calculate energy smearing
public_data_hist = np.genfromtxt(
    "../icecube_10year_ps/irfs/IC86_II_smearing.csv",
    skip_header=1
)
log_sm_emin, log_sm_emax = public_data_hist[:,0], public_data_hist[:,1]
log_sm_emids = (log_sm_emin + log_sm_emax) / 2.
log_sm_ereco_min, log_sm_ereco_max = public_data_hist[:,4], public_data_hist[:,5]
log_sm_ereco_mids = (log_sm_ereco_min + log_sm_ereco_max) / 2.
fractional_event_counts = public_data_hist[:,10]
eri = get_mids(np.arange(0.5, 9, 0.2))
log_emids = get_mids(np.log10(ebins))
ee, rr = np.meshgrid(log_emids, eri)

e_ereco_kdes = gaussian_kde(
    (log_sm_emids, log_sm_ereco_mids),
    weights=fractional_event_counts
)
# kvals has shape ereco x etrue
kvals = e_ereco_kdes([ee.flatten(),rr.flatten()]).reshape(len(eri), len(log_emids))
with open("../resources/energy_smearing_kde.pckl", "wb") as f:
    pickle.dump(kvals, f)
print("finished!")