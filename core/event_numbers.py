#!/usr/bin/env python3

import numpy as np
import pickle
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from scipy.interpolate import InterpolatedUnivariateSpline, RegularGridInterpolator
from aeff_calculations import aeff_eval_e_sd, get_aeff_and_binnings
from tools import get_mids, _trans
from settings import poles, E_NORM
from tqdm import tqdm

print("Calculate detection efficiencies")
aeff_2d, log_ebins, ebins, sindec_bins, ra_bins = get_aeff_and_binnings("full")
emids = get_mids(ebins)
ewidth = np.diff(ebins)
sindec_mids = get_mids(sindec_bins)
sindec_width = np.diff(sindec_bins)
ra_width = np.diff(ra_bins)


### calculate raw neutrino rate ~ detection efficiency
# Res = integral dE ( A_eff * (E/GeV)**(-gamma) ) / delta sindec
tcks = dict()
for ii, gamma in enumerate(np.round(np.arange(1.4, 3.6, step=0.1), decimals=1)):
    tcks[gamma] = dict()
    for det in ['Plenum-1', 'Plenum-2', 'IceCube', 'Gen-2', 'P-ONE', 'KM3NeT', 'Baikal-GVD']:
        Res = np.sum(
            aeff_eval_e_sd(
                aeff_2d[det], sindec_width, ewidth, ra_width) * (emids/E_NORM)**(-gamma), 
            axis=-1)
        # pad with boundary values such that the detection efficiency 
        # is defined up to sindec of -1 and 1
        padded_sd = np.concatenate([[-1], sindec_mids, [1]])
        padded_res = np.concatenate([[Res[0]], Res, [Res[-1]]])
        tcks[gamma][det] = InterpolatedUnivariateSpline(padded_sd, np.log(padded_res))
        
with open("../resources/detection_efficiencies.pckl", "wb") as f:
    pickle.dump((tcks, padded_sd), f)
    
    
print("Calculate instantaneous detection efficiencies")
# same, but instantaneous, i.e. time/RA dependence
num = 500
ra_vals = np.linspace(0, 2*np.pi, num)
ra_val_mids = get_mids(ra_vals)
ra_val_width = np.diff(ra_vals)
dec_vals = np.linspace(-np.pi/2, np.pi/2, num)
dec_val_mids = get_mids(dec_vals)

inst_rel_events_ra_dec = {}
rel_events_ra_dec = {}

for ii, gamma in tqdm(enumerate(np.round(np.arange(1.4, 3.6, step=0.1), decimals=1))):
    inst_rel_events_ra_dec[gamma] = {}
    rel_events_ra_dec[gamma] = {}

    rel_tmp = np.exp(tcks[gamma]["IceCube"](sindec_mids)) \
            / np.exp(tcks[gamma]["IceCube"](0))
    rel_tmp = rel_tmp[::-1,np.newaxis] * np.ones_like(
        np.atleast_2d(ra_vals))
    
    # pad with zeros to have full declination coverage without nasty boundary issues
    shape = np.shape(rel_tmp)
    padded_array = np.zeros(np.array(shape) + [2, 0])
    padded_array[1:shape[0]+1,:shape[1]] = rel_tmp
    padded_dec = np.concatenate([[-np.pi/2], np.arcsin(sindec_mids), [np.pi/2]])
    padded_array[0] = padded_array[1]
    padded_array[-1] = padded_array[-2]
    
    grid2d = RegularGridInterpolator(
        (padded_dec, ra_vals),
        padded_array,
        method='linear',
        bounds_error=False,
        fill_value=None
    )
    # grid elements are calculated for each energy bin, grid is theta x phi
    # coordinate grid in equatorial coordinates (icrs)
    # these will be the integration coordinates
    pp, tt = np.meshgrid(ra_vals, dec_vals)
    eq_grid = SkyCoord(
        pp * u.radian,
        tt * u.radian,
        frame="icrs"
    )
    inst_rel_events_ra_dec[gamma]["Plenum-1"] = np.zeros_like(pp)
    inst_rel_events_ra_dec[gamma]["Plenum-2"] = np.zeros_like(pp)
    rel_events_ra_dec[gamma]["Plenum-1"] = np.zeros_like(pp)
    rel_events_ra_dec[gamma]["Plenum-2"] = np.zeros_like(pp)
    # loop over detectors
    for k, coord in poles.items():
        if "Plenum" in k: continue
        # local detector
        loc = EarthLocation(lat=coord["lat"], lon=coord["lon"])
        # arbitrary time, doesnt matter here
        time = Time('2020-2-20 00:00:00')
        # transform integration coordinates to local frame
        local_coords_grid = eq_grid.transform_to(AltAz(obstime=time, location=loc))
        inst_rel_events_ra_dec[gamma][k] = grid2d(
            (local_coords_grid.alt.rad, local_coords_grid.az.rad))
        # average over right ascension
        rel_events_ra_dec[gamma][k] = np.sum(inst_rel_events_ra_dec[gamma][k] / len(ra_val_mids), axis=1)
        rel_events_ra_dec[gamma][k] = rel_events_ra_dec[gamma][k][:,np.newaxis] \
                    * np.ones_like(np.atleast_2d(ra_vals))
        if "Gen" in k:
            inst_rel_events_ra_dec[gamma]["Plenum-2"] += inst_rel_events_ra_dec[gamma][k]
            rel_events_ra_dec[gamma]["Plenum-2"] += rel_events_ra_dec[gamma][k]
        elif "Ice" in k:
            inst_rel_events_ra_dec[gamma]["Plenum-1"] += inst_rel_events_ra_dec[gamma][k]
            rel_events_ra_dec[gamma]["Plenum-1"] += rel_events_ra_dec[gamma][k]
        else:
            inst_rel_events_ra_dec[gamma]["Plenum-1"] += inst_rel_events_ra_dec[gamma][k]
            rel_events_ra_dec[gamma]["Plenum-1"] += rel_events_ra_dec[gamma][k]
            inst_rel_events_ra_dec[gamma]["Plenum-2"] += inst_rel_events_ra_dec[gamma][k]
            rel_events_ra_dec[gamma]["Plenum-2"] += rel_events_ra_dec[gamma][k]

with open("../resources/rel_events_ra_dec.pckl", "wb") as f:
    pickle.dump((rel_events_ra_dec, ra_vals, dec_vals), f)
    
with open("../resources/inst_rel_events_ra_dec.pckl", "wb") as f:
    pickle.dump((inst_rel_events_ra_dec, ra_vals, dec_vals), f)