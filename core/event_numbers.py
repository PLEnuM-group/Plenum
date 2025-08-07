import numpy as np
import healpy as hp
import pickle
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from scipy.interpolate import InterpolatedUnivariateSpline
from aeff_calculations import calc_aeff_factor, setup_aeff_grid, earth_rotation
from tools import get_mids
import settings as st
from settings import join, LOCALPATH, PHI_NGC, E0_NGC, poles, LIVETIME
from fluxes import PL_flux, astro_flux

from tqdm import tqdm

print("Calculate detection efficiencies...")
for _hemi in ["full", "upgoing"]:
    print(f"... for {_hemi} detector.")
    # both files have the same information, but the 'MH' file contains Mephistograms instead of arrays
    if True:
        with open(join(LOCALPATH, f"effective_area_MH_{_hemi}.pckl"), "rb") as f:
            aeff_2d = pickle.load(f)
    else:
        with open(join(BASEPATH, f"resources/effective_area_upgoing.pckl"), "rb") as f:
            aeff_2d = pickle.load(f)

    ### calculate raw neutrino rate ~ detection efficiency
    # Res = integral dE ( A_eff * (E/GeV)**(-gamma) ) / delta sindec
    gamma_range = [2.0, 2.5, 3.0]  # np.round(np.arange(1.5, 3.6, step=0.1), decimals=1)
    ## NEW: base on PS acceptance

    tcks = dict()
    for ii, gamma in enumerate(gamma_range):
        flux = PL_flux(PHI_NGC, gamma, E0_NGC, "powerlaw")
        tcks[gamma] = dict()
        for det in poles:
            if "Plenum" in det:
                continue
            _res = []
            for dec in get_mids(st.dec_bins):
                src_config = dict(
                    sindec_mids=st.sindec_mids,
                    livetime=LIVETIME,
                    ewidth=st.ewidth,
                    dpsi_max=0,
                    grid_2d=1,
                    diff_or_ps="ps",
                    dec=dec,
                )

                aeff_factor_signal = calc_aeff_factor(aeff_2d[det], **src_config)

                k_s = astro_flux(
                    aeff_factor_signal,
                    st.emids,
                    energy_resolution=None,
                    phi_scaling=1,
                    flux_shape=flux,
                )
                _res.append(np.sum(k_s))
            _res = np.array(_res)

            tcks[gamma][det] = InterpolatedUnivariateSpline(st.sindec_mids, _res)

    with open(join(LOCALPATH, f"detection_efficiencies_{_hemi}.pckl"), "wb") as f:
        pickle.dump((tcks, st.sindec_mids), f)

    print("Calculate instantaneous detection efficiencies")
    # same, but instantaneous, i.e. time/RA dependence
    inst_rel_events_ra_dec = {}
    rel_events_ra_dec = {}

    nside = 2**8
    npix = hp.nside2npix(nside)
    pix = np.arange(npix)
    hp_angles = hp.pix2ang(nside, pix)

    # binning setup
    _azi = hp_angles[1]
    _zen = hp_angles[0] - np.pi / 2

    # for rotation
    hp_coords = SkyCoord(_azi * u.radian, _zen * u.radian, frame="icrs")

    # for integration
    pp, tt = np.meshgrid(st.ra_mids, np.arcsin(st.sindec_mids))
    eq_coords = SkyCoord(pp * u.radian, tt * u.radian, frame="icrs")

    for ii, gamma in tqdm(enumerate(gamma_range)):
        inst_rel_events_ra_dec[gamma] = {}
        # rel_events_ra_dec[gamma] = {}

        rel_tmp = tcks[gamma]["IceCube"](st.sindec_mids)

        grid2d, _ = setup_aeff_grid(
            [rel_tmp],
            st.sindec_mids,
            st.ra_mids,
            st.ra_width,
            local=True,
            log_int=False,
        )

        for k, coord in poles.items():
            if "Plenum" in k:
                continue
            new_aeff = earth_rotation(
                poles[k]["lat"],
                poles[k]["lon"],
                eq_coords,
                hp_coords,
                grid2d,
                st.ra_width,
                log_aeff=False,
                return_3D=True,
                time=Time("2025-01-01 00:00:00"),
            )
            inst_rel_events_ra_dec[gamma][k] = new_aeff[0] / np.max(new_aeff)

    with open(join(LOCALPATH, f"inst_rel_events_ra_dec_{_hemi}.pckl"), "wb") as f:
        pickle.dump((inst_rel_events_ra_dec, st.ra_mids, np.arcsin(st.sindec_mids)), f)
    # with open(join(LOCALPATH, "rel_events_ra_dec.pckl"), "wb") as f:
    #     pickle.dump((rel_events_ra_dec, ra_vals, dec_vals), f)
