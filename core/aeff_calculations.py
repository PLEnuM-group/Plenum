import numpy as np
import healpy as hp
from os.path import join
import pickle
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from scipy.interpolate import RegularGridInterpolator, griddata, RectBivariateSpline
from settings import (
    BASEPATH,
    LIVETIME,
    GAMMA_ASTRO,
    PHI_ASTRO,
    poles,
    ra_width,
    interpolation_method,
    GEN2_FACTOR,
)
import settings as st
from tools import get_mids, array_source_interp, read_effective_area


# with three dimensions: sindec, energy, ra
def aeff_eval(aeff, sindec_width, e_width, ra_width):
    return ((aeff * sindec_width).T * e_width)[:, :, np.newaxis] * np.atleast_2d(
        ra_width
    )


# with two dimensions: sindec, energy
def aeff_eval_e_sd(aeff, sindec_width, e_width, ra_width=2 * np.pi):
    """
    Evaluate the effective area for given sin(declination), energy, and right ascension width.

    Parameters:
    - aeff (float or numpy.ndarray): Effective area values.
    - sindec_width (numpy.ndarray): Array of sin(declination) bin widths.
    - e_width (float or numpy.ndarray): Energy bin width.
    - ra_width (float or numpy.ndarray, optional): Right ascension bin width.
      Defaults to 2 * pi if not provided.

    Returns:
    float or numpy.ndarray: Effective area evaluated for the given parameters.

    """
    ra_width = np.atleast_1d(ra_width)
    return aeff * sindec_width[:, np.newaxis] * e_width * np.sum(ra_width)


def calc_aeff_factor(aeff, ewidth, livetime, **config):
    """
    Calculate the factor for effective area multiplied by livetime and appropriate bin widths
    for flux integration to obtain the expected number of events.

    Parameters:
    -----------
    aeff : numpy.ndarray
        Effective area values.

    ewidth : float
        Energy bin width.

    livetime : float
        Livetime for the observation.

    Keyword Parameters:
    -------------------
    diff_or_ps : str, default='ps'
        Determines whether to perform a point source ('ps') or diffuse ('diff') calculation.

    dec : float, default=0
        Declination angle in *radian*.

    sindec_mids : numpy.ndarray
        Array of sin(declination) midpoints for point source calculation.

    sindec_width : numpy.ndarray
        Array of sin(declination) bin widths for diffuse calculation.

    dpsi_max : float, default=0
        Maximum angular distance for point source calculation.

    grid_2d : int, default=1
        Grid dimensionality for point source calculation (2D grid for point source, 1 for other calculations).

    Returns:
    -------
    numpy.ndarray
        The factor for effective area multiplied by livetime and bin widths.

    Notes:
    ------
    - For 'ps' (point source) calculation, the factor is determined based on declination, sindec midpoints,
      maximum angular distance, and the specified grid dimensionality.
    - For 'diff' (diffuse) calculation, the factor is determined using sindec bin widths.
    - Ensure that 'diff_or_ps' is set to either 'diff' or 'ps'.
    """
    diff_or_ps = config.pop("diff_or_ps", "ps")

    if diff_or_ps == "ps":
        dec = config.pop("dec", 0)
        sindec_mids = config.pop("sindec_mids")
        dpsi_max = config.pop("dpsi_max", 0)
        grid_2d = config.pop("grid_2d", 1)

        # Get the appropriate aeff slice based on the chosen declination
        aeff_factor = (
            grid_2d
            * array_source_interp(dec, aeff, sindec_mids, axis=1)
            * livetime
            * ewidth
        )

        if dpsi_max > 0:
            # Solid angle integration for background aeff factor
            aeff_factor *= (
                np.deg2rad(dpsi_max) ** 2 * np.pi
            )  # Solid angle approximation

    elif diff_or_ps == "diff":
        sindec_width = config.pop("sindec_width")
        aeff_factor = aeff_eval_e_sd(aeff, sindec_width, ewidth) * livetime

    else:
        raise ValueError(
            f"Invalid value for 'diff_or_ps': {diff_or_ps}. Must be 'diff' or 'ps'."
        )

    return aeff_factor


def setup_aeff_grid(
    aeff_baseline,
    sindec_mids_,
    ra_,
    ra_width_,
    local=False,
    log_int=False,
    method=interpolation_method,
):
    """
    Build a RegularGridInterpolator from the effective area, and make the corres-
    ponding coordinate grid for evaluation. Note that the effective area can be
    interpolated as log(aeff).

    We're a bit sloppy with the naming here and use sindec binning for any case
    but since sin(dec) = cos(theta) for IceCube, this works
    and the ordering in local coordinates stays the same also for other detectors"""

    grid2d = []

    # pad the arrays so that we don't get ugly edge effects
    padded_sindec_mids = np.concatenate([[-1], sindec_mids_, [1]])
    # loop over rows of aeff = per slice in energy
    for aeff in aeff_baseline:
        padded_aeff = np.pad(aeff, pad_width=1, mode="edge")

        if not local:
            # this is the IceCube case, where we need to spin
            # a_eff "upside down" from equatorial to local coordinates
            padded_aeff = (
                padded_aeff[::-1, np.newaxis] / ra_width_ * np.ones((1, len(ra_)))
            )
        else:
            padded_aeff = (
                padded_aeff[:, np.newaxis] / ra_width_ * np.ones((1, len(ra_)))
            )
        # added ra as new axis and normalize accordingly
        padded_aeff /= len(ra_)
        # pad the arrays so that we don't get ugly edge effects
        grid2d.append(
            RegularGridInterpolator(
                (padded_sindec_mids, ra_),
                np.log(padded_aeff) if log_int else padded_aeff,
                method=method,  # pchip is slow, but accurate
                bounds_error=False,
                fill_value=None,
            )
        )
    # grid elements are calculated for each energy bin, grid is theta x phi
    # coordinate grid in equatorial coordinates (icrs)
    # these will be the integration coordinates (without padding)
    pp, tt = np.meshgrid(ra_, np.arcsin(sindec_mids_))
    eq_coords = SkyCoord(pp * u.radian, tt * u.radian, frame="icrs")
    return grid2d, eq_coords


def earth_rotation(
    coord_lat,
    coord_lon,
    eq_coords,
    hp_coords,
    grid2d,
    _ra_width,
    log_aeff=False,
    return_3D=False,
    time=None,
):
    """
    Idea: transform the integration over R.A. per sin(dec) into local coordinates

    Disclaimer: the local coordinate trafo will not 100%-accurately recover
    the original coordinates if you're at the north pole. it accounts for the
    minuscle wobble of Earth's axis wrt. equatorial coordinates.

    """
    # local detector
    loc = EarthLocation(lat=coord_lat, lon=coord_lon)
    # arbitrary time, doesnt really matter here
    if time is None:
        time = Time("2025-01-01 12:00:00")
    # transform integration coordinates to local frame
    local_coords = hp_coords.transform_to(AltAz(obstime=time, location=loc))
    # these local coordinates match the coordinates of the A_eff in grid2d

    # sum up the contributions over the transformed RA axis per declination
    # loop over the energy bins to get the same shape of aeff as before
    # sum along transformed ra coordinates
    new_aeff = []
    for _grid in grid2d:
        if log_aeff:
            rot_aeff = np.exp(
                _grid((np.sin(local_coords.alt.rad), local_coords.az.rad))
                * _ra_width[0]
            )
        else:
            rot_aeff = (
                _grid((np.sin(local_coords.alt.rad), local_coords.az.rad))
                * _ra_width[0]
            )
        new_aeff.append(
            hp.get_interp_val(
                rot_aeff,
                np.pi / 2 - eq_coords.dec.radian,
                np.pi - eq_coords.ra.radian,
            )
        )
    new_aeff = np.array(new_aeff)
    if return_3D:
        return new_aeff
    else:
        return np.sum(new_aeff, axis=2)


def aeff_rotation(coord_lat, coord_lon, eq_coords, grid2d, ra_width, log_aeff=False):
    """
    ### OLD !!! ### see above (earth_rotation) for new version.

    Idea: transform the integration over R.A. per sin(dec) into local coordinates

    Disclaimer: the local coordinate trafo will not 100%-accurately recover
    the original coordinates if you're at the north pole. it accounts for the
    minuscle wobble of Earth's axis wrt. equatorial coordinates.

    """
    # local detector
    loc = EarthLocation(lat=coord_lat, lon=coord_lon)
    # arbitrary time, doesnt matter here
    time = Time("2021-6-21 00:00:00")
    # transform integration coordinates to local frame
    local_coords = eq_coords.transform_to(AltAz(obstime=time, location=loc))
    # these local coordinates match the coordinates of the A_eff in grid2d

    # sum up the contributions over the transformed RA axis per declination
    # loop over the energy bins to get the same shape of aeff as before
    # sum along transformed ra coordinates
    if log_aeff:
        return np.array(
            [
                np.sum(
                    np.exp(
                        grid2d[i]((np.sin(local_coords.alt.rad), local_coords.az.rad))
                    )
                    * ra_width,  # integrate over RA
                    axis=1,
                )
                for i in range(len(grid2d))
            ]
        )
    else:
        return np.array(
            [
                np.sum(
                    grid2d[i]((np.sin(local_coords.alt.rad), local_coords.az.rad))
                    * ra_width,  # integrate over RA
                    axis=1,
                )
                for i in range(len(grid2d))
            ]
        )


def get_aeff_and_binnings(key="full", verbose=False):
    """
    key: "full" for full effective area
    "upgoing" for effective area >-5deg

    Returns:
    aeff_2d, log_ebins, ebins, sindec_bins
    """
    with open(
        join(BASEPATH, f"resources/tabulated_logE_sindec_aeff_{key}.pckl"), "rb"
    ) as f:
        log_ebins, sindec_bins, aeff_2d = pickle.load(f)
    ebins = np.power(10, log_ebins)
    if verbose:
        print(len(ebins) - 1, "log_10(energy) bins")
        print(len(sindec_bins) - 1, "declination bins")
    return aeff_2d, log_ebins, ebins, sindec_bins


def padded_interpolation(array, *bins, **rgi_kwargs):
    if np.ndim(array) == 1:
        bins = (bins,)
    assert np.ndim(array) == len(bins), "axes dimensions don't match"

    # get coordinate mids
    mids = []
    for b in bins:
        mids.append(get_mids(b))

    # add the bin boundaries back in at the beginning and end
    padded_mids = []
    for i, m in enumerate(mids):
        padded_mids.append(np.concatenate([[bins[i][0]], m, [bins[i][-1]]]))

    # return the rgi of the array padded with its edge values
    # mode="constant", constant_values=0
    return RegularGridInterpolator(
        padded_mids, np.pad(array, 1, mode="edge"), **rgi_kwargs
    )


def setup_coordinates(nside=2**8, ra_mids=st.ra_mids, sindec_mids=st.sindec_mids):
    # Healpy interpolation and rotation setup
    npix = hp.nside2npix(nside)
    pix = np.arange(npix)
    hp_angles = hp.pix2ang(nside, pix)

    # binning setup
    _azi = hp_angles[1]
    _zen = hp_angles[0] - np.pi / 2

    # for rotation
    hp_coords = SkyCoord(_azi * u.radian, _zen * u.radian, frame="icrs")

    # for integration
    pp, tt = np.meshgrid(ra_mids, np.arcsin(sindec_mids))
    eq_coords = SkyCoord(pp * u.radian, tt * u.radian, frame="icrs")
    return hp_coords, eq_coords


if __name__ == "__main__":
    # get all info from data release first
    public_data_aeff = read_effective_area()
    # log10(E_nu/GeV)_min log10(E_nu/GeV)_max
    # Dec_nu_min[deg] Dec_nu_max[deg]
    # A_Eff[cm^2]

    # the file contains all bin edges

    sindec_bins = np.unique(
        np.sin(np.deg2rad([public_data_aeff.Dec_nu_min, public_data_aeff.Dec_nu_max]))
    )
    sindec_bins = np.round(sindec_bins, 2)
    sindec_mids = get_mids(sindec_bins)
    sindec_width = np.diff(sindec_bins)

    log_ebins = np.unique([public_data_aeff.logE_nu_min, public_data_aeff.logE_nu_max])
    ebins = np.power(10, log_ebins)
    emids = np.power(10, get_mids(log_ebins))
    ewidth = np.diff(ebins)

    # re-shape into 2D array with (A(E) x A(delta))
    # and switch the eff area ordering
    aeff_icecube_full = (
        public_data_aeff["A_eff"].values.reshape(len(sindec_mids), len(emids)).T
    )
    # some event numbers for checking
    aeff_factor = (
        (aeff_icecube_full * sindec_width).T * ewidth * LIVETIME * np.sum(ra_width)
    )
    astro_ev = aeff_factor * (emids / 1e5) ** (-GAMMA_ASTRO) * PHI_ASTRO
    print("icecube (full) astro events:", np.sum(astro_ev))

    # cut at delta > -5deg
    min_idx = np.searchsorted(sindec_mids, np.sin(np.deg2rad(-5)))
    print(
        f"Below {np.rad2deg(np.arcsin(sindec_bins[min_idx])):1.2f} deg, A_eff is set to 0"
    )
    aeff_icecube_upgoing = np.copy(aeff_icecube_full)
    aeff_icecube_upgoing[:, :min_idx] = 0

    # some event numbers for checking
    aeff_factor = (
        (aeff_icecube_upgoing * sindec_width).T * ewidth * LIVETIME * np.sum(ra_width)
    )
    astro_ev = aeff_factor * (emids / 1e5) ** (-GAMMA_ASTRO) * PHI_ASTRO
    print("icecube (upgoing) astro events:", np.sum(astro_ev))

    aeff_i = {}
    # aeff_i["Plenum-1"] = np.zeros_like(aeff_2d["icecube"]) # should be added via event rate, if at all
    grid2d, _ = setup_aeff_grid(
        aeff_icecube_upgoing,
        sindec_mids,
        st.ra_mids,
        st.ra_width,
        local=True,
        log_int=False,
    )
    hp_coords, eq_coords = setup_coordinates()

    # loop over detectors
    detectors = [
        "IceCube",
        "P-ONE",
        "KM3NeT",
        "Baikal-GVD",
        "TRIDENT",
        "HUNT",
        "NEON",
        "Horizon",
    ]
    for k in detectors:
        aeff_i[k] = earth_rotation(
            poles[k]["lat"],
            poles[k]["lon"],
            eq_coords,
            hp_coords,
            grid2d,
            ra_width,
            log_aeff=False,
        )
    # some event numbers for checking
    aeff_factor = (
        (aeff_i["IceCube"] * st.sindec_width).T
        * ewidth
        * LIVETIME
        * np.sum(st.ra_width)
    )
    astro_ev = aeff_factor * (emids / 1e5) ** (-GAMMA_ASTRO) * PHI_ASTRO
    print("icecube (after rotation, upgoing) astro events:", np.sum(astro_ev))

    ## GEN-2 will have ~7.5x effective area ==> 5times better discovery potential
    aeff_i["Gen-2"] = aeff_i["IceCube"] * GEN2_FACTOR
    aeff_i["TRIDENT"] *= st.TRIDENT_FACTOR
    aeff_i["NEON"] *= st.NEON_FACTOR
    aeff_i["HUNT"] *= st.HUNT_FACTOR

    ## save to disc
    savefile = join(BASEPATH, "resources/tabulated_logE_sindec_aeff_upgoing.pckl")
    print("Saving up-going effective areas to", savefile)
    with open(savefile, "wb") as f:
        pickle.dump((log_ebins, st.sindec_bins, aeff_i), f)

    # same but wit FULL icecube effective area
    print("starting full effective area calculation...")
    grid2d, _ = setup_aeff_grid(
        aeff_icecube_full,
        sindec_mids,
        st.ra_mids,
        st.ra_width,
        local=True,
        log_int=False,
    )
    aeff_i_full = {}
    # aeff_i_full["Plenum-1"] = np.zeros_like(aeff_2d["icecube_full"]) # should be added via event rate, if at all

    # loop over detectors
    for k in detectors:
        aeff_i_full[k] = earth_rotation(
            poles[k]["lat"],
            poles[k]["lon"],
            eq_coords,
            hp_coords,
            grid2d,
            ra_width,
            log_aeff=False,
        )
    # some event numbers for checking
    aeff_factor = (
        (aeff_i_full["IceCube"] * st.sindec_width).T
        * ewidth
        * LIVETIME
        * np.sum(st.ra_width)
    )
    astro_ev = aeff_factor * (emids / 1e5) ** (-GAMMA_ASTRO) * PHI_ASTRO
    print("icecube (after rotation, full) astro events:", np.sum(astro_ev))
    # GEN-2 will have ~7.5x effective area ==> 5times better discovery potential
    aeff_i_full["Gen-2"] = aeff_i_full["IceCube"] * GEN2_FACTOR
    aeff_i_full["TRIDENT"] *= st.TRIDENT_FACTOR
    aeff_i_full["NEON"] *= st.NEON_FACTOR
    aeff_i_full["HUNT"] *= st.HUNT_FACTOR

    # save
    savefile = join(BASEPATH, "resources/tabulated_logE_sindec_aeff_full.pckl")
    print("Saving full effective areas to", savefile)
    with open(savefile, "wb") as f:
        pickle.dump((log_ebins, st.sindec_bins, aeff_i_full), f)

    print("finished!")
