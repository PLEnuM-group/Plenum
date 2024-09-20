import numpy as np
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
        Declination angle in degrees.

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
    sindec_mids,
    ra_,
    ra_width,
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
    padded_sindec_mids = np.concatenate([[-1], sindec_mids, [1]])
    # loop over rows of aeff = per slice in energy
    for aeff in aeff_baseline:
        padded_aeff = np.pad(aeff, pad_width=1, mode="edge")

        if not local:
            # this is the IceCube case, where we need to spin
            # a_eff "upside down" from equatorial to local coordinates
            padded_aeff = (
                padded_aeff[::-1, np.newaxis] / ra_width * np.ones((1, len(ra_)))
            )
        else:
            padded_aeff = padded_aeff[:, np.newaxis] / ra_width * np.ones((1, len(ra_)))
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
    pp, tt = np.meshgrid(ra_, np.arcsin(sindec_mids))
    eq_coords = SkyCoord(pp * u.radian, tt * u.radian, frame="icrs")
    return grid2d, eq_coords


def aeff_rotation(coord_lat, coord_lon, eq_coords, grid2d, ra_width, log_aeff=False):
    """
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
    x = np.arange(0, aeff_icecube_full.shape[1])
    y = np.arange(0, aeff_icecube_full.shape[0])
    # mask invalid values
    array = np.ma.masked_invalid(np.log10(aeff_icecube_full))
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    GD1 = griddata((x1, y1), newarr.ravel(), (xx, yy), method="linear")
    GD1[np.isnan(GD1)] = -7

    interp_aeff = 10**GD1
    interp_aeff[np.isnan(interp_aeff)] = 0
    smth_aeff_spl = RectBivariateSpline(x, y, GD1.T, s=20)

    aeff_icecube_full = 10 ** smth_aeff_spl(x, y).T
    aeff_icecube_full[aeff_icecube_full < 1e-4] = 0

    # some event numbers for checking
    aeff_factor = (
        (aeff_icecube_full * sindec_width).T * ewidth * LIVETIME * np.sum(ra_width)
    )
    astro_ev = aeff_factor * (emids / 1e5) ** (-GAMMA_ASTRO) * PHI_ASTRO
    print("icecube (full) astro events:", np.sum(astro_ev))
    # should be something like 2300 neutrinos for given parameters

    # finer interpolation for further steps
    rgi = padded_interpolation(
        aeff_icecube_full, ebins, sindec_bins, method=interpolation_method
    )
    ss, em = np.meshgrid(st.sindec_mids, st.emids)
    aeff_2d = dict()
    aeff_2d["icecube_full"] = rgi((em, ss))  # np.exp(rgi((em, ss)))

    # cut at delta > -5deg
    min_idx = np.searchsorted(st.sindec_mids, np.sin(np.deg2rad(-5)))
    print(
        f"Below {np.rad2deg(np.arcsin(st.sindec_bins[min_idx])):1.2f} deg, A_eff is set to 0"
    )
    aeff_2d["icecube"] = np.copy(aeff_2d["icecube_full"])
    aeff_2d["icecube"][:, :min_idx] = 0

    print("starting aeff rotations")
    grid2d, eq_coords = setup_aeff_grid(
        aeff_2d["icecube"], st.sindec_mids, st.ra_mids, st.ra_width, log_int=False
    )
    aeff_i = {}
    aeff_i["Plenum-1"] = np.zeros_like(aeff_2d["icecube"])

    # loop over detectors
    for k in ["IceCube", "P-ONE", "KM3NeT", "Baikal-GVD"]:
        aeff_i[k] = aeff_rotation(
            poles[k]["lat"],
            poles[k]["lon"],
            eq_coords,
            grid2d,
            ra_width,
            log_aeff=False,
        )
        aeff_i["Plenum-1"] += aeff_i[k]

    ## GEN-2 will have ~7.5x effective area ==> 5times better discovery potential
    aeff_i["Gen-2"] = aeff_i["IceCube"] * GEN2_FACTOR
    ## in plenum-2, IC is replaced by Gen-2
    aeff_i["Plenum-2"] = aeff_i["Plenum-1"] - aeff_i["IceCube"] + aeff_i["Gen-2"]

    ## save to disc
    savefile = join(BASEPATH, "resources/tabulated_logE_sindec_aeff_upgoing.pckl")
    print("Saving up-going effective areas to", savefile)
    with open(savefile, "wb") as f:
        pickle.dump((st.logE_bins, st.sindec_bins, aeff_i), f)

    # same but wit FULL icecube effective area
    print("starting full effective area calculation...")
    grid2d, eq_coords = setup_aeff_grid(
        aeff_2d["icecube_full"], st.sindec_mids, st.ra_mids, st.ra_width, log_int=False
    )

    aeff_i_full = {}
    aeff_i_full["Plenum-1"] = np.zeros_like(aeff_2d["icecube_full"])

    # loop over detectors
    for k in ["IceCube", "P-ONE", "KM3NeT", "Baikal-GVD"]:
        aeff_i_full[k] = aeff_rotation(
            poles[k]["lat"],
            poles[k]["lon"],
            eq_coords,
            grid2d,
            ra_width,
            log_aeff=False,
        )
        aeff_i_full["Plenum-1"] += aeff_i_full[k]

    # GEN-2 will have ~7.5x effective area ==> 5times better discovery potential
    aeff_i_full["Gen-2"] = aeff_i_full["IceCube"] * GEN2_FACTOR
    aeff_i_full["Plenum-2"] = (
        aeff_i_full["Plenum-1"] - aeff_i_full["IceCube"] + aeff_i_full["Gen-2"]
    )

    # save
    savefile = join(BASEPATH, "resources/tabulated_logE_sindec_aeff_full.pckl")
    print("Saving full effective areas to", savefile)
    with open(savefile, "wb") as f:
        pickle.dump((st.logE_bins, st.sindec_bins, aeff_i_full), f)

    print("finished!")
