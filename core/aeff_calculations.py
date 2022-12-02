import numpy as np
from os.path import join
import pickle
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from scipy.interpolate import RegularGridInterpolator
from settings import BASEPATH, LIVETIME, GAMMA_ASTRO, PHI_ASTRO, poles
from tools import get_mids, array_source_interp

# with three dimensions: sindec, energy, ra
def aeff_eval(aeff, sindec_width, e_width, ra_width):
    return ((aeff * sindec_width).T * e_width)[:, :, np.newaxis] * np.atleast_2d(
        ra_width
    )


# with two dimensions: sindec, energy
def aeff_eval_e_sd(aeff, sindec_width, e_width, ra_width):
    return (aeff * sindec_width).T * e_width * np.sum(ra_width)  # = 2pi


def calc_aeff_factor(aeff, ewidth, livetime=LIVETIME, **config):
    diff_or_ps = config.pop("diff_or_ps", "ps")
    # choose diff or ps calculation
    if diff_or_ps == "ps":
        dec = config.pop("dec", 0)
        sindec_mids = config.pop("sindec_mids")
        dpsi_max = config.pop("dpsi_max", 0)  ## default value will evaluate PS flux
        grid_2d = config.pop(
            "grid_2d", 1
        )  ## 2D grid for PS, or unity for other calculations

        # get the right aeff slice that matches the chosen declination
        aeff_factor = (
            grid_2d * array_source_interp(dec, aeff, sindec_mids) * livetime * ewidth
        )
        if dpsi_max > 0:
            # solid angle integration for background aeff factor
            aeff_factor *= np.deg2rad(dpsi_max) ** 2 * np.pi  # solid angle approx.
    elif diff_or_ps == "diff":
        sindec_width = config.pop("sindec_width")
        aeff_factor = (aeff * sindec_width).T * ewidth * 2 * np.pi * livetime
    else:
        print(diff_or_ps, "must be 'diff' or 'ps'")
    return aeff_factor


def setup_aeff_grid(aeff_baseline, sindec_mids, ra_mids, ra_width, local=False):
    """
    Build a RegularGridInterpolator from the effective area, and make the corres-
    ponding coordinate grid for evaluation.

    We're a bit sloppy with the naming here and use sindec binning for any case
    but since sin(dec) = cos(theta) for IceCube, this works
    and the ordering in local coordinates stays the same also for other detectors"""

    grid2d = []

    # pad the arrays so that we don't get ugly edge effects
    sd_endvalues = (
        np.arcsin(sindec_mids)[0]
        - (np.arcsin(sindec_mids)[1] - np.arcsin(sindec_mids)[0]),
        np.arcsin(sindec_mids)[-1]
        + (np.arcsin(sindec_mids)[-1] - np.arcsin(sindec_mids)[-2]),
    )
    ra_endvalues = (ra_mids[0] - ra_width[0], ra_mids[-1] + ra_width[-1])
    padded_sindec_mids = np.pad(
        np.arcsin(sindec_mids),
        pad_width=1,
        mode="linear_ramp",
        end_values=sd_endvalues,
    )
    padded_ra_mids = np.pad(
        ra_mids,
        pad_width=1,
        mode="linear_ramp",
        end_values=ra_endvalues,
    )
    for aeff in aeff_baseline:

        if not local:
            # this is the IceCube case, where we need to spin
            # a_eff "upside down" from equatorial to local coordinates
            padded_aeff = aeff[::-1, np.newaxis] / np.atleast_2d(ra_width)
        else:
            padded_aeff = aeff[:, np.newaxis] / np.atleast_2d(ra_width)
        # added ra as new axis and normalize accordingly
        padded_aeff /= len(ra_mids)
        # pad the arrays so that we don't get ugly edge effects
        padded_aeff = np.pad(padded_aeff, pad_width=1, mode="edge")
        grid2d.append(
            RegularGridInterpolator(
                (padded_sindec_mids, padded_ra_mids),
                padded_aeff,
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )
        )
    # grid elements are calculated for each energy bin, grid is theta x phi
    # coordinate grid in equatorial coordinates (icrs)
    # these will be the integration coordinates (without padding)
    pp, tt = np.meshgrid(ra_mids, np.arcsin(sindec_mids))
    eq_coords = SkyCoord(pp * u.radian, tt * u.radian, frame="icrs")
    return grid2d, eq_coords


def aeff_rotation(coord_lat, coord_lon, eq_coords, grid2d, ra_width):
    """
    Idea: transform the integration over R.A. per sin(dec) into local coordinates
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
    return np.array(
        [
            np.sum(
                grid2d[i]((local_coords.alt.rad, local_coords.az.rad))
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
    aeff_2d, log_ebins, ebins, sindec_bins, ra_bins
    """
    with open(
        join(BASEPATH, f"resources/tabulated_logE_sindec_aeff_{key}.pckl"), "rb"
    ) as f:
        log_ebins, sindec_bins, aeff_2d = pickle.load(f)
    ebins = np.power(10, log_ebins)
    ra_bins = np.linspace(0, np.pi * 2, num=101)
    if verbose:
        print(len(ebins) - 1, "log_10(energy) bins")
        print(len(sindec_bins) - 1, "declination bins")
        print(len(ra_bins) - 1, "RA bins")
    return aeff_2d, log_ebins, ebins, sindec_bins, ra_bins


if __name__ == "__main__":
    # get all info from data release first
    d_public = np.genfromtxt(
        join(BASEPATH, "resources/IC86_II_effectiveArea.csv"), skip_header=1
    )
    # log10(E_nu/GeV)_min log10(E_nu/GeV)_max
    # Dec_nu_min[deg] Dec_nu_max[deg]
    # A_Eff[cm^2]
    # the file contains all bin edges
    aeff = d_public[:, 4]

    sindec_bins = np.unique(np.sin(np.deg2rad([d_public[:, 2], d_public[:, 3]])))
    sindec_bins = np.round(sindec_bins, 2)
    sindec_mids = get_mids(sindec_bins)
    sindec_width = np.diff(sindec_bins)

    log_ebins = np.unique([d_public[:, 0], d_public[:, 1]])
    ebins = np.power(10, log_ebins)
    emids = get_mids(ebins)
    ewidth = np.diff(ebins)

    ra_bins = np.linspace(0, np.pi * 2, num=101)
    ra_mids = get_mids(ra_bins)
    ra_width = np.diff(ra_bins)

    aeff_2d = dict()
    # re-shape into 2D array with (A(E) x A(delta))
    # and switch the eff area ordering
    aeff_2d["icecube_full"] = aeff.reshape(len(sindec_mids), len(emids)).T

    # cut at delta > -5deg
    min_idx = np.searchsorted(sindec_mids, np.sin(np.deg2rad(-5)))
    print(f"Below {np.rad2deg(np.arcsin(sindec_bins[min_idx])):1.2f}deg aeff is 0")
    aeff_2d["icecube"] = np.copy(aeff_2d["icecube_full"])
    aeff_2d["icecube"][:, :min_idx] = 0

    # some event numbers for checking
    det = "icecube"
    aeff_factor = (aeff_2d[det] * sindec_width).T * ewidth * LIVETIME * np.sum(ra_width)
    astro_ev = aeff_factor * (emids / 1e5) ** (-GAMMA_ASTRO) * PHI_ASTRO
    print(det)
    print("astro events:", np.sum(astro_ev))
    # should be something like 2300 neutrinos for given parameters

    print("starting aeff rotations")
    grid2d, eq_coords = setup_aeff_grid(
        aeff_2d["icecube"], sindec_mids, ra_mids, ra_width
    )
    aeff_i = {}
    aeff_i["Plenum-1"] = np.zeros_like(aeff_2d["icecube"])

    # loop over detectors
    for k in ["IceCube", "P-ONE", "KM3NeT", "Baikal-GVD"]:
        aeff_i[k] = aeff_rotation(
            poles[k]["lat"], poles[k]["lon"], eq_coords, grid2d, ra_width
        )
        aeff_i["Plenum-1"] += aeff_i[k]

    ## GEN-2 will have ~7.5x effective area ==> 5times better discovery potential
    aeff_i["Gen-2"] = aeff_i["IceCube"] * 5 ** (1 / 0.8)
    ## in plenum-2, IC is replaced by Gen-2
    aeff_i["Plenum-2"] = aeff_i["Plenum-1"] - aeff_i["IceCube"] + aeff_i["Gen-2"]

    ## save to disc
    savefile = join(BASEPATH, "resources/tabulated_logE_sindec_aeff_upgoing.pckl")
    print("Saving up-going effective areas to", savefile)
    with open(savefile, "wb") as f:
        pickle.dump((np.log10(ebins), sindec_bins, aeff_i), f)

    # same but wit FULL icecube effective area
    print("starting full effective area calculation...")
    grid2d, eq_coords = setup_aeff_grid(
        aeff_2d["icecube_full"], sindec_mids, ra_mids, ra_width
    )

    aeff_i_full = {}
    aeff_i_full["Plenum-1"] = np.zeros_like(aeff_2d["icecube_full"])

    # loop over detectors
    for k in ["IceCube", "P-ONE", "KM3NeT", "Baikal-GVD"]:
        aeff_i_full[k] = aeff_rotation(
            poles[k]["lat"], poles[k]["lon"], eq_coords, grid2d, ra_width
        )
        aeff_i_full["Plenum-1"] += aeff_i_full[k]

    # GEN-2 will have ~7.5x effective area ==> 5times better discovery potential
    aeff_i_full["Gen-2"] = aeff_i_full["IceCube"] * 5 ** (1 / 0.8)
    aeff_i_full["Plenum-2"] = (
        aeff_i_full["Plenum-1"] - aeff_i_full["IceCube"] + aeff_i_full["Gen-2"]
    )

    # save
    savefile = join(BASEPATH, "resources/tabulated_logE_sindec_aeff_full.pckl")
    print("Saving full effective areas to", savefile)
    with open(savefile, "wb") as f:
        pickle.dump((log_ebins, sindec_bins, aeff_i_full), f)

    print("finished!")
