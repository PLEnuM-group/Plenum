import numpy as np
from os.path import exists, join
import pickle
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde
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
    if diff_or_ps == "ps":
        dec = config.pop("dec", 0)
        sindec_mids = config.pop("sindec_mids")
        dpsi_max = config.pop("dpsi_max", 0)  ## default value will evaluate PS flux
        grid_2d = config.pop(
            "grid_2d", 1
        )  ## 2D grid for PS, or unity for other calculations
        aeff_factor = (
            array_source_interp(dec, aeff, sindec_mids) * livetime * ewidth
        ) * grid_2d
        if dpsi_max > 0:
            # solid angle integration for background flux
            aeff_factor *= np.deg2rad(dpsi_max) ** 2 * np.pi  # solid angle approx.
    elif diff_or_ps == "diff":
        sindec_width = config.pop("sindec_width")
        aeff_factor = (aeff * sindec_width).T * ewidth * 2 * np.pi * livetime
    else:
        print(diff_or_ps, "must be 'diff' or 'ps'")
    return aeff_factor


def setup_aeff_grid(aeff_baseline, sindec_mids, ra_mids, ra_width):
    # Interpolated grid of the effective area in "local" coordinates
    # (= icecube's native coordinates)
    grid2d = [
        RegularGridInterpolator(
            (np.arcsin(sindec_mids), ra_mids),  # transform dec to local theta
            # switch back to local zenith, add ra as new axis and normalize accordingly
            aeff_baseline[i][::-1, np.newaxis] / np.atleast_2d(ra_width) / len(ra_mids),
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        for i in range(len(aeff_baseline))
    ]
    # grid elements are calculated for each energy bin, grid is theta x phi
    # coordinate grid in equatorial coordinates (icrs)
    # these will be the integration coordinates
    pp, tt = np.meshgrid(ra_mids, np.arcsin(sindec_mids))
    eq_coords = SkyCoord(pp * u.radian, tt * u.radian, frame="icrs")
    return grid2d, eq_coords


def aeff_rotation(coord_lat, coord_lon, eq_coords, grid2d, ra_width):
    """
    ## Idea: transform the integration over R.A. per sin(dec) into local coordinates
    # **Note:** here, equal contributions of each detector are assumed.
    # In case one wants to use different livetimes, the effective areas have to multiplied
    # individually before calculating e.g. the expected number of astrophysical events
    """
    # local detector
    loc = EarthLocation(lat=coord_lat, lon=coord_lon)
    # arbitrary time, doesnt matter here
    time = Time("2021-6-21 00:00:00")
    # transform integration coordinates to local frame
    local_coords = eq_coords.transform_to(AltAz(obstime=time, location=loc))
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
    """key: "full" for full effective area
    "upgoing" for effective area >-5deg
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


def energy_smearing(ematrix, ev):
    return (ematrix @ ev.T).T


def get_energy_psf_grid(logE_mids, delta_psi_max=2, bins_per_psi2=25, renew_calc=False):
    """

    Calculate the 2D grids of the resolution matrix in log10(energy) and
    psi^2 (angular distance from source, squared) as a function of zenith.
    If the file already exists, it will be loaded from disk,
    otherwise it will be calculated and saved to disc.
    The 'renew_calc' argument will force the calculation even if the file already
    exists (see below).


    Parameters:
    -----------
    logE_mids: array, floats
        Mids of the logE axis of the 2D grid used for evaluation.
    delta_psi_max: number, default is 2 (degree)
        2D grid is evaluated from 0 to (delta_psi_max degrees)^2.
    bins_per_psi2: int, default is 25
        bins per square-degree, i.e. with delta_psi_max = 2 the grid will have
        2^2 * 25 = 100 bins.
    renew_calc: bool, default is False
        Force to renew the calculation even if the file already exists.


    Returns:
    --------
    all_grids: dict
        2D grids for each of the zenith bins in the smearing matrix
    psi2_bins: array
        the coordinates of the psi2 axis

    """

    # energy-PSF function
    filename = join(
        BASEPATH,
        f"resources/e_psf_grid_psimax-{delta_psi_max}_bins-{bins_per_psi2}.pckl",
    )
    if exists(filename) and not renew_calc:
        print("file exists:", filename)
        with open(filename, "rb") as f:
            all_grids, psi2_bins = pickle.load(f)
        return all_grids, psi2_bins
    else:
        print("calculating grids...")
        public_data_hist = np.genfromtxt(
            join(BASEPATH, "resources/IC86_II_smearing.csv"), skip_header=1
        )
        logE_sm_min, logE_sm_max = public_data_hist[:, 0], public_data_hist[:, 1]
        logE_sm_mids = (logE_sm_min + logE_sm_max) / 2.0
        log_psf_mids = np.log10((public_data_hist[:, 6] + public_data_hist[:, 7]) / 2.0)
        dec_sm_min, dec_sm_max = public_data_hist[:, 2], public_data_hist[:, 3]
        dec_sm_mids = (dec_sm_min + dec_sm_max) / 2.0
        fractional_event_counts = public_data_hist[:, 10]
        all_grids = {}
        for dd in np.unique(dec_sm_mids):
            mask = dec_sm_mids == dd

            ## set up the psi2-energy function and binning
            e_psi_kdes = gaussian_kde(
                (logE_sm_mids[mask], log_psf_mids[mask]),
                weights=fractional_event_counts[mask],
            )

            # psi² representation
            psi2_bins = np.linspace(
                0, delta_psi_max**2, delta_psi_max**2 * bins_per_psi2 + 1
            )
            psi2_mids = get_mids(psi2_bins)
            log_psi_mids = np.log10(np.sqrt(psi2_mids))
            # KDE was produced in log(E_true) and log(Psi)
            e_eval, psi_eval = np.meshgrid(logE_mids, log_psi_mids)
            psi_kvals = e_psi_kdes([e_eval.flatten(), psi_eval.flatten()]).reshape(
                len(log_psi_mids), len(logE_mids)
            )

            # new grid for analysis in psi^2 and e_true
            _, psi_grid = np.meshgrid(logE_mids, psi2_mids)
            all_grids[f"dec-{dd}"] = psi_kvals / psi_grid / 2 / np.log(10)
            # normalize per energy to ensure that signal event numbers are not changed
            all_grids[f"dec-{dd}"] /= np.sum(all_grids[f"dec-{dd}"], axis=0)
        with open(filename, "wb") as f:
            pickle.dump((all_grids, psi2_bins), f)
        print("file saved to:", filename)
        return all_grids, psi2_bins


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
    sindec_mids = get_mids(sindec_bins)
    sindec_width = np.diff(sindec_bins)

    ebins = np.unique(
        np.power(10, [d_public[:, 0], d_public[:, 1]])
    )  # all bin edges in order
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
        pickle.dump((np.log10(ebins), sindec_bins, aeff_i_full), f)

    print("finished!")
