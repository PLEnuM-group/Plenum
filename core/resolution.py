import numpy as np
from os.path import exists, join
import pickle
from scipy.stats import gaussian_kde
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.special import erf
from settings import BASEPATH, E_MIN, E_MAX
from tools import get_mids
from mephisto import Mephistogram


def energy_smearing(ematrix, ev):
    """Matrix multiplication to translate from E_true to E_reco"""
    return (ematrix @ ev.T).T


def get_baseline_energy_res(step_size=0.1, region="up", renew_calc=False):
    """TODO"""

    filename = join(BASEPATH, f"local/energy_smearing_2D_step-{step_size}.pckl")
    if exists(filename) and not renew_calc:
        print("file exists:", filename)
        with open(filename, "rb") as f:
            all_grids, logE_bins, logE_reco_bins = pickle.load(f)
    else:
        print("calculating grids...")
        # energy binning
        logE_bins = np.arange(E_MIN, E_MAX + step_size, step=step_size)
        logE_reco_bins = np.arange(E_MIN, E_MAX, step=step_size)
        logE_mids = get_mids(logE_bins)
        logE_reco_mids = get_mids(logE_reco_bins)
        ee, rr = np.meshgrid(logE_mids, logE_reco_mids)

        # load smearing matrix
        public_data_hist = np.genfromtxt(
            join(BASEPATH, "resources/IC86_II_smearing.csv"), skip_header=1
        )

        log_sm_emids = (public_data_hist[:, 0] + public_data_hist[:, 1]) / 2.0
        log_sm_ereco_mids = (public_data_hist[:, 4] + public_data_hist[:, 5]) / 2.0
        fractional_event_counts = public_data_hist[:, 10]
        dec_sm_min, dec_sm_max = public_data_hist[:, 2], public_data_hist[:, 3]
        dec_sm_mids = (dec_sm_min + dec_sm_max) / 2.0

        # down-going (=South): -90 -> -10 deg
        # horizontal: -10 -> 10 deg
        # up-going (=North): 10 -> 90 deg
        all_grids = {}

        # loop over declination bins
        for dd in np.unique(dec_sm_mids):
            dec_mask = dec_sm_mids == dd
            # energy resolution per declination bin
            e_ereco_kdes = gaussian_kde(
                (log_sm_emids[dec_mask], log_sm_ereco_mids[dec_mask]),
                weights=fractional_event_counts[dec_mask],
            )
            all_grids[f"dec-{dd}"] = e_ereco_kdes([ee.flatten(), rr.flatten()]).reshape(
                len(logE_reco_mids), len(logE_mids)
            )
            all_grids[f"dec-{dd}"] /= np.sum(all_grids[f"dec-{dd}"], axis=0)

        with open(filename, "wb") as f:
            pickle.dump((all_grids, logE_bins, logE_reco_bins), f)

    return all_grids, logE_bins, logE_reco_bins


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
        f"local/e_psf_grid_psimax-{delta_psi_max}_bins-{bins_per_psi2}.pckl",
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


def double_erf(x, shift_l, shift_r, N=1):  # sigma_r, , sigma_l
    sigma_r = sigma_l = 0.4
    # normalized such that it goes from 0 to N and back to 0
    return (
        N / 4 * (erf((x - shift_l) / sigma_l) + 1) * (-erf((x - shift_r) / sigma_r) + 1)
    )


def g_norm(x, loc, scale, N):
    return np.exp(-0.5 * ((x - loc) / scale) ** 2) * N
    # return norm.pdf(x, loc, scale) * N


def comb(x, shift_l, shift_r, N, loc, scale, n):  # sigma_r, sigma_l,
    return double_erf(x, shift_l, shift_r, N) + g_norm(x, loc, scale, n)


def fit_eres_params(eres_mephisto):
    """energy-resolution mephistogram logEreco - logE axes
    Note that the fit parameters are tuned and hardcoded to work,
    the fit might not work for other binnings or resolutions.

    Black Magic.
    """
    assert "reco" in eres_mephisto.axis_names[0]

    fit_params = np.zeros_like(
        eres_mephisto.bin_mids[1],
        dtype=[
            ("shift_l", float),
            ("shift_r", float),
            ("N", float),
            ("loc", float),
            ("scale", float),
            ("n", float),
        ],
    )

    for ii, _ in enumerate(eres_mephisto.bin_mids[1]):
        # find the mode of E-reco distribution in each slice of E-true
        max_ind = np.argmax(eres_mephisto.histo[:, ii])
        # save the corresponding max value
        kv_mode = eres_mephisto.histo[max_ind, ii]
        # estimate the flanks of the normal by referring to the mode / 2
        height = kv_mode / 2

        # selection above height, then take first and last bin
        flanks = eres_mephisto.bin_mids[0][
            np.where(eres_mephisto.histo[:, ii] >= height)[0][[0, -1]]
        ]
        fit_d, _ = curve_fit(
            comb,
            eres_mephisto.bin_mids[0],
            eres_mephisto.histo[:, ii],
            p0=[
                flanks[0] - 0.5,  # shift_l
                flanks[1],  # shift_r
                0.025 / 2,  # N
                flanks[1],  # loc
                0.5,  # scale
                kv_mode,  # n
            ],
            bounds=[
                (  # lower
                    1,  # shift_l
                    flanks[0] + 0.5,  # shift_r
                    0.017 / 2,  # N
                    1,  # loc
                    0.1,  # scale
                    kv_mode * 0.85,  # n
                ),
                (  # upper
                    flanks[1] - 0.5,  # shift_l
                    20,  # shift_r
                    0.035 / 2,  # N
                    9,  # loc
                    10,  # scale
                    0.3,  # n
                ),
            ],
        )
        fit_params[ii] = tuple(fit_d)
    return fit_params


def smooth_eres_fit_params(fit_params, logE_mids, s=40):
    fit_splines = {}
    smoothed_fit_params = np.zeros_like(fit_params)
    for n in fit_params.dtype.names:
        fit_splines[n] = UnivariateSpline(
            logE_mids,
            fit_params[n],
            k=1,
            s=np.max(fit_params[n]) / s,
        )
        smoothed_fit_params[n] = tuple(fit_splines[n](logE_mids))
    smoothed_fit_params = np.array(smoothed_fit_params)

    return smoothed_fit_params


def artificial_eres(fit_params, logE_reco_bins, logE_bins):
    logE_reco_mids = get_mids(logE_reco_bins)
    artificial_2D = []
    for fit_d in fit_params:
        artificial_2D.append(comb(logE_reco_mids, *fit_d))
    artificial_2D = np.array(artificial_2D).T
    artificial_2D /= np.sum(artificial_2D, axis=0)
    artificial_2D = Mephistogram(
        artificial_2D,
        (logE_reco_bins, logE_bins),
    )
    return artificial_2D


def one2one_eres(fit_params, logE_reco_bins, logE_bins):
    logE_reco_mids = get_mids(logE_reco_bins)
    logE_mids = get_mids(logE_bins)
    artificial_2D = []
    for ii, fit_d in enumerate(fit_params):
        tmp_fit = fit_d.copy()
        diff = logE_mids[ii] - tmp_fit["loc"]
        tmp_fit["loc"] = logE_mids[ii]
        tmp_fit["shift_r"] += diff
        artificial_2D.append(comb(logE_reco_mids, *tmp_fit))
    artificial_2D = np.array(artificial_2D).T
    artificial_2D /= np.sum(artificial_2D, axis=0)
    artificial_2D = Mephistogram(
        artificial_2D,
        (logE_reco_bins, logE_bins),
    )
    return artificial_2D


def improved_eres(impro_factor, smoothed_fit_params, logE_reco_bins, logE_bins):
    logE_reco_mids = get_mids(logE_reco_bins)
    logE_mids = get_mids(logE_bins)
    artificial_2D = []
    for jj, fit_d in enumerate(smoothed_fit_params):
        sigma = fit_d["scale"] / (1 + impro_factor)

        tmp_fit = fit_d.copy()
        # improve resolution
        tmp_fit["scale"] = sigma
        tmp_fit["n"] /= 1 + impro_factor

        # improve degeneracy
        diff = logE_mids[jj] - tmp_fit["loc"]
        tmp_fit["loc"] = logE_mids[jj]
        tmp_fit["shift_r"] += diff
        combined = comb(logE_reco_mids, *tmp_fit)
        combined /= np.sum(combined)
        artificial_2D.append(combined)

    artificial_2D = np.array(artificial_2D).T
    artificial_2D /= np.sum(artificial_2D, axis=0)
    artificial_2D = Mephistogram(
        artificial_2D,
        (logE_reco_bins, logE_bins),
    )
    return artificial_2D
