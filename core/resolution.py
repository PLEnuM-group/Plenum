import numpy as np
from os.path import exists, join
import pickle
from scipy.stats import gaussian_kde
from settings import BASEPATH, E_MIN, E_MAX
from tools import get_mids


def energy_smearing(ematrix, ev):
    """Matrix multiplication to translate from E_true to E_reco"""
    return (ematrix @ ev.T).T


def get_baseline_energy_res(step_size=0.1, region="up", renew_calc=False):
    """TODO"""

    filename = join(BASEPATH, f"local/energy_smearing_2D_step-{step_size}.pckl")
    if exists(filename) and not renew_calc:
        print("file exists:", filename)
        with open(filename, "rb") as f:
            all_grids = pickle.load(f)
        return all_grids
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
            pickle.dump(all_grids, f)


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
