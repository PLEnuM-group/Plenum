import numpy as np
from os.path import exists, join
import pickle
from scipy.stats import gaussian_kde
from scipy.interpolate import UnivariateSpline, RegularGridInterpolator
from scipy.optimize import curve_fit
from scipy.special import erf
import settings as st
from tools import get_mids
from mephisto import Mephistogram


def energy_smearing(ematrix, ev):
    """Matrix multiplication with the energy resolution
    to translate an event histogram from E_true to E_reco.

    Expected formats: Both Mephistograms
    ematrix: logE x logE_reco
    ev: any x logE
    """
    if isinstance(ematrix, Mephistogram):
        # try both options
        try:
            return ev @ ematrix.T()
        except ValueError:
            return ev @ ematrix
    else:  # backward compatibility
        return (ematrix @ ev.T).T


def get_baseline_energy_res(step_size=0.1, renew_calc=False):
    """TODO"""

    filename = join(st.LOCALPATH, f"energy_smearing_2D_step-{step_size}.pckl")
    if exists(filename) and not renew_calc:
        print("file exists:", filename)
        with open(filename, "rb") as f:
            all_grids, logE_bins, logE_reco_bins = pickle.load(f)
    else:
        print("calculating grids...")
        # energy binning
        logE_bins = np.arange(st.E_MIN, st.E_MAX + step_size, step=step_size)
        logE_reco_bins = np.arange(st.E_MIN, st.E_MAX, step=step_size)
        logE_mids = get_mids(logE_bins)
        logE_reco_mids = get_mids(logE_reco_bins)
        ee, rr = np.meshgrid(logE_mids, logE_reco_mids)

        # load smearing matrix
        public_data_hist = np.genfromtxt(
            join(st.BASEPATH, "resources/IC86_II_smearing.csv"), skip_header=1
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


def get_energy_psf_grid(logE_bins, delta_psi_max=2, bins_per_psi2=25, renew_calc=False):
    """

    Calculate the 2D grids of the resolution matrix in log10(energy) and
    psi^2 (angular distance from source, squared) as a function of zenith.
    If the file already exists, it will be loaded from disk,
    otherwise it will be calculated and saved to disc.
    The 'renew_calc' argument will force the calculation even if the file already
    exists (see below).

    Formula:
    $f_x(x) = kde(x)$ with $ x = \log_{10}(y) \Leftrightarrow y = 10^x$
    Transform: $z = y² = 10^{(2\cdot x)}$ with $ x = \frac{\log_{10}(z)}{2} := g(z)$
    $\Rightarrow f_z(z) = | \frac{d}{dz} g(z) | \cdot f(g(z)) = \frac{1}{2\cdot z \cdot \log(10)} kde(\frac{\log_{10}(z)}{2})$

    Parameters:
    -----------
    logE_bins: array, floats
        Bins of the logE axis of the 2D grid used for evaluation.
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
    logE_mids = get_mids(logE_bins)
    # energy-PSF function
    filename = join(
        st.LOCALPATH,
        f"e_psf_grid_psimax-{delta_psi_max}_bins-{bins_per_psi2}.pckl",
    )
    if exists(filename) and not renew_calc:
        print("file exists:", filename)
        with open(filename, "rb") as f:
            all_grids, psi2_bins, logE_bins = pickle.load(f)
    else:
        print("calculating grids...")
        public_data_hist = np.genfromtxt(
            join(st.BASEPATH, "resources/IC86_II_smearing.csv"), skip_header=1
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
            grid_tmp = psi_kvals / psi_grid / 2 / np.log(10)

            all_grids[f"dec-{dd}"] = grid_tmp / np.sum(grid_tmp, axis=0)
        with open(filename, "wb") as f:
            pickle.dump((all_grids, psi2_bins, logE_bins), f)
        print("file saved to:", filename)
    return all_grids, psi2_bins, logE_bins


def double_erf(x, shift_l, shift_r, N=1):
    sigma_r = sigma_l = 0.4
    # normalized such that it goes from 0 to N and back to 0
    return (
        N / 4 * (erf((x - shift_l) / sigma_l) + 1) * (-erf((x - shift_r) / sigma_r) + 1)
    )


def g_norm(x, loc, scale, N):
    return np.exp(-0.5 * ((x - loc) / scale) ** 2) * N


def comb(x, shift_l, shift_r, N, loc, scale, n):
    return double_erf(x, shift_l, shift_r, N) + g_norm(x, loc, scale, n)


def fit_eres_params(eres_mephisto):
    """energy-resolution mephistogram logEreco - logE axes

    Note that the fit parameters are tuned and hardcoded to work
    with the chosen bins, settings, and resolution boundaries.
    The fit might not work well for other binnings or resolutions.

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
                    flanks[1] - 0.6,  # shift_l
                    20,  # shift_r
                    0.037 / 2,  # N
                    9,  # loc
                    10,  # scale
                    0.3,  # n
                ),
            ],
        )
        fit_params[ii] = tuple(fit_d)
    return fit_params


def smooth_eres_fit_params(fit_params, logE_mids, s=40, k=1):
    fit_splines = {}
    smoothed_fit_params = np.zeros_like(fit_params)
    for n in fit_params.dtype.names:
        smoothing_factor = (
            np.max(fit_params[n]) / s if n != "N" else np.max(fit_params[n]) / s / 10
        )
        fit_splines[n] = UnivariateSpline(
            logE_mids,
            fit_params[n],
            k=k,
            s=smoothing_factor,
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


if __name__ == "__main__":
    # Build resolution functions and save them as mephistograms
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--renew_calc", action="store_true")
    args = parser.parse_args()

    # Psi² - Energy resolution
    # already in right binning
    psi_e_res, _, _ = get_energy_psf_grid(
        st.logE_bins,
        delta_psi_max=st.delta_psi_max,
        bins_per_psi2=st.bins_per_psi2,
        renew_calc=args.renew_calc,
    )  # function writes these also to disk
    for k in psi_e_res:
        psi_e_res[k] = Mephistogram(
            psi_e_res[k],
            (st.psi2_bins, st.logE_bins),
            ("Psi**2", "log(E/GeV)"),
            make_hist=False,
        )
    # save to disk
    with open(join(st.LOCALPATH, "Psi2_res_mephistograms.pckl"), "wb") as f:
        pickle.dump(psi_e_res, f)

    # generate standard energy resolution
    all_E_grids, logE_bins_old, logE_reco_bins_old = get_baseline_energy_res(
        renew_calc=args.renew_calc
    )
    logE_mids_old = get_mids(logE_bins_old)
    logE_reco_mids_old = get_mids(logE_reco_bins_old)

    pad_logE = np.concatenate([[logE_bins_old[0]], logE_mids_old, [logE_bins_old[-1]]])
    pad_reco = np.concatenate(
        [[logE_reco_bins_old[0]], logE_reco_mids_old, [logE_reco_bins_old[-1]]]
    )
    # update to common binning
    lge_grid, lre_grid = np.meshgrid(st.logE_mids, st.logE_reco_mids)
    all_E_histos = {}
    for k in all_E_grids:
        eres_rgi = RegularGridInterpolator(
            (pad_reco, pad_logE),
            np.pad(all_E_grids[k], 1, mode="edge"),
            bounds_error=False,
            fill_value=1e-16,
        )
        eres_tmp = eres_rgi((lre_grid, lge_grid))
        eres_tmp[np.isnan(eres_tmp)] = 0
        # normalize
        eres_tmp /= np.sum(eres_tmp, axis=0)

        all_E_histos[k] = Mephistogram(
            eres_tmp.T,
            (st.logE_bins, st.logE_reco_bins),
            ("log(E/GeV)", "log(E_reco/GeV)"),
            make_hist=False,
        )
    # save to disk
    with open(join(st.LOCALPATH, "Eres_mephistograms.pckl"), "wb") as f:
        pickle.dump(all_E_histos, f)

    # combine horizontal and upgoing resolutions
    eres_up_mh = all_E_histos["dec-0.0"] + all_E_histos["dec-50.0"]
    eres_up_mh.normalize(axis=1)  # normalize per log(E)

    with open(join(st.LOCALPATH, "energy_smearing_MH_up.pckl"), "wb") as f:
        pickle.dump(eres_up_mh, f)

    # we need the transposed matrix for further calculations
    eres_up_T = eres_up_mh.T()
    # Parameterize the smearing matrix
    fit_params = fit_eres_params(eres_up_T)
    np.save(join(st.LOCALPATH, "Eres_fits.npy"), fit_params)
    smoothed_fit_params = smooth_eres_fit_params(
        fit_params, eres_up_T.bin_mids[1], s=40, k=3
    )
    np.save(join(st.LOCALPATH, "Eres_fits_smoothed.npy"), smoothed_fit_params)

    # Artificial resolution matrices
    ## Best reproduction based on the fit parameters
    artificial_2D = artificial_eres(fit_params, *eres_up_T.bins)
    artificial_2D.axis_names = eres_up_T.axis_names
    with open(join(st.LOCALPATH, "artificial_energy_smearing_MH_up.pckl"), "wb") as f:
        pickle.dump(artificial_2D.T(), f)

    ## 1:1 reco reproduction
    artificial_one2one = one2one_eres(smoothed_fit_params, *eres_up_T.bins)
    artificial_one2one.axis_names = eres_up_T.axis_names
    with open(
        join(st.LOCALPATH, "idealized_artificial_energy_smearing_MH_up.pckl"), "wb"
    ) as f:
        pickle.dump(artificial_one2one.T(), f)

    ## Improved artificial energy smearing
    for ii, impro_factor in enumerate([0.1, 0.2, 0.5]):

        artificial_2D_impro = improved_eres(
            impro_factor, smoothed_fit_params, *eres_up_T.bins
        )
        artificial_2D_impro.axis_names = eres_up_T.axis_names
        filename = join(
            st.LOCALPATH,
            f"improved_{impro_factor}_artificial_energy_smearing_MH_up.pckl",
        )
        with open(filename, "wb") as f:
            pickle.dump(artificial_2D_impro.T(), f)
