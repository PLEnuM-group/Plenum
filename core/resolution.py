import numpy as np
from os.path import exists, join
import pickle
from scipy.stats import gaussian_kde
from scipy.interpolate import UnivariateSpline, RegularGridInterpolator
from scipy.optimize import curve_fit
from scipy.special import erf
import settings as st
from tools import get_mids, read_smearing_matrix
from mephisto import Mephistogram
from pandas import DataFrame
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    RationalQuadratic,
    Matern,
)


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


def get_baseline_eres(renew_calc=False):
    """Make a smooth energy resolution from the smearing matrix using gaussian processes in 1D and 2D."""
    filename = join(st.LOCALPATH, "GP_Eres_mephistograms.pckl")

    if exists(filename) and not renew_calc:
        print("file exists:", filename)
        with open(filename, "rb") as f:
            GP_mephistograms = pickle.load(f)
    else:
        public_data_df = read_smearing_matrix()
        # first step: 1D predictions per slice in neutrino energy
        kernel = C(0.1, (1e-3, 1e3)) + RationalQuadratic(
            length_scale_bounds=(1e-4, 1e2)
        )
        e_vals = np.linspace(1, 9, num=100)
        e_reso_predictions = []

        for (emin, decmin), series in public_data_df.groupby(
            ["logE_nu_min", "Dec_nu_min"]
        ):
            (emid,) = (emin + np.unique(series.logE_nu_max)) / 2.0
            (decmid,) = (decmin + np.unique(series.Dec_nu_max)) / 2.0

            if decmin == -90:
                alpha = 1e-3
                # cut out misreconstructed events, they produce shitty features
                mask = series.PSF_min < 60
                # reduce binning size
                raw_erecobins = np.unique(
                    np.concatenate([series.logE_reco_min, series.logE_reco_max])
                )[::4]

            else:
                mask = series.PSF_max < 180  # will evaluate to True everywhere
                alpha = 1.2e-4

                raw_erecobins = np.unique(
                    np.concatenate([series.logE_reco_min, series.logE_reco_max])
                )
            # add additional bin boundaries to cover the whole space
            cur_erecobins = np.concatenate(
                [
                    e_vals[e_vals < raw_erecobins[0]],
                    raw_erecobins,
                    e_vals[e_vals > raw_erecobins[-1]],
                ]
            )
            cur_ereco_mids = (
                series.loc[mask].logE_reco_min + series.loc[mask].logE_reco_max
            ) / 2.0
            h, ed = np.histogram(
                cur_ereco_mids,
                weights=series.loc[mask].Fractional_Counts,
                bins=cur_erecobins,
            )
            mids = get_mids(ed)
            # pad with zeros
            # mids = np.concatenate([[mids[0] - 0.5], mids, [mids[-1] + 0.5]])
            # h = np.concatenate([[0], h, [0]])

            gp = GaussianProcessRegressor(
                kernel=kernel, alpha=alpha, n_restarts_optimizer=5
            )
            gp.fit(mids.reshape(-1, 1), np.sqrt(h))
            prediction = gp.predict(e_vals.reshape(-1, 1)) ** 2
            # adapt prediction such that it will be 0 outside the original binning
            prediction[e_vals > np.max(series.logE_reco_max)] = 0
            prediction[e_vals < np.min(series.logE_reco_min)] = 0
            prediction /= np.sum(prediction)  # normalize
            # set small values to zero to avoid picking up random fluctuations at the borders
            prediction[prediction < 2e-4] = 0

            e_reso_predictions.append(
                {
                    "emid": emid,
                    "decmid": decmid,
                    "P_ereco": prediction,
                }
            )

        e_reso_predictions = DataFrame(e_reso_predictions)

        # second step: 2D prediction based on the smoothed 1D predictions
        GP_mephistograms = {}
        for decmid, selection in e_reso_predictions.groupby(["decmid"]):
            input_eres = np.sqrt(np.stack(selection.P_ereco)).flatten()
            input_e, input_eR = np.meshgrid(selection.emid, e_vals, indexing="ij")
            input_X = np.array([input_e.flatten(), input_eR.flatten()]).T

            kernel = C(1, (1e-3, 1e3)) + Matern(nu=5 / 2)

            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=3e-4 if decmid == -50 else 2e-5,
                n_restarts_optimizer=7,
            )
            gp.fit(input_X, input_eres)

            # Input space
            x1x2 = np.array(list(product(st.logE_mids, st.logE_reco_mids)))
            eres_pred, MSE = gp.predict(x1x2, return_std=True)
            eres_mesh = np.reshape(
                eres_pred, (len(st.logE_mids), len(st.logE_reco_mids))
            )
            eres_mesh[eres_mesh <= 3e-2] = 0

            GP_mephistograms[f"dec-{decmid}"] = Mephistogram(
                eres_mesh**2,
                bins=(st.logE_bins, st.logE_reco_bins),
                axis_names=("log(E/GeV)", "log(E_reco/GeV)"),
            )
            GP_mephistograms[f"dec-{decmid}"].normalize(axis=1)
        # save to disc
        with open(filename, "wb") as f:
            pickle.dump(GP_mephistograms, f)

    return GP_mephistograms


def get_baseline_energy_res_kde(step_size=0.1, renew_calc=False):
    """OUTDATED"""

    filename = join(st.LOCALPATH, f"energy_smearing_2D_step-{step_size}_KDE.pckl")
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
        public_data_df = read_smearing_matrix()

        log_sm_emids = (public_data_df["logE_nu_min"] + public_data_df["logE_nu_max"]) / 2.0
        log_sm_ereco_mids = (public_data_df["logE_reco_min"] + public_data_df["logE_reco_max"]) / 2.0
        fractional_event_counts = public_data_df["Fractional_Counts"]
        dec_sm_min, dec_sm_max = public_data_df["Dec_nu_min"], public_data_df["Dec_nu_max"]
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
        the bin boundaries of the psi2 axis
    logE_bins : array
        the bin boundaries of the logE axis

    """
    logE_mids = get_mids(logE_bins)
    # energy-PSF function
    filename = join(
        st.LOCALPATH,
        f"e_psf_grid_psimax-{delta_psi_max}_bins-{bins_per_psi2}_KDE.pckl",
    )
    if exists(filename) and not renew_calc:
        print("file exists:", filename)
        with open(filename, "rb") as f:
            all_grids, psi2_bins, logE_bins = pickle.load(f)
    else:
        print("calculating grids...")
        public_data_df = read_smearing_matrix()
        
        logE_sm_min, logE_sm_max = public_data_df["logE_nu_min"], public_data_df["logE_nu_max"]
        logE_sm_mids = (logE_sm_min + logE_sm_max) / 2.0
        log_psf_mids = np.log10((public_data_df["PSF_min"] + public_data_df["PSF_max"]) / 2.0)
        dec_sm_min, dec_sm_max = public_data_df["Dec_nu_min"], public_data_df["Dec_nu_max"]
        dec_sm_mids = (dec_sm_min + dec_sm_max) / 2.0
        fractional_event_counts = public_data_df["Fractional_Counts"]
        all_grids = {}

        # psi² representation
        psi2_bins = np.linspace(
            0, delta_psi_max**2, num=int(delta_psi_max**2 * bins_per_psi2) + 1
        )
        psi2_mids = get_mids(psi2_bins)
        log_psi_mids = np.log10(np.sqrt(psi2_mids))
        # KDE was produced in log(E_true) and log(Psi)
        e_eval, psi_eval = np.meshgrid(logE_mids, log_psi_mids)

        for dd in np.unique(dec_sm_mids):
            mask = dec_sm_mids == dd

            ## set up the psi2-energy function and binning
            e_psi_kdes = gaussian_kde(
                (logE_sm_mids[mask], log_psf_mids[mask]),
                weights=fractional_event_counts[mask],
            )

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


def comb(x, shift_l, N, loc, scale, n):  # shift_r
    return double_erf(x, shift_l, loc, N) + g_norm(x, loc, scale, n)


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
            ("shift_l", float),  # plateau
            ("N", float),  # plateau
            ("loc", float),  # gauss
            ("scale", float),  # gauss
            ("n", float),  # gauss
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
                2.5,  # flanks[0] + 0.1,  # shift_l
                kv_mode * 0.3,  # 0.01, # N
                flanks[1],  # loc
                0.5,  # scale
                kv_mode,  # n
            ],
            bounds=[
                (  # lower
                    2,  # flanks[0] - 0.3,  # shift_l
                    kv_mode * 0.2,  # 0.005, #0.007,  # N
                    1,  # loc
                    0.18,  # scale
                    kv_mode * 0.5,  # 0,  #  n
                ),
                (  # upper
                    3,  # flanks[0] + 1,  # shift_l
                    0.04,  # kv_mode*1.05,  # N
                    9,  # loc
                    3,  # scale
                    kv_mode * 1.05,  # 0.3,  # n
                ),
            ],
        )
        fit_params[ii] = tuple(fit_d)
    return fit_params


def smooth_eres_fit_params(fit_params, logE_mids, s=40, k=1):
    """Larger s -> larger precision (= less smoothing)"""
    fit_splines = {}
    smoothed_fit_params = np.zeros_like(fit_params)
    for n in fit_params.dtype.names:
        smoothing_factor = (
            np.max(fit_params[n])
            / s
            # if n != "n"
            # else 1e-4  # np.max(fit_params[n]) / s / 20
        )
        fit_splines[n] = UnivariateSpline(
            logE_mids,
            fit_params[n],
            k=k,
            s=smoothing_factor,
        )
        smoothed_fit_params[n] = tuple(fit_splines[n](logE_mids))
        if n == "n":
            smoothed_fit_params[n][smoothed_fit_params[n] < 0] = 0
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
        tmp_fit["loc"] = logE_mids[ii]
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
        tmp_fit["loc"] = logE_mids[jj]
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
    parser.add_argument("--kde", action="store_true")
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

    # generate standard energy resolution based on KDEs
    if args.kde:
        ident = "KDE"
        print("running KDE resolution smoothing")
        all_E_grids, logE_bins_old, logE_reco_bins_old = get_baseline_energy_res_kde(
            renew_calc=args.renew_calc
        )
        logE_mids_old = get_mids(logE_bins_old)
        logE_reco_mids_old = get_mids(logE_reco_bins_old)
        pad_logE = np.concatenate(
            [[logE_bins_old[0]], logE_mids_old, [logE_bins_old[-1]]]
        )
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
        with open(join(st.LOCALPATH, f"Eres_mephistograms_{ident}.pckl"), "wb") as f:
            pickle.dump(all_E_histos, f)
        # combine horizontal and upgoing resolutions
        eres_up_mh = all_E_histos["dec-0.0"] + all_E_histos["dec-50.0"]
        eres_up_mh.normalize(axis=1)  # normalize per log(E)
        with open(join(st.LOCALPATH, f"energy_smearing_{ident}_up.pckl"), "wb") as f:
            pickle.dump(eres_up_mh, f)

        # we need the transposed matrix for further calculations
        eres_up_T = eres_up_mh.T()
    else:
        ident = "GP"
        print("running gaussian-process resolution smoothing")
        ## NEW: smooth the resolution functions using gaussian processes (GP)
        GP_mephistograms = get_baseline_eres(renew_calc=args.renew_calc)
        # combine horizontal and upgoing resolutions
        gp_eres = GP_mephistograms["dec-0.0"] + GP_mephistograms["dec-50.0"]
        gp_eres.normalize(axis=1)  # normalize per log(E)

        with open(join(st.LOCALPATH, f"energy_smearing_{ident}_up.pckl"), "wb") as f:
            pickle.dump(gp_eres, f)

        # we need the transposed matrix for further calculations
        eres_up_T = gp_eres.T()

    # Parameterize the smearing matrix
    fit_params = fit_eres_params(eres_up_T)
    np.save(join(st.LOCALPATH, f"Eres_fits_{ident}.npy"), fit_params)
    # smoothed version
    smoothed_fit_params = smooth_eres_fit_params(
        fit_params, eres_up_T.bin_mids[1], s=45, k=3
    )
    np.save(join(st.LOCALPATH, f"Eres_fits_smoothed_{ident}.npy"), smoothed_fit_params)

    # Artificial resolution matrices
    ## Best reproduction based on the fit parameters
    artificial_2D = artificial_eres(fit_params, *eres_up_T.bins)
    artificial_2D.axis_names = eres_up_T.axis_names
    with open(join(st.LOCALPATH, f"artificial_energy_smearing_{ident}_up.pckl"), "wb") as f:
        pickle.dump(artificial_2D.T(), f)

    ## Best reproduction based on the smoothed fit parameters
    artificial_2D = artificial_eres(smoothed_fit_params, *eres_up_T.bins)
    artificial_2D.axis_names = eres_up_T.axis_names
    with open(
        join(st.LOCALPATH, f"artificial_smoothed_energy_smearing_{ident}_up.pckl"), "wb"
    ) as f:
        pickle.dump(artificial_2D.T(), f)

    ## 1:1 reco reproduction
    artificial_one2one = one2one_eres(smoothed_fit_params, *eres_up_T.bins)
    artificial_one2one.axis_names = eres_up_T.axis_names
    with open(
        join(st.LOCALPATH, f"idealized_artificial_energy_smearing_{ident}_up.pckl"), "wb"
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
            f"improved_{impro_factor}_artificial_energy_smearing_{ident}_up.pckl",
        )
        with open(filename, "wb") as f:
            pickle.dump(artificial_2D_impro.T(), f)
