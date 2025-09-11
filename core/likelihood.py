# This file contains a selection of functions that are used within
# the Plenum notebooks
import numpy as np
from tools import array_source_interp
from pathlib import Path
from fluxes import astro_flux, atmo_background, flux_collection
from aeff_calculations import calc_aeff_factor

# get the baseline path of this project
tmp_path = str(Path(__file__).parent.resolve())
BASEPATH = "/".join(tmp_path.split("/")[:-1])


def poisson_llh(mu_i, k_i, summed=True):
    """
    Calculate the -2 log(Poisson Log-Likelihood) for given model parameters.

    Parameters:
    - mu_i (numpy.ndarray): Array of model predictions for each data point.
    - k_i (numpy.ndarray): Array of observed counts for each data point.

    Returns:
    float: The -2 log(Poisson Log-Likelihood) value.

    The Poisson Log-Likelihood is calculated using the formula:

    L(data k | model mu) = prod_{i,j} mu_ij ** k_ij / k_ij! * exp(-mu_ij)

    For numerical stability, the log of the Poisson probability is directly evaluated,
    utilizing Stirling's approximation for the factorial function.

    -2 log(L) = -2 [k_i log(mu_i) - mu_i - 0.5 log(2 pi k_i) + k_i - k_i log(k_i)]

    Special cases are handled to avoid numerical issues:
    - mu -> 0, k > 0      --> P -> 0
    - k -> 0, mu > 0      --> P -> exp(-mu)
    - k -> 0, mu -> 0     --> P -> 1

    Note:
    Since Asimov data can have floating-point values, this function is implemented
    instead of using scipy.stats.poisson.logpmf, which fails for floats in k_i.

    """
    log_LLH = np.zeros_like(mu_i)
    # k == 0, mu > 0:
    _mask = (k_i == 0) & (mu_i > 0)
    log_LLH[_mask] = -mu_i[_mask]
    # k == 0, mu == 0:
    _mask = (k_i == 0) & (mu_i == 0)
    log_LLH[_mask] = 0
    # k > 0, mu==0: should not happen! we'll raise an error
    _mask = (k_i > 0) & (mu_i == 0)
    if np.count_nonzero(_mask) > 0:
        raise ValueError("Invalid case of k > 0, mu==0")
    # k > 0, mu > 0
    _mask = (k_i > 0) & (mu_i > 0)
    log_LLH[_mask] = (
        k_i[_mask] * np.log(mu_i[_mask])
        - mu_i[_mask]
        - 0.5 * np.log(2 * np.pi * k_i[_mask])
        + k_i[_mask]
        - k_i[_mask] * np.log(k_i[_mask])
    )

    if summed:
        return -2 * np.sum(log_LLH)
    else:
        return -2 * log_LLH


# Set up LLH function
# $ \mathcal{L}({\rm data}~k~ |~{\rm hypothesis}~\mu)
#     = \prod_{{\rm bin\,}ij}^{N_{\rm bins}} \frac{\mu_{ij}^{k_{ij}}}{k_{ij}!}\cdot
#     \exp \left( -\mu_{ij} \right)$


# Background hypothesis $H_0(\mu = N_B)$ : only atmospheric neutrino flux

# Signal hypothesis $H_1(\mu = \{N_B, N_S, \gamma\})$: atmospheric neutrino flux + astrophysical neutrino flux

# Idea: data ($k$) are the perfect representation of our expectation; the hypothesis ($\mu$) is the model with the free parameters we'd like to know


def ps_llh_single(
    x,
    aeff_factor_s,
    aeff_factor_b,
    bckg_flux,
    k_i,
    energy_resolution,
    e_0,
    phi_0,
    shape,
    verbose=False,
    flux_shape=None,
    signal_parameters=None,  # only needed if multiple signal models are mixed
    fixed_BG=False,
    summed=True,
):
    ## with the new mixture implementation this becomes a bit unwieldy - update TODO
    """
    Calculate the log-likelihood using Poisson statistics for a single dataset assuming consistent properties.

    Parameters:
        x (list): Fit parameters.
            x[0]: Background normalization scaling.
            x[1]: Signal normalization scaling.
            x[2:]: Other signal parameters. See 'astro_flux' for further shapes and parameters.
        aeff_factor_s (float): Effective area factor for the signal.
        aeff_factor_b (float): Effective area factor for the background.
        bckg_flux (list): Background flux values.
        k_i (array-like): Observation/Asimov data.
        energy_resolution (float): Energy resolution.
        e_0 (float): Normalization energy - if multiple models are given, all will have the same e_0
        phi_0 (float): Normalization flux - if multiple models are given, all will have the same phi_0
        shape (str): Flux shape.
        verbose (bool, optional): Whether to print additional information. Default is False.
        flux_shape (named_tuple, optional): Use a splined model flux instead of an analytic flux descpription.
            phi_0 and e_0 are not used then. see fluxes.py for definition; model_spline should be formatted as flux = 10 ** model_spline(log10_E).
            (see fluxes.astro_flux)
        signal_parameters: list of number of parameters per signal model
        fixed_BG (bool): if False, calculate background from aeff_factor_b; if True,
            assume aeff_factor_b is already the full background.

    Returns:
        float: -2 * Log-likelihood value calculated using Poisson statistics. See 'poisson_llh'.

    Note:
        This function assumes that there is only one dataset with consistent properties.
    """
    # Calculate the background contribution
    if fixed_BG:
        mu_b = aeff_factor_b * x[0]
    else:
        mu_b = (
            atmo_background(
                aeff_factor=aeff_factor_b,
                bckg_vals=bckg_flux,
                energy_resolution=energy_resolution,
            )
            * x[0]
        )
    # Calculate the signal contribution
    # check if it's a multi-model mixture:
    if type(shape) == list:
        all_fluxes = []
        first_signal_index = 1
        for ii, shape_i in enumerate(shape):
            # here we generate a flux tuple with the current parameters
            if "model_flux" in shape_i:
                flux_i = flux_collection["model_flux"](
                    x[first_signal_index], flux_shape.model_spline, shape_i
                )
            else:
                flux_i = flux_collection[shape_i](
                    phi_0 * x[first_signal_index],
                    *x[first_signal_index + 1 :],
                    e_0,
                    shape_i
                )
                # else: flux shape is already fixed as model flux with just
                # the normalization (x[1]) as free parameter
            first_signal_index += signal_parameters[ii]
            all_fluxes.append(flux_i)

        mu_s = astro_flux(
            aeff_factor=aeff_factor_s,
            emids=10 ** aeff_factor_s.bin_mids[1],
            energy_resolution=energy_resolution,
            phi_scaling=1,  # normalization factor -> now in flux shape normalization to allow for relative scaling
            flux_shape=all_fluxes,
        )
    # else, just do the regular calculation
    else:
        if not "model_flux" in shape:
            # here we generate a flux tuple with the current parameters
            flux_shape = flux_collection[shape](phi_0, *x[2:], e_0, shape)
            # else: flux shape is already fixed as model flux with just
            # the normalization (x[1]) as free parameter

        mu_s = astro_flux(
            aeff_factor=aeff_factor_s,
            emids=10 ** aeff_factor_s.bin_mids[1],
            energy_resolution=energy_resolution,
            phi_scaling=x[1],  # normalization factor
            flux_shape=flux_shape,
        )
    if verbose:
        # Print additional information if verbose mode is enabled
        print(x[0], x[1], *x[2:])
        print(flux_collection[shape](phi_0, *x[2:], e_0, shape))
        print(np.sum(mu_b), np.sum(mu_s))

    # Calculate the total expected events for the Poisson LLH
    mu_i = mu_s + mu_b

    # Calculate -2 * log-likelihood using Poisson statistics
    return poisson_llh(mu_i, k_i, summed=summed)


def ps_llh_multi(
    x,
    all_aeff_factor_s,
    all_aeff_factor_b,
    all_bckg_flux,
    all_k,
    all_eres,
    shape,
    e_0,
    phi_0,
    flux_shape=None,
    signal_parameters=None,
    summed=True,
):
    """
    Calculate the total log-likelihood across multiple datasets with different properties.

    Parameters:
        x (list): Fit parameters.
        all_aeff_factor_s (list): List of effective area factors for the signal for each dataset.
        all_aeff_factor_b (list): List of effective area factors for the background for each dataset.
        all_bckg_flux (list): List of background flux values for each dataset.
        all_k (list): List of observation/Asimov data for each dataset.
        all_eres (list): List of energy resolutions for each dataset.
        shape (str): Flux shape.
        e_0 (float): Normalization energy.
        phi_0 (float): Normalization flux.

    Returns:
        float: Total log-likelihood value across all datasets.

    Note:
        This function assumes that there are multiple datasets with different properties.
    """
    llh = 0
    for i, aeffs in enumerate(all_aeff_factor_s):
        llh += ps_llh_single(
            x=x,
            aeff_factor_s=aeffs,
            aeff_factor_b=all_aeff_factor_b[i],
            bckg_flux=all_bckg_flux[i],
            k_i=all_k[i],
            energy_resolution=all_eres[i],
            e_0=e_0,
            phi_0=phi_0,
            shape=shape,
            flux_shape=flux_shape,
            signal_parameters=signal_parameters,
            summed=summed,
        )
    return llh


def setup_multi_llh(
    eres,
    conf,
    aeff_2d,
    bckg_histo,
    bg_config,
    sig_config,
    src_flux,
    verbose=False,
    return_s_b=False,
):
    """
    Set up the components required for calculating the log-likelihood across multiple datasets.

    Parameters:
        eres (dict or array): Dictionary of arrays/mephistograms or single array/mephistogram representing the
                                  energy resolutions for each dataset. If a dictionary is provided,
                                  it should map dataset identifiers to their corresponding energy resolution.
                                  If a float is provided, the same energy resolution will be used for all datasets.
        conf (tuple): Tuple containing two lists - the first list represents dataset identifiers,
                      and the second list contains corresponding scaling factors corresponding to
                      either a lifetime scaling or effective-area scaling.
        verbose (bool, optional): Whether to print additional information. Default is False.
        return_s_b (bool, optional): Whether to return the individual signal and background histograms. Default is False.

    Returns:
        tuple: A tuple containing the following components for each dataset:
            - all_aeff_factor_s (list): List of effective area factors for the signal.
            - all_aeff_factor_b (list): List of effective area factors for the background.
            - all_k (list): List of observation/Asimov data.
            - all_bckg_flux (list): List of background flux values.
            - if return_s_b: all_k_s, all_k_b (lists): List of signal and background histograms

    Note:
        This function assumes that the configuration parameters and required functions (e.g., `calc_aeff_factor`,
        `atmo_background`, `astro_flux`, `array_source_interp`) are defined and accessible in the global namespace.
        If `eres` is a dictionary, each dataset identifier should have an associated energy resolution value.
        If `eres` is an  array/mephistogram, the same energy resolution will be used for all datasets.
    """

    all_aeff_factor_s = []
    all_aeff_factor_b = []
    all_k = []
    all_bckg_flux = []
    all_eres = []
    if return_s_b:
        all_k_s = []
        all_k_b = []

    for ident, factor in zip(*conf):
        # Calculate effective area factors for background and signal
        aeff_factor_bckg = calc_aeff_factor(aeff_2d[ident], **bg_config) * factor
        aeff_factor_signal = calc_aeff_factor(aeff_2d[ident], **sig_config) * factor

        # Calculate background flux
        bckg_flux = array_source_interp(
            bg_config["dec"],
            bckg_histo[ident],
            bckg_histo[ident].bin_mids[0],
            axis=1,
        )
        # Determine the energy resolution for the current dataset
        current_eres = eres[ident] if isinstance(eres, dict) else eres

        # Calculate asimov data for atmospheric background
        k_b = atmo_background(
            aeff_factor=aeff_factor_bckg,
            bckg_vals=bckg_flux,
            energy_resolution=current_eres,
        )

        # Calculate asimov data for astrophysical signal with power law spectrum
        k_s = astro_flux(
            aeff_factor=aeff_factor_signal,
            emids=10 ** aeff_factor_signal.bin_mids[1],
            energy_resolution=current_eres,
            phi_scaling=1,
            flux_shape=src_flux,
        )

        if verbose:
            print("Asimov data sum:")
            print("Background:", np.sum(k_b))
            print("Signal:", np.sum(k_s))

        all_aeff_factor_s.append(aeff_factor_signal)
        all_aeff_factor_b.append(aeff_factor_bckg)
        all_k.append(k_s + k_b)
        all_bckg_flux.append(bckg_flux)
        all_eres.append(current_eres)
        if return_s_b:
            all_k_s.append(k_s)
            all_k_b.append(k_b)
    if return_s_b:
        return (
            all_aeff_factor_s,
            all_aeff_factor_b,
            all_k,
            all_bckg_flux,
            all_eres,
            all_k_b,
            all_k_s,
        )
    else:
        return all_aeff_factor_s, all_aeff_factor_b, all_k, all_bckg_flux, all_eres
