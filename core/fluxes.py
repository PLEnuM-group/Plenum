from settings import *
from resolution import energy_smearing
from collections import namedtuple
from pandas import read_csv, read_table
from scipy.interpolate import InterpolatedUnivariateSpline

# we base the flux models on named-tuples
PL_flux = namedtuple("PL_flux", "norm gamma E0 shape")
PLcut_flux = namedtuple("PLcut_flux", "norm gamma e_cut E0 shape")
LogP_flux = namedtuple("LogP_flux", "norm alpha beta E0 shape")
model_flux = namedtuple("model_flux", "norm model_spline shape")
DPL_flux = namedtuple("DPL_flux", "norm gamma_1 gamma_2 E_break E0 shape")
Sig_flux = namedtuple("Sig_flux", "norm gamma depletion growth transition E0 shape")
DipBump_flux = namedtuple(
    "DipBump_flux", "norm gamma amplitude mean_energy width E0 shape"
)
# use the right keywords here for the shape
# such that def astro_flux can identify them
flux_collection = {
    "powerlaw": PL_flux,
    "powerlaw with cutoff": PLcut_flux,
    "log-parabola": LogP_flux,
    "model_flux": model_flux,
    "double powerlaw": DPL_flux,
    "sigmoid": Sig_flux,
    "bump": DipBump_flux,
}


# atmospheric backgound smearing
def atmo_background(aeff_factor, bckg_vals, energy_resolution=None):
    """Calculate the number of neutrinos of atmospheric background flux
    as a function neutrino energy or reconstructed energy.

    If aeff_factor is 2D, eg. in sin(dec) and log(E_true),
    the result will also be 2D, in sin(dec) and log(E_reco) then.

    Parameters:
    aeff_factor: array
        effective area multiplied with binning and livetime
    bckg_vals: atmospheric background flux binned the same way as aeff_factor
    energy_resolution: Optional
        If the energy resolution matrix is given,
        calculate the background events for reconstructed energy;
        If None, return background events for true neutrino energy

    """
    if energy_resolution is not None:
        return energy_smearing(energy_resolution, aeff_factor * bckg_vals)
    else:
        return aeff_factor * bckg_vals


## Adapted from Mauricio's fluxes
# These are the basic shapes
def gaussian(x, mu, sigma):
    """Not normalized, then parameters are easier to interpret"""
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2))  # /(np.sqrt(2*np.pi) * sigma)


def power_law(energy, e_scale, gamma, phi0):
    """Generic form of power-law spectrum: (energy / e_scale) ** (-gamma) * phi0
    energy: array
        energy values to evaluate the power-law spectrum
    e_scale: float
        normalization of the energy scale
    gamma: float (positive)
        spectral index, will be multiplied by -1
    phi0: float
        flux normalization at the given energy scale

    """
    return (energy / e_scale) ** (-gamma) * phi0


def cut_off(energy, e_cut):
    return np.exp(-energy / e_cut)


def parabola_index(alpha, beta, energy, enorm):
    return alpha + beta * np.log10(energy / enorm)


def sigmoid(fraction_depletion, growth_rate, energy, energy_nu_trans):
    factor = 1 - fraction_depletion
    factor /= 1 + np.exp(-growth_rate * (energy - energy_nu_trans))
    factor += fraction_depletion
    return factor


# combine basic shapes to actual fluxes
def astro_flux(
    aeff_factor,
    emids,
    energy_resolution,
    phi_scaling,
    flux_shape,
):
    """
    Wrapper for different astro flux shapes to put into TS minimizer.
    flux_shape must be a named tuple

    Possible shapes and their parameters:

    ° powerlaw:
        - norm
        - gamma

    ° powerlaw * cutoff:
        - norm
        - gamma
        - e_cut

    ° log-parabola:
        - norm
        - alpha
        - beta

    ° double powerlaw:
        - norm
        - gamma_1 (E < E_Break)
        - gamma_2 (E >= E_Break)
        - e_break

    ° powerlaw * dip/bump:
        - norm
        - gamma
        - amplitude --- sign of amplitude is defined by 'dip' or 'bump'
        - e_mean
        - width

    ° powerlaw * sigmoid:
        - norm
        - gamma
        - depletion
        - growth_rate
        - e_trans

    ° model flux
        - norm
        - model_spline (formatted as flux = 10 ** model_spline(log10_E))
    """
    flux_base = 1
    if "model_flux" in flux_shape.shape:
        flux_base *= (
            flux_shape.norm
            * phi_scaling
            * aeff_factor
            * 10 ** flux_shape.model_spline(np.log10(emids))
        )

    if "powerlaw" in flux_shape.shape:
        _gamma_astro = flux_shape.gamma
        flux_base *= aeff_factor * power_law(
            emids, flux_shape.E0, _gamma_astro, flux_shape.norm * phi_scaling
        )

    if "double" in flux_shape.shape:
        _gamma_2 = flux_shape.gamma_2
        _E_break = np.power(10, flux_shape.e_break)
        phi_2 = (
            flux_shape.norm
            * phi_scaling
            * (_E_break / flux_shape.E0) ** (-_gamma_astro + _gamma_2)
        )
        flux_base_2 = aeff_factor * power_law(emids, flux_shape.E0, _gamma_2, phi_2)
        ### merge the two powerlaw shapes
        if type(flux_base) == np.ndarray or type(flux_base) == list:
            flux_base[:, emids >= _E_break] = flux_base_2[:, emids >= _E_break]
        elif type(flux_base) == float:
            if emids >= _E_break:
                flux_base = flux_base_2
        else:
            raise ValueError(f"??? invalid type of flux_base array ({type(flux_base)})")

    if "cutoff" in flux_shape.shape:
        _energy_cut = np.power(10, flux_shape.e_cut)
        flux_base *= cut_off(emids, _energy_cut)

    if "bump" in flux_shape.shape or "dip" in flux_shape.shape:
        amp = np.power(10, flux_shape.amplitude)
        energy_mean = np.power(10, flux_shape.e_mean)
        sigma = np.power(10, flux_shape.sigma)
        amp = amp if "bump" in flux_shape.shape else -1 * amp
        flux_base *= 1.0 + amp * gaussian(emids, energy_mean, sigma)

    if "sigmoid" in flux_shape.shape:
        fraction_depletion = np.power(10, flux_shape.depletion)
        growth_rate = np.power(10, flux_shape.growth_rate)
        energy_nu_trans = np.power(10, flux_shape.e_trans)
        flux_base *= sigmoid(fraction_depletion, growth_rate, emids, energy_nu_trans)

    if "parabola" in flux_shape.shape:
        _alpha_astro = flux_shape.alpha
        _beta_astro = flux_shape.beta
        index = parabola_index(_alpha_astro, _beta_astro, emids, flux_shape.E0)
        flux_base *= aeff_factor * power_law(
            emids, flux_shape.E0, index, flux_shape.norm * phi_scaling
        )
    ## energy smearing
    if energy_resolution is not None:
        return energy_smearing(energy_resolution, flux_base)
    else:
        return flux_base


# model fluxes
# Model fit Inoue et al 2023 ICRC
# https://pos.sissa.it/444/1161/pdf
inoue_data = read_csv(
    "/home/hpc/capn/capn102h/repos/Plenum/local/neutrino_models/inoue_icrc2023.txt",
    skipinitialspace=True,
)
inoue_data["E_GeV"] = 10 ** (inoue_data["logE_eV"] - 9)
inoue_data["flux"] = (
    (10 ** inoue_data["E2_flux_erg"])
    * 1e3  # it should be erg_to_GeV =~ 624, but it seems like it's just 1E3 in the plot ... (?!)
    / (inoue_data["E_GeV"] ** 2)
)

inoue_src_flux = model_flux(
    1,
    InterpolatedUnivariateSpline(
        np.log10(inoue_data["E_GeV"]), np.log10(inoue_data["flux"]), k=3
    ),
    "model_flux",
)
# see https://doi.org/10.3847/1538-4357/ac1c77
disk_corona_flux = read_table(
    "/home/hpc/capn/capn102h/repos/Plenum/local/ngc_1068_flux_template_Lx_43.txt",
    sep="\t",
    skiprows=1,
    names=["energy", "raw_flux"],
)
disk_corona_flux["flux"] = (
    disk_corona_flux["raw_flux"] / 13**2 * 1e9 / 3
)  # scale from 1 Mpc to 13 Mpc distance; per eV -> per GeV; all-flavor to single flavor

disk_corona_flux["energy_GeV"] = disk_corona_flux["energy"] / 1e9
disk_corona_flux["E2_flux"] = (
    disk_corona_flux["flux"] * disk_corona_flux["energy_GeV"] ** 2
)
mask = disk_corona_flux["flux"] > 0
kheirandish_src_flux = model_flux(
    1,
    InterpolatedUnivariateSpline(
        np.log10(disk_corona_flux["energy_GeV"][mask]),
        np.log10(disk_corona_flux["flux"][mask]),
        k=3,
    ),
    "model_flux",
)

# parameters from https://arxiv.org/abs/1908.09551
baseline_pl_flux = PL_flux(PHI_ASTRO, GAMMA_ASTRO, E_NORM, "powerlaw")
bounds_pl_flux = PL_flux((0.1, 10.0), (1.0, 4.0), E_NORM, "powerlaw")
fancy_pl_flux = PL_flux(
    r"$\Phi_0 /({\rm 10^{-18} GeV cm^2 s sr})$",
    r"$\gamma_{\rm astro}$",
    r"$E_0$",
    "powerlaw",
)

# parameters inspired by
# https://www.institut3b.physik.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaawyqakk
baseline_cut_flux = PLcut_flux(PHI_0 * 1.5, 2.0, 6, E_NORM, "powerlaw with cutoff")
bounds_cut_flux = PLcut_flux(
    (0.1, 10.0), (1.0, 4.0), (4.5, 8.5), E_NORM, "powerlaw with cutoff"
)
fancy_cut_flux = PLcut_flux(
    r"$\Phi_0 /({\rm 10^{-18} GeV cm^2 s sr})$",
    r"$\gamma_{\rm astro}$",
    r"$\log_{10}$(Cut-off energy/GeV)",
    r"$E_0$",
    "powerlaw with cutoff",
)

baseline_para_flux = LogP_flux(PHI_0 * 1.7, 2.0, 0.5, E_NORM, "log-parabola")
bounds_para_flux = LogP_flux(
    (0.1, 10.0), (1.0, 4.0), (0.1, 1.9), E_NORM, "log-parabola"
)
fancy_para_flux = LogP_flux(
    r"$\Phi_0 /({\rm 10^{-18} GeV cm^2 s sr})$",
    r"$\alpha$",
    r"$\beta$",
    r"$E_0$",
    "log-parabola",
)
baseline_double_flux = DPL_flux(
    PHI_0 * 1.2, 2.0, 3.5, np.log10(1e6), E_NORM, "double-powerlaw"
)
bounds_double_flux = DPL_flux(
    (0.1, 10.0), (1.0, 4.0), (1.5, 4.0), (5, 7), E_NORM, "double-powerlaw"
)
fancy_double_flux = DPL_flux(
    r"$\Phi_0 /({\rm 10^{-18} GeV cm^2 s sr})$",
    r"$\gamma_1$",
    r"$\gamma_2$",
    r"$E_{\rm break}$",
    r"$E_0$",
    "double-powerlaw",
)

# just made up some parameters
baseline_sig_flux = Sig_flux(
    PHI_ASTRO,
    GAMMA_ASTRO,
    np.log10(0.1),
    np.log10(1e-5),
    np.log10(4e5),
    E_NORM,
    "powerlaw with sigmoid",
)
bounds_sig_flux = Sig_flux(
    (0.1, 10.0),
    (1.0, 4.0),
    (-2.0, 0),
    (-6, -4),
    (4.5, 6.5),
    E_NORM,
    "powerlaw with sigmoid",
)
fancy_sig_flux = Sig_flux(
    r"$\Phi_0 /({\rm 10^{-18} GeV cm^2 s sr})$",
    r"$\gamma_{\rm astro}$",
    "Depletion fraction",
    "Growth rate",
    r"$\log_{10}$(Transition energy/GeV)",
    r"$E_0$",
    "powerlaw with sigmoid",
)
# Dip
baseline_dip_flux = DipBump_flux(
    PHI_ASTRO,
    GAMMA_ASTRO,
    np.log10(0.7),
    np.log10(6e5),
    np.log10(6e5 / 3),
    E_NORM,
    "powerlaw with dip",
)
bounds_dip_flux = DipBump_flux(
    (0.1, 10.0),
    (1.0, 4.0),
    (-2.0, 1),
    (4.0, 7.0),
    (3.5, 7.0),
    E_NORM,
    "powerlaw with dip",
)
fancy_dip_flux = DipBump_flux(
    r"$\Phi_0 /({\rm 10^{-18} GeV cm^2 s sr})$",
    r"$\gamma_{\rm astro}$",
    "Amplitude",
    r"$\log_{10}$(Mean energy/GeV)",
    r"$\log_{10}$(Width/GeV)",
    r"$E_0$",
    "powerlaw with dip",
)
# Bump
baseline_bump_flux = DipBump_flux(
    PHI_ASTRO,
    GAMMA_ASTRO,
    np.log10(2.2),
    np.log10(4.5e5),
    np.log10(4.5e5 / 3),
    E_NORM,
    "powerlaw with bump",
)
bounds_bump_flux = DipBump_flux(
    (0.1, 10.0),
    (1.0, 4.0),
    (-2.0, 1),
    (4.0, 7.0),
    (3.5, 7.0),
    E_NORM,
    "powerlaw with bump",
)
fancy_bump_flux = DipBump_flux(
    r"$\Phi_0 /({\rm 10^{-18} GeV cm^2 s sr})$",
    r"$\gamma_{\rm astro}$",
    "Amplitude",
    r"$\log_{10}$(Mean energy/GeV)",
    r"$\log_{10}$(Width/GeV)",
    r"$E_0$",
    "powerlaw with bump",
)


def plot_spectrum(energy, events, labels, title, f, ax, **kwargs):
    """Make a step-like plot of energy spectrum"""
    ls = kwargs.pop("ls", ["-"] * len(events))
    color = kwargs.pop("color", [None] * len(events))
    ylim = kwargs.pop("ylim", (0.1, 3e4))
    xlim = kwargs.pop("xlim", (1.8, 9))
    ylabel = kwargs.pop("ylabel", r"# events")
    xlabel = kwargs.pop("xlabel", r"$E_{\mu \, \rm reco}$/GeV")

    for i, (ev, lab) in enumerate(zip(events, labels)):
        ax.plot(
            energy,
            ev,
            drawstyle="steps-mid",
            label=lab,
            ls=ls[i],
            color=color[i],
        )
    ax.legend()
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    f.tight_layout()
    return f, ax
