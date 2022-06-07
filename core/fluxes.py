import pickle
from settings import *
from aeff_calculations import energy_smearing

# energy smearing matrix
with open("../resources/energy_smearing_kde.pckl", "rb") as f:
    _, kvals, logE_reco_bins = pickle.load(f)
# normalize per bin in true energy
normed_kvals = kvals / np.sum(kvals, axis=0)

# atmospheric backgound smearing
def atmo_background(aeff_factor, spl_vals, normed_kvals=None):
    if normed_kvals is not None:
        return energy_smearing(normed_kvals, aeff_factor * spl_vals)
    else:
        return aeff_factor * spl_vals


## Adapted from Mauricio's fluxes
# These are the basic shapes
def gaussian(x, mu, sigma):
    """Not normalized, then parameters are easier to interpret"""
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2))  # /(np.sqrt(2*np.pi) * sigma)


def power_law(energy, e_scale, gamma, phi0):
    """ Generic form of power-law spectrum: (energy / e_scale) ** (-gamma) * phi0
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
    shape,
    aeff_factor,
    emids,
    enorm,
    *args,
    phi_0=PHI_0,
    normed_kvals=None,
):
    """
    Wrapper for different astro flux shapes to put into TS minimizer.
    Possible shapes and their parameters:

    ° powerlaw:
        args[0]: gamma
        args[1]: phi scaling (phi normalization will be phi_0 * phi scaling)

    ° powerlaw * cutoff:
        args[0]: gamma
        args[1]: phi scaling
        args[2]: cutoff energy

    ° log-parabola:
        args[0]: alpha parameter
        args[1]: phi scaling
        args[2]: beta parameter

    ° double powerlaw:
        args[0]: gamma_1 (E < E_Break)
        args[1]: phi scaling
        args[2]: gamma_2 (E >= E_Break)
        args[3]: E_Break

    ° powerlaw * dip/bump:
        args[0]: gamma
        args[1]: phi scaling
        args[2]: amplitude --- sign of amplitude is defined by 'dip' or 'bump'
        args[3]: mean energy
        args[4]: width

    ° powerlaw * sigmoid:
        args[0]: gamma
        args[1]: phi scaling
        args[2]: fraction of depletion
        args[3]: growth rate
        args[4]: transition energy
    """
    if "powerlaw" in shape:
        _gamma_astro = args[0]
        _phi_astro_scaling = args[1]
        tmp = aeff_factor * power_law(
            emids, enorm, _gamma_astro, phi_0 * _phi_astro_scaling
        )

    if "double" in shape:
        _gamma_2 = args[2]
        _E_break = np.power(10, args[3])
        phi_2 = (
            phi_0
            * _phi_astro_scaling
            * (_E_break / enorm) ** (-_gamma_astro + _gamma_2)
        )
        tmp_2 = aeff_factor * power_law(emids, enorm, _gamma_2, phi_2)
        ### merge the two powerlaw shapes
        if type(tmp) == np.ndarray or type(tmp) == list:
            tmp[:, emids >= _E_break] = tmp_2[:, emids >= _E_break]
        elif type(tmp) == float:
            if emids >= _E_break:
                tmp = tmp_2
        else:
            raise ValueError(f"??? invalid type of tmp array ({type(tmp)})")

    if "cutoff" in shape:
        _energy_cut = np.power(10, args[2])
        tmp *= cut_off(emids, _energy_cut)

    if "bump" in shape or "dip" in shape:
        amp = np.power(10, args[2])
        energy_mean = np.power(10, args[3])
        sigma = np.power(10, args[4])
        amp = amp if "bump" in shape else -1 * amp
        tmp *= 1.0 + amp * gaussian(emids, energy_mean, sigma)

    if "sigmoid" in shape:
        fraction_depletion = np.power(10, args[2])
        growth_rate = np.power(10, args[3])
        energy_nu_trans = np.power(10, args[4])
        tmp *= sigmoid(fraction_depletion, growth_rate, emids, energy_nu_trans)

    if "parabola" in shape:
        _alpha_astro = args[0]
        _phi_astro_scaling = args[1]
        _beta_astro = args[2]
        index = parabola_index(_alpha_astro, _beta_astro, emids, enorm)
        tmp = aeff_factor * power_law(emids, enorm, index, phi_0 * _phi_astro_scaling)
    ## energy smearing
    if normed_kvals is not None:
        tmp = energy_smearing(normed_kvals, tmp)
    return tmp


### some generic shape parameters used in the diffuse-style analysis
shape_params = {
    # parameters from https://arxiv.org/abs/1908.09551
    "powerlaw": {
        "baseline": np.array([1, GAMMA_ASTRO, PHI_ASTRO_FACTOR]),
        "bounds": np.array([(0.9, 1.1), (1.0, 4.0), (0.2, 3.0)]),
        "names": np.array(["conv_scaling", "gamma_astro", "Phi_0"]),
        "fancy_names": np.array(
            [
                "Conv. scaling",
                r"$\gamma_{\rm astro}$",
                r"$\Phi_0 /({\rm 10^{-18} GeV cm^2 s sr})$",
            ]
        ),
    },
    # parameters inspired by https://www.institut3b.physik.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaawyqakk
    "powerlaw with cutoff": {
        "baseline": np.array([1, 2.0, 1.5, 6]),
        "bounds": np.array([(0.9, 1.1), (1.0, 4.0), (0.2, 3.0), (4.5, 8.5)]),
        "names": np.array(["conv_scaling", "gamma_astro", "Phi_0", "Cut_off"]),
        "fancy_names": np.array(
            [
                "Conv. scaling",
                r"$\gamma_{\rm astro}$",
                r"$\Phi_0 /({\rm 10^{-18} GeV cm^2 s sr})$",
                r"$\log_{10}$(Cut-off energy/GeV)",
            ]
        ),
    },
    # parameters inspired by https://www.institut3b.physik.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaawyqakk
    "log-parabola": {
        "baseline": np.array([1, 2, 1.7, 0.5]),
        "bounds": np.array([(0.9, 1.1), (1.0, 4.0), (0.2, 3.0), (0.1, 1.9)]),
        "names": np.array(["conv_scaling", "alpha", "Phi_0", r"beta"]),
        "fancy_names": np.array(
            [
                "Conv. scaling",
                r"$\alpha$",
                r"$\Phi_0 /({\rm 10^{-18} GeV cm^2 s sr})$",
                r"$\beta$",
            ]
        ),
    },
    # parameters inspired by https://www.institut3b.physik.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaawyqakk
    "double powerlaw": {
        "baseline": np.array([1, 2.0, 1.2, 3.5, np.log10(1e6)]),
        "bounds": np.array([(0.9, 1.1), (1.0, 4.0), (0.2, 3.0), (1.5, 4.0), (5, 7)]),
        "names": np.array(
            ["conv_scaling", "gamma_astro", "Phi_0", "gamma_2", "E_break"]
        ),
        "fancy_names": np.array(
            [
                "Conv. scaling",
                r"$\gamma_1$",
                r"$\Phi_0 /({\rm 10^{-18} GeV cm^2 s sr})$" r"$\gamma_2$",
                r"$E_{\rm break}$",
            ]
        ),
    },
    "powerlaw with sigmoid": {
        "baseline": np.array(
            [
                1,
                GAMMA_ASTRO,
                PHI_ASTRO_FACTOR,
                np.log10(0.1),
                np.log10(1e-5),
                np.log10(4e5),
            ]
        ),
        "bounds": np.array(
            [(0.9, 1.1), (1.0, 4.0), (0.2, 3.0), (-2.0, 0), (-6, -4), (4.5, 6.5)]
        ),
        "names": np.array(
            [
                "conv_scaling",
                "gamma_astro",
                "Phi_0",
                "depletion",
                "growth",
                "transition",
            ]
        ),
        "fancy_names": np.array(
            [
                "Conv. scaling",
                r"$\gamma_{\rm astro}$",
                r"$\Phi_0 /({\rm 10^{-18} GeV cm^2 s sr})$",
                "Depletion fraction",
                "Growth rate",
                r"$\log_{10}$(Transition energy/GeV)",
            ]
        ),
    },
    "powerlaw with dip": {
        "baseline": np.array(
            [
                1,
                GAMMA_ASTRO,
                PHI_ASTRO_FACTOR,
                np.log10(0.7),
                np.log10(6e5),
                np.log10(6e5 / 3),
            ]
        ),
        "bounds": np.array(
            [(0.9, 1.1), (1.0, 4.0), (0.2, 3.0), (-2.0, 1), (4.0, 7.0), (3.5, 7.0)]
        ),
        "names": np.array(
            [
                "conv_scaling",
                "gamma_astro",
                "Phi_0",
                "amplitude",
                "mean_energy",
                "width",
            ]
        ),
        "fancy_names": np.array(
            [
                "Conv. scaling",
                r"$\gamma_{\rm astro}$",
                r"$\Phi_0 /({\rm 10^{-18} GeV cm^2 s sr})$",
                "Amplitude",
                r"$\log_{10}$(Mean energy/GeV)",
                r"$\log_{10}$(Width/GeV)",
            ]
        ),
    },
    "powerlaw with bump": {
        "baseline": np.array(
            [
                1,
                GAMMA_ASTRO,
                PHI_ASTRO_FACTOR,
                np.log10(2.2),
                np.log10(4.5e5),
                np.log10(4.5e5 / 3),
            ]
        ),
        "bounds": np.array(
            [(0.9, 1.1), (1.0, 4.0), (0.2, 3.0), (-2.0, 1), (4.0, 7.0), (3.5, 7.0)]
        ),
        "names": np.array(
            [
                "conv_scaling",
                "gamma_astro",
                "Phi_0",
                "amplitude",
                "mean_energy",
                "width",
            ]
        ),
        "fancy_names": np.array(
            [
                "Conv. scaling",
                r"$\gamma_{\rm astro}$",
                r"$\Phi_0 /({\rm 10^{-18} GeV cm^2 s sr})$",
                "Amplitude",
                r"$\log_{10}$(Mean energy/GeV)",
                r"$\log_{10}$(Width/GeV)",
            ]
        ),
    },
}

# some randomization of guessing parameters
rs = np.random.RandomState(seed=667)
for shape in shape_params:
    shape_params[shape]["guess"] = np.copy(
        shape_params[shape]["baseline"]
    ) * rs.uniform(0.98, 1.02, size=len(shape_params[shape]["baseline"]))
# alternative: give it the truth to supress minimizer errors
# for shape in shape_params:
#    shape_params[shape]["guess"] = deepcopy(shape_params[shape]["baseline"])


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
