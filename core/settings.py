import numpy as np
from os.path import join, exists
from os import makedirs
from pathlib import Path

import seaborn as sns
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
from tools import get_mids
from scipy.stats import norm

import mkl

mkl.set_num_threads(1)

# get the baseline path of this project
tmp_path = str(Path(__file__).parent.resolve())
BASEPATH = "/".join(tmp_path.split("/")[:-1])
# local path contains resources that are not mirrored on git
LOCALPATH = join(BASEPATH, "local")
if not exists(LOCALPATH):
    makedirs(LOCALPATH)
    print("Created local directory", LOCALPATH)

interpolation_method = "linear"  # for all interpolators

E_MIN = 2  # 100 GeV
E_MAX = 9  # 10⁹ GeV = 1 EeV
LIVETIME_DAYS = 3186  # new ngc paper
LIVETIME = LIVETIME_DAYS * 24 * 3600
LIVETIME_86 = 2198.2 * 24 * 3600  # only IC86-II and later in 10yr-PS paper
# LIVETIME = 10 * 360 * 24 * 3600  # 360 days of data taking per year in seconds
E_NORM = 1e5  # normalization energy of power law (E/E_NORM)^gamma
# --> 100 TeV
# Diffuse nu-mu paper now available at https://arxiv.org/abs/2111.10299
GAMMA_ASTRO = 2.37
PHI_ASTRO_FACTOR = 1.44
# we use this factor such that PHI_ASTRO_FACTOR can be of order 1
PHI_0 = 1.0e-18  # * (E/100 TeV)^gamma / GeV / sr / cm^2 / s
PHI_ASTRO = PHI_ASTRO_FACTOR * PHI_0  # * (E/100 TeV)^gamma / GeV / sr / cm^2 / s

# Science Paper parameters of NGC 1068
GAMMA_NGC = 3.2
PHI_NGC = 5e-14  # @ 1 TeV / GeV cm² s
E0_NGC = 1e3
# cutoff parameters
Gamma_cut = 2.0
logE_cut = 3.4  # log10 (Ecut / GeV)

# baseline binning
step = 0.05
logE_bins = np.arange(2, 9.05, step=step)
logE_reco_bins = np.arange(2, 9.0, step=step)
sindec_bins = np.linspace(-1, 1, num=101)
ra_bins = np.linspace(0, np.pi * 2, num=100)

delta_psi_max = 3  # 3 for soft spectra, 1 for hard spectra
bins_per_psi2 = 25
psi2_bins = np.linspace(
    0, delta_psi_max**2, num=int(delta_psi_max**2 * bins_per_psi2) + 1
)
psi2_mids = get_mids(psi2_bins)

# inferred binning/bin mids
logE_mids = get_mids(logE_bins)
ebins = np.power(10, logE_bins)
# emids = np.power(10, logE_mids)
emids = get_mids(ebins)
ewidth = np.diff(ebins)

logE_reco_mids = get_mids(logE_reco_bins)

sindec_mids = get_mids(sindec_bins)
sindec_width = np.diff(sindec_bins)

ra_mids = get_mids(ra_bins)
ra_width = np.diff(ra_bins)

# gaussian sigma values, one sided
sigma5 = (1 - norm.cdf(5)) / 2
sigma3 = (1 - norm.cdf(3)) / 2
sigma2 = (1 - norm.cdf(2)) / 2
sigma1 = (1 - norm.cdf(1)) / 2

# important object coordinates
sgr_a = SkyCoord(0, 0, unit="rad", frame="galactic")
txs0506 = SkyCoord(77.36, 5.69, unit="deg", frame="icrs")
ngc1068 = SkyCoord(40.67, -0.01, unit="deg", frame="icrs")


# plot settings
matplotlib = sns.mpl
plt = matplotlib.pyplot
plt.style.use(join(BASEPATH, "style.mplstyle"))
colorlist = plt.rcParams["axes.prop_cycle"].by_key()["color"]
warnings.filterwarnings("ignore")

# default cool colors
linestyles = ["-", "--", "-.", ":"]
colors = [
    (0.4287136896080445, 0.8230641296498253, 0.7976237339879146),
    (0.17544617305706944, 0.6466723383995561, 0.6162725242360733),
    (0.024792120980593635, 0.4191425610223744, 0.39370216536046365),
    (0.713458847396123, 0.261074771588123, 0.29025904200122044),
]
many_colors = ["0.7", "0.4", "0."]
many_colors.extend(
    sns.cubehelix_palette(start=0.9, rot=0, n_colors=3, light=0.7, dark=0.3, hue=1.5)
)
many_colors.extend(
    sns.cubehelix_palette(start=2.4, rot=0, n_colors=3, light=0.7, dark=0.3, hue=1.5)
)
many_colors = np.array(many_colors, dtype=object)
ecap_cmap = sns.cubehelix_palette(
    start=(149 - 256) / 256 * 3 + 1,
    rot=0,
    light=1,
    dark=0.2,
    hue=1,
    as_cmap=True,
    reverse=True,
)
from matplotlib.colors import ListedColormap

ecap_listed_cmap = ListedColormap(
    sns.cubehelix_palette(
        start=(149 - 256) / 256 * 3 + 1, rot=0, light=1, dark=0.1, hue=1.2, n_colors=12
    )
)
# define location of experiments
# and plot settings
GEN2_FACTOR = 7.5
detector_configurations = {
    "IceCube": (["IceCube"], [1]),
    "IceCube-Gen2": (["IceCube"], [GEN2_FACTOR]),
    "IceCube+Gen2": (["IceCube", "IceCube"], [1, GEN2_FACTOR]),
    "P-ONE": (["IceCube", "P-ONE"], [1, 1]),
    "P-ONE-only": (["P-ONE"], [1]),
    "KM3NeT": (["IceCube", "KM3NeT"], [1, 1]),
    "IC+P1+K3": (["IceCube", "P-ONE", "KM3NeT"], [1, 1, 1]),
    "Plenum-1": (["IceCube", "P-ONE", "KM3NeT", "Baikal-GVD"], [2, 1, 1, 1]),
    "Plenum-2": (
        ["IceCube", "P-ONE", "KM3NeT", "Baikal-GVD"],
        [1 + GEN2_FACTOR, 1, 1, 1],
    ),
}

poles = {
    "IceCube": {
        "lon": 1 * u.deg,
        "lat": -90 * u.deg,
        "color": "k",
        "ls": "-",
        "label": "IceCube",
        "marker": "d",
    },
    "Gen-2": {
        "lon": 1 * u.deg,
        "lat": -90 * u.deg,
        "color": "gray",
        "ls": "-.",
        "label": "IceCube x 7.5",
        "marker": "d",
    },
    "P-ONE": {  # cascadia basin node location
        "lon": -127.729272 * u.deg,
        "lat": 47.742345 * u.deg,
        "color": colors[2],
        "ls": "--",
        "label": "at P-ONE location",
        "marker": "v",
    },
    "KM3NeT": {
        "lon": (16 + 6 / 60) * u.deg,
        "lat": (36 + 16 / 60) * u.deg,
        "color": colors[0],
        "ls": "--",
        "label": "at KM3NeT location",
        "marker": "d",
    },
    "Baikal-GVD": {
        "lon": 108.1650 * u.deg,
        "lat": 53.5587 * u.deg,
        "color": colors[1],
        "ls": "--",
        "label": "at GVD location",
        "marker": "d",
    },
    "Plenum-1": {"color": "#339999", "ls": "-", "label": r"PLE$\nu$M-1", "marker": "s"},
    "Plenum-2": {
        "color": "#a50000",
        "ls": "-.",
        "label": r"PLE$\nu$M-2",
        "marker": "o",
    },
}
