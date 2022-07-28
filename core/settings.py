import numpy as np
import matplotlib.colors as mc

try:
    import colorsys
except:
    print("Could not import colorsys.")
    colorsys = None
import seaborn as sns
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings

warnings.filterwarnings("ignore")

BASEPATH = "/home/lisajsch/repos/Plenum/"

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

# define location of experiments
# some plot settings
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
    "P-ONE": {
        "lon": -123.3656 * u.deg,
        "lat": 48.4284 * u.deg,
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


# aeff in m^2
# energy in GeV
LIVETIME = 10 * 3600 * 24 * 365.24  # 10 years in seconds
E_NORM = 1e5  # normalization energy of power law (E/E_NORM)^gamma
# --> 100 TeV
# from phd thesis of Joeran Stettner (IC diffuse benchmark on numu)
# temporary link: https://www.institut3b.physik.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaawyqakk
# UPDATE: paper now available at https://arxiv.org/abs/2111.10299
GAMMA_ASTRO = 2.37
PHI_ASTRO_FACTOR = 1.44

# previous PUBLISHED work, ICRC2019: https://arxiv.org/abs/1908.09551
# GAMMA_ASTRO = 2.28

# we use this factor such that PHI_ASTRO_FACTOR can be of order 1
PHI_0 = 1.0e-18  # * (E/100 TeV)^gamma / GeV / sr / cm^2 / s
PHI_ASTRO = PHI_ASTRO_FACTOR * PHI_0  # * (E/100 TeV)^gamma / GeV / sr / cm^2 / s

sgr_a = SkyCoord(0, 0, unit="rad", frame="galactic")
txs0506 = SkyCoord(77.36, 5.69, unit="deg", frame="icrs")
ngc1068 = SkyCoord(40.67, -0.01, unit="deg", frame="icrs")


def reset_palette(n_colors, pal="crest"):
    sns.set_palette(pal, n_colors=n_colors)


def slightly_change_color(color, amount=0.2):
    """slightly change the color hue"""
    if not colorsys:
        print("Cannot change color.")
        return color
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    ld = 0.1 if c[1] <= 0.5 else -0.1
    return colorsys.hls_to_rgb(c[0] + amount, c[1] + ld, c[2])


def change_color_ld(color, amount=0.2):
    """slightly change the color lightness/darkness"""
    if not colorsys:
        print("Cannot change color.")
        return color
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    c_new = np.clip(c[1] + amount, 0, 1)
    return colorsys.hls_to_rgb(c[0], c_new, c[2])
