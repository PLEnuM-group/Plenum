import os
import numpy as np
import scipy as sp
import pickle
import matplotlib.colors as mc
from matplotlib.colors import LogNorm
import colorsys
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

matplotlib = sns.mpl
plt = matplotlib.pyplot
sns.set(
    context="notebook",
    style="ticks",
    font="serif",
    font_scale=1.5,
    rc={
        'font.family': 'serif',
        'axes.grid': True,
        'grid.color': "0.9",
        'mathtext.fontset': 'cm',
        'image.cmap': 'mako',
        'savefig.format': 'pdf'
    }
)
sns.set_palette("crest", n_colors=4)

# aeff in m^2
# energy in GeV
livetime = 10 * 3600 * 24 * 365.24 # 10 years in seconds
gamma_astro = -2.37
phi_astro = 1.36 * 10**(-18) # * (E/100 TeV)^gamma / GeV / sr / cm^2 / s
# from phd thesis of Joeran Stettner (IC diffuse benchmark on numu)
# temporary link: https://www.institut3b.physik.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaawyqakk
# will be published as paper soon
# could also switch to his previous work, ICRC2019: https://arxiv.org/abs/1908.09551
# with phi=1.44, gamma=2.28


def reset_palette(n_colors, pal="crest"):
    sns.set_palette(pal, n_colors=n_colors)

def calc_mids(arr):
    return (arr[1:] + arr[:-1]) * 0.5


def slightly_change_color(color, amount=0.2):
    """ slightly change the color hue"""
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    ld = 0.1 if c[1]<=0.5 else -0.1
    return colorsys.hls_to_rgb(c[0]+amount, c[1] + ld, c[2])