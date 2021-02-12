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
    return colorsys.hls_to_rgb(c[0]+amount, c[1], c[2])