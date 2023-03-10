# This file contains a selection of functions that are used within
# the Plenum notebooks
import numpy as np
from pathlib import Path
from matplotlib.ticker import NullLocator
import matplotlib.colors as mc
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from pandas import read_table
from os.path import join

# get the baseline path of this project
tmp_path = str(Path(__file__).parent.resolve())
BASEPATH = "/".join(tmp_path.split("/")[:-1])

try:
    import healpy as hp
except:
    print("Could not import healpy.")
    print("functions plot_area, add_catalog, and add_extended_plane will not work.")

try:
    import colorsys
except:
    colorsys = None


def read_effective_area():
    column_names = [
        "logE_nu_min",
        "logE_nu_max",
        "Dec_nu_min",
        "Dec_nu_max",
        "A_eff",
    ]

    public_data_aeff = read_table(
        join(BASEPATH, "resources/IC86_II_effectiveArea.csv"),
        delim_whitespace=True,
        skiprows=1,
        names=column_names,
    )
    return public_data_aeff


def read_smearing_matrix():
    """Read the public-data smearing matrix into a data frame."""

    column_names = [
        "logE_nu_min",
        "logE_nu_max",
        "Dec_nu_min",
        "Dec_nu_max",
        "logE_reco_min",
        "logE_reco_max",
        "PSF_min",
        "PSF_max",
        "AngErr_min",
        "AngErr_max",
        "Fractional_Counts",
    ]

    public_data_df = read_table(
        join(BASEPATH, "resources/IC86_II_smearing.csv"),
        delim_whitespace=True,
        skiprows=1,
        names=column_names,
    )
    return public_data_df


def get_scaler(x, thresh, key_x="log10(p)", key_y="scaler"):
    """Powerlaw interpolation wrapper using a pandas.DataFrame input.
    It takes x as the DataFrame, key_x/key_y as x and y coordinates to evaluate the DF.
    Next, log10 of x and y is calculated such that a linear 'ax + b' polynomial fit can be applied.
    The polynomial is then evaluated at -log10(thresh), then translated back to the original form (10**...).

    Originally, this was used to calculate the threshold for discovery by interpolation.
    That's why it looks this complicated.
    """
    return np.power(
        10,
        np.poly1d(np.polyfit(np.log10(x[key_x]), np.log10(x[key_y]), 1))(
            np.log10(-np.log10(thresh))
        ),
    )


def poisson_llh(mu_i, k_i):
    """Calculate the -2 log(Poisson LLH).

    L(data k | model mu)  = prod_{i,j} mu_ij ** k_ij / k_ij! * exp(-mu_ij)

    For numerical stability, we directly evaluate the log of the poisson probability
    (see https://en.wikipedia.org/wiki/Stirling%27s_approximation for approximation of the faculty function).

    Since we are using Asimov data that can have floating point values, we need to implement the function
    instead of using scipy.stats.poisson.logpmf. (It fails for floats in k_i!!)

    -2 log (L) = -2 [k_i log(mu_i) - mu_i - 0.5 log(2 pi k_i) + k_i - k_i log(k_i)]

    We treat some special cases that cause problems in log:

    * mu -> 0, k>0     --> P -> 0
    * k -> 0, mu>0     --> P -> exp(-mu)
    * k -> 0, mu -> 0  --> P -> 1

    """
    log_LLH = np.zeros_like(mu_i)
    # k == 0, mu > 0:
    _mask = (k_i == 0) & (mu_i > 0)
    log_LLH[_mask] = -mu_i[_mask]
    # k == 0, mu == 0:
    _mask = (k_i == 0) & (mu_i == 0)
    log_LLH[_mask] = 0
    # k > 0, mu==0: should not happen! we'll assign a very negative value
    _mask = (k_i > 0) & (mu_i == 0)
    log_LLH[_mask] = -1e16
    # k > 0, mu > 0
    _mask = (k_i > 0) & (mu_i > 0)
    log_LLH[_mask] = (
        k_i[_mask] * np.log(mu_i[_mask])
        - mu_i[_mask]
        - 0.5 * np.log(2 * np.pi * k_i[_mask])
        + k_i[_mask]
        - k_i[_mask] * np.log(k_i[_mask])
    )

    return -2 * np.sum(log_LLH)


def array_source_interp(dec, array, sindec_mids, axis=0):
    """Select a slice of an array with sindec coordinates that matches the chosen dec.

    Parameters:
    -----------
    dec: declination value between -np.pi/2 and +np.pi/2

    array: 2D array where one axis is the sindec axis which we want to slice out

    sindec_mids: sindec coordinates of one axis of the array

    axis: optional, default=0
        Indicates which axis is the sindec axis.
        If not given, it will assume it's axis 0.

    """
    # Find the correct bin of sindec where dec is in
    low_ind = np.digitize(np.sin(dec), sindec_mids)

    # Check which dimension of the array we need to pick out
    # axis 0 is the standard, transpose if it's axis 1
    if axis == 1:
        if isinstance(array, np.ndarray):
            array = array.T
        else:
            array = array.T()

    if low_ind >= len(sindec_mids):
        # print("end of range")
        array_interp = array[:, -1]
    elif low_ind == 0:
        # print("low end range")
        array_interp = array[:, low_ind]
    else:
        # interpolate the array values within the bin boundaries
        array_interp = np.zeros(len(array))
        for i in range(len(array)):
            array_interp[i] = np.interp(
                np.sin(dec),
                [sindec_mids[low_ind - 1], sindec_mids[low_ind]],
                [array[i, low_ind - 1], array[i, low_ind]],
            )
    return array_interp


def sort_contour(xvals, yvals):
    """Take the coordinates of a closed contour and sort them properly for plotting"""
    # center around (0, 0)
    x_cms = np.mean(xvals)
    y_cms = np.mean(yvals)
    # calculate the angle for sorting
    angles = np.arctan2(yvals - y_cms, xvals - x_cms)
    # sorting indices by angle
    indx = np.argsort(angles)
    # get the sorted coordinates
    xsorted = xvals[indx]
    ysorted = yvals[indx]
    # close the contour
    xsorted = np.concatenate([xsorted, [xsorted[0]]])
    ysorted = np.concatenate([ysorted, [ysorted[0]]])
    return xsorted, ysorted


def sigma2pval(sigma, one_sided=True):
    """Translate a sigma value to a p-value, one_sided or two_sided"""
    if one_sided:
        return norm.sf(sigma)
    else:
        return 1.0 - (norm.cdf(sigma) - norm.cdf(-sigma))


def get_mids(bins, ext=False):
    """Calculate the bin mids from an array of bin edges."""
    res = (bins[1:] + bins[:-1]) * 0.5
    if ext == False:
        return res
    else:
        res[0], res[-1] = bins[0], bins[-1]
        return res


# angular distance between two points on a skymap
def ang_dist(src_ra, src_dec, ra, dec):
    """Calculate the angular distance between a source/sources and a set of angular coordinates"""
    # convert src_ra, dec to numpy arrays if not already done
    src_ra = np.atleast_1d(src_ra)[:, np.newaxis]
    src_dec = np.atleast_1d(src_dec)[:, np.newaxis]

    cos_ev = np.sqrt(1.0 - np.sin(dec) ** 2)

    cosDist = np.cos(src_ra - ra) * np.cos(src_dec) * cos_ev + np.sin(src_dec) * np.sin(
        dec
    )

    # handle possible floating precision errors
    cosDist[np.isclose(cosDist, -1.0) & (cosDist < -1)] = -1.0
    cosDist[np.isclose(cosDist, 1.0) & (cosDist > 1)] = 1.0
    dist = np.arccos(cosDist)

    return dist


def interpolate_quantile_value(q, xedges, yvals):
    r"""Interpolate quantile values from a histogram.

    Parameters:
    -----------
    q: quantile, float between 0 and 1
    xedges: bin edges of the histogram with length n+1
    yvals: heights of the bins with length n

    Returns:
    --------
    Quantile value based on interpolated histogram values,
    """
    cumulative_yvals = np.cumsum(yvals).astype(float)
    cumulative_yvals /= cumulative_yvals[-1]

    mids = (xedges[:-1] + xedges[1:]) * 0.5
    mask_s = cumulative_yvals <= q
    mask_l = cumulative_yvals > q
    try:
        x1 = np.atleast_1d(mids[mask_s])[-1]
        x2 = np.atleast_1d(mids[mask_l])[0]
        y1 = np.atleast_1d(cumulative_yvals[mask_s])[-1]
        y2 = np.atleast_1d(cumulative_yvals[mask_l])[0]
    except:
        # this means the quantile is either in the lowest or the highest bin
        if np.count_nonzero(mask_s) == 0:
            print("Quantile is in lowest bin")
            return xedges[0]
        elif np.count_nonzero(mask_l) == 0:
            print("Quantile is in highest bin")
            return xedges[1:][yvals > 0][-1]
        else:
            print("Something weird happened??? Please check")
            return None
    return np.interp(q, [y1, y2], [x1, x2])


def getAngDist(ra1, dec1, ra2, dec2):
    """
    ### From PyAstronomy package ###
    Calculate the angular distance between two coordinates.

    Parameters
    ----------
    ra1 : float, array
        Right ascension of the first object in degrees.
    dec1 : float, array
        Declination of the first object in degrees.
    ra2 : float, array
        Right ascension of the second object in degrees.
    dec2 : float, array
        Declination of the second object in degrees.
    Returns
    -------
    Angle : float, array
        The angular distance in DEGREES between the first
        and second coordinate in the sky.

    """

    delt_lon = (ra1 - ra2) * np.pi / 180.0
    delt_lat = (dec1 - dec2) * np.pi / 180.0
    # Haversine formula
    dist = 2.0 * np.arcsin(
        np.sqrt(
            np.sin(delt_lat / 2.0) ** 2
            + np.cos(dec1 * np.pi / 180.0)
            * np.cos(dec2 * np.pi / 180.0)
            * np.sin(delt_lon / 2.0) ** 2
        )
    )

    return dist / np.pi * 180.0


# convert ra, dec error to ungular uncertainty (this is just an approximation)
def conv2ang_uncertainty(ra_err, dec_err, dec):
    err = np.sqrt(dec_err**2 + (np.cos(np.radians(dec)) * ra_err) ** 2) / np.sqrt(2.0)
    return err


# plot an area on a skymap, vals contain the values of all pixels
def plot_area(
    vals,
    ax,
    npix=768,
    colorbar=dict(cmap="viridis"),
    masked=True,
    galactic=False,
    **kwargs
):
    r"""Plot a 2dim function on the map using pcolormesh

    Parameters
    ----------
    npix: size of the xy grid = npix * npix/2
    colorbar : dict
    Arguments passed to matplotlibs colorbar function.

    """
    # values for each pixel
    vals = np.asarray(vals)

    # create a xy grid of both angles
    x = np.linspace(-np.pi, np.pi, npix)
    y = np.linspace(0, np.pi, npix // 2)
    X, Y = np.meshgrid(x, y)

    YY, XX = Y.ravel(), X.ravel()

    # get the pixel number for each point on the xy grid
    pix = hp.ang2pix(hp.npix2nside(len(vals)), YY, XX)

    # select the respective pixel value for each
    # xy point and reshape to the grid shape
    Z = np.reshape(vals[pix], X.shape)
    kwargs.setdefault("shading", "gouraud")

    # shifr declination to values from -90 to 90 degrees
    lon = np.linspace(-np.pi, np.pi, len(x))
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, len(y))
    if galactic:
        lon = lon[::-1]

    if masked == True:
        Z[Z == 0] = 1.0e-10
    Z_masked = np.ma.masked_where(Z == 0, Z)

    mesh = ax.pcolormesh(lon, lat, Z_masked, **kwargs)

    return mesh


# Fast rotation function
def rotate(ra1, dec1, ra2, dec2, ra3, dec3):
    r"""Rotation matrix for rotation of (ra1, dec1) onto (ra2, dec2).
    The rotation is performed on (ra3, dec3).

    """

    def cross_matrix(x):
        r"""Calculate cross product matrix

        A[ij] = x_i * y_j - y_i * x_j

        """
        skv = np.roll(np.roll(np.diag(x.ravel()), 1, 1), -1, 0)
        return skv - skv.T

    ra1 = np.atleast_1d(ra1)
    dec1 = np.atleast_1d(dec1)
    ra2 = np.atleast_1d(ra2)
    dec2 = np.atleast_1d(dec2)

    ra3 = np.atleast_1d(ra3)
    dec3 = np.atleast_1d(dec3)

    cos_alpha = np.cos(ra2 - ra1) * np.cos(dec1) * np.cos(dec2) + np.sin(dec1) * np.sin(
        dec2
    )

    # correct rounding errors
    cos_alpha[cos_alpha > 1] = 1
    cos_alpha[cos_alpha < -1] = -1

    alpha = np.arccos(cos_alpha)
    vec1 = np.vstack(
        [np.cos(ra1) * np.cos(dec1), np.sin(ra1) * np.cos(dec1), np.sin(dec1)]
    ).T
    vec2 = np.vstack(
        [np.cos(ra2) * np.cos(dec2), np.sin(ra2) * np.cos(dec2), np.sin(dec2)]
    ).T

    vec3 = np.vstack(
        [np.cos(ra3) * np.cos(dec3), np.sin(ra3) * np.cos(dec3), np.sin(dec3)]
    ).T

    nvec = np.cross(vec1, vec2)
    norm = np.sqrt(np.sum(nvec**2, axis=1))
    nvec[norm > 0] /= norm[np.newaxis, norm > 0].T

    one = np.diagflat(np.ones(3))
    nTn = np.array([np.outer(nv, nv) for nv in nvec])
    nx = np.array([cross_matrix(nv) for nv in nvec])

    R = np.array(
        [
            (1.0 - np.cos(a)) * nTn_i + np.cos(a) * one + np.sin(a) * nx_i
            for a, nTn_i, nx_i in zip(alpha, nTn, nx)
        ]
    )

    vec = np.array([np.dot(R[0], vec_i.T) for vec_i in vec3])

    ra = np.arctan2(vec[:, 1], vec[:, 0])
    dec = np.arcsin(vec[:, 2])

    ra += np.where(ra < 0.0, 2.0 * np.pi, 0.0)

    return ra, dec


def set_ticks(ax, fs=20, galactic=False):
    xnames = (r"24h", r"0h")
    if galactic:
        xnames = (r"180$^{\circ}$", r"-180$^{\circ}$")

    xticks = np.linspace(-180.0, 180.0, 7)

    if hasattr(ax, "set_longitude_grid_ends"):
        ax.set_longitude_grid_ends(85.0)
        # create labels and grid
        ax.xaxis.set_major_locator(NullLocator())
        xmargin = np.pi / 80.0
        ax.text(
            -np.pi - xmargin,
            0.0,
            r" {0:s}".format(xnames[0]),
            size=fs,  # "large",
            weight="semibold",
            horizontalalignment="right",
            verticalalignment="center",
        )
        ax.text(
            np.pi + xmargin,
            0.0,
            r" {0:s}".format(xnames[1]),
            size=fs,  # "large",
            weight="semibold",
            horizontalalignment="left",
            verticalalignment="center",
        )

        yticks = ax.yaxis.get_major_ticks()
        yticks[5].label1.set_visible(False)

    ax.yaxis.set_tick_params(labelsize=fs)
    return


# function to add event with angular uncertainty to a skymap
def add_event(ax, ra_i, dec_i, sigma_i, coords="ra", **kwargs):

    if abs(dec_i) > np.radians(70):
        ext = 20.0
    elif abs(dec_i) > np.radians(30):
        ext = 10.0
    else:
        ext = 3.0

    val = ext * np.degrees(sigma_i)

    n = 200
    # get all points on the map for which the angular distance is euqal the 1 sigma level
    ra_bins = np.linspace(np.degrees(ra_i) - val, np.degrees(ra_i) + val, num=n)
    dec_bins = np.linspace(np.degrees(dec_i) - val, np.degrees(dec_i) + val, num=n)
    xx, yy = np.meshgrid(ra_bins, dec_bins, indexing="ij")

    DIST = getAngDist(xx, yy, np.rad2deg(ra_i), np.rad2deg(dec_i)) - np.rad2deg(sigma_i)

    c = kwargs.pop("color", "black")
    d = np.zeros_like(DIST)
    d[DIST == 0.0] = 1

    if coords == "ra":
        xx, yy = _trans(np.radians(xx), np.radians(yy))
        ra_i, dec_i = _trans(ra_i, dec_i)

    res = ax.contour(xx, yy, DIST, levels=[0.0], colors=c, alpha=0.8, **kwargs)
    return res


def add_catalog(ax, cat, n, key="sign_avg", tck=None, vals=None, **kwargs):
    m = cat[key] >= 0
    mb2 = np.abs(cat["Glat"]) > 10.0
    cat = cat[m & mb2]
    ind = np.argsort(cat[key])[-n:]

    _ra = np.radians(cat[ind]["ra"])
    _dec = np.radians(cat[ind]["dec"])

    _ra, _dec = _trans(_ra, _dec)

    if vals is not None:
        ind = hp.ang2pix(hp.npix2nside(len(vals)), _dec + np.pi / 2.0, _ra)
        alphas = vals[ind]
        label = kwargs.pop("label", None)
        n = 0
        for i, deci in enumerate(_dec):

            alpha = alphas[i] / np.max(alphas)
            kwargs["alpha"] = alpha
            if alphas[i] == np.max(alphas) and n == 0:
                ax.scatter(_ra[i], deci, label=label, **kwargs)
                n += 1
            else:
                ax.scatter(_ra[i], deci, **kwargs)
        return

    ax.scatter(_ra, _dec, **kwargs)
    return


def _trans(ra, dec):
    r"""Transform ra and dec such that they can be plotted on a hammer skymap
    0h right side, 24h left side
    """
    try:
        x, y = np.pi * u.rad - np.atleast_1d(ra), np.atleast_1d(dec)
    except:
        x, y = np.pi - np.atleast_1d(ra), np.atleast_1d(dec)

    return x, y


def add_plane(ax, coords="ra", color="black", label="Galactic center/plane", **kwargs):
    in_deg = True if "transform" in kwargs else False

    c = SkyCoord(frame="galactic", l=0.0, b=0.0, unit="deg")
    gc = SkyCoord(l=0 * u.degree, b=0 * u.degree, frame="galactic")
    if coords == "ra":
        cra, cdec = _trans(gc.fk5.ra, gc.fk5.dec)
    else:
        cra, cdec = gc.fk5.ra, gc.fk5.dec

    if in_deg:
        cra, cdec = cra.deg, cdec.deg
    else:
        cra, cdec = cra.rad, cdec.rad

    ax.plot(
        cra,
        cdec,
        marker="o",
        ms=15,
        c=color,
        linestyle="dotted",
        label=label,
        alpha=0.8,
        **kwargs
    )

    num2 = 150
    gc = SkyCoord(
        l=np.linspace(-np.pi, np.pi, num2) * u.rad, b=0 * u.rad, frame="galactic"
    )
    if coords == "ra":
        cra, cdec = _trans(gc.icrs.ra, gc.icrs.dec)
    else:
        cra, cdec = gc.icrs.ra, gc.icrs.dec
    ind = np.argsort(cra)
    cra, cdec = cra[ind], cdec[ind]
    cra = cra.wrap_at("180d")
    if in_deg:
        cra, cdec = cra.deg, cdec.deg
    else:
        cra, cdec = cra.rad, cdec.rad
    ax.plot(
        cra, cdec, marker="None", c=color, linestyle="dotted", linewidth=3, **kwargs
    )
    return


def add_obj(ax, name, coords="ra", marker="o", c="red", **kwargs):

    ras = {
        "txs": np.radians(77.36),
        "ngc": np.radians(40.67),
    }
    decs = {
        "txs": np.radians(5.69),
        "ngc": np.radians(-0.01),
    }
    labels = {"txs": "TXS 0506+056", "ngc": "NGC 1068"}

    if coords == "ra":
        _ra, _dec = _trans(ras[name], decs[name])
    else:
        _ra, _dec = ras[name], decs[name]
    if "transform" in kwargs:
        _ra, _dec = np.rad2deg(_ra), np.rad2deg(_dec)

    ax.plot(
        _ra,
        _dec,
        marker=marker,
        ms=15,
        c=c,
        linestyle="None",
        label=kwargs.pop("label", labels[name]),
        **kwargs
    )

    return


def add_extended_plane(ax, color="black", ngrid=500, **kwargs):
    NSIDE = 2**6
    npix = hp.nside2npix(NSIDE)

    # add also the lowest and upper line
    evals = dict()
    for blat in [-10.0, 10]:
        num2 = 500
        gp_ra = np.zeros(num2)
        gp_dec = np.zeros(num2)
        for i, x_i in enumerate(np.linspace(0, 359, num2)):
            gc = SkyCoord(l=x_i * u.degree, b=blat * u.degree, frame="galactic")
            _gp = gc.fk5
            gp_ra[i] = _gp.ra.rad
            gp_dec[i] = _gp.dec.rad

        cra, cdec = _trans(gp_ra, gp_dec)

        _args = np.argsort(cra)
        spl = UnivariateSpline(cra[_args], cdec[_args], s=1e-4)
        evals[blat] = spl

    fit_vals = np.linspace(-np.pi, np.pi, 100)
    ax.fill_between(
        fit_vals, evals[-10.0](fit_vals), evals[10.0](fit_vals), color="red", alpha=0.5
    )

    return True


def shade_sky(ax, sky="all", **kwargs):
    fit_vals = np.linspace(-np.pi, np.pi, 500)
    if sky == "all":
        ymin = -np.pi / 2 * np.ones_like(fit_vals)
        ymax = np.pi / 2 * np.ones_like(fit_vals)
    elif sky == "north":
        ymin = np.radians(-5.0) * np.ones_like(fit_vals)
        ymax = np.pi / 2 * np.ones_like(fit_vals)
    elif sky == "south":
        ymin = -np.pi / 2 * np.ones_like(fit_vals)
        ymax = np.radians(-5.0) * np.ones_like(fit_vals)

    ax.fill_between(fit_vals, ymin, ymax, **kwargs)

    return True


# color helper functions
def reset_palette(n_colors, sns, pal="crest"):
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
