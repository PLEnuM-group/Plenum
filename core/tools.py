# This file contains a selection of functions that are used within 
# the Plenum notebooks
import numpy as np
import healpy as hp
from PyAstronomy import pyasl
from matplotlib.ticker import (AutoMinorLocator, FixedLocator, FuncFormatter,
                               MultipleLocator, NullLocator, LogLocator)

from astropy import units as u
from astropy.coordinates import SkyCoord

from scipy.stats import norm 
def sigma2pval(sigma, one_sided=True):
    if one_sided:
        return norm.sf(sigma)
    else: 
        return 1.-(norm.cdf(sigma)-norm.cdf(-sigma))


def get_mids(bins, ext=False):
    res = (bins[1:]+bins[:-1])/2.
    if ext==False:
        return res
    else:
        res[0], res[-1] = bins[0], bins[-1]
        return res

# angular distance between two points on a skymap
def ang_dist(src_ra, src_dec, ra, dec):
    # convert src_ra, dec to numpy arrays if not already done
    src_ra = np.atleast_1d(src_ra)[:, np.newaxis]
    src_dec = np.atleast_1d(src_dec)[:, np.newaxis]

    cos_ev = np.sqrt(1. - np.sin(dec)**2)

    cosDist = (
        np.cos(src_ra - ra) * np.cos(src_dec) * cos_ev +
        np.sin(src_dec) * np.sin(dec)
        )

    # handle possible floating precision errors
    cosDist[np.isclose(cosDist, -1.) & (cosDist < -1)] = -1.
    cosDist[np.isclose(cosDist, 1.) & (cosDist > 1)] = 1.
    dist = np.arccos(cosDist)
    
    return dist





#convert ra, dec error to ungular uncertainty (this is just an approximation)
def conv2ang_uncertainty(ra_err, dec_err, dec):
    err = np.sqrt(dec_err**2 + (np.cos(np.radians(dec)) * ra_err)**2) / np.sqrt(2.)
    return err


# plot an area on a skymap, vals contain the values of all pixels
def plot_area(vals, ax, npix=768, colorbar=dict(cmap='viridis'), masked=True, 
        galactic=False,**kwargs):
    r""" Plot a 2dim function on the map using pcolormesh

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
    y = np.linspace( 0, np.pi, npix // 2)
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
    lat = np.linspace(-np.pi/2., np.pi/2., len(y))
    if galactic:
        lon = lon[::-1]


    if masked == True:
        Z[Z==0] = 1.e-10
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

    cos_alpha = np.cos(ra2 - ra1) * np.cos(dec1) * np.cos(dec2) \
                  + np.sin(dec1) * np.sin(dec2)

    # correct rounding errors
    cos_alpha[cos_alpha >  1] =  1
    cos_alpha[cos_alpha < -1] = -1

    alpha = np.arccos(cos_alpha)
    vec1 = np.vstack([np.cos(ra1) * np.cos(dec1),
                      np.sin(ra1) * np.cos(dec1),
                      np.sin(dec1)]).T
    vec2 = np.vstack([np.cos(ra2) * np.cos(dec2),
                      np.sin(ra2) * np.cos(dec2),
                      np.sin(dec2)]).T

    vec3 = np.vstack([np.cos(ra3) * np.cos(dec3),
                      np.sin(ra3) * np.cos(dec3),
                      np.sin(dec3)]).T

    nvec = np.cross(vec1, vec2)
    norm = np.sqrt(np.sum(nvec**2, axis=1))
    nvec[norm > 0] /= norm[np.newaxis, norm > 0].T

    one = np.diagflat(np.ones(3))
    nTn = np.array([np.outer(nv, nv) for nv in nvec])
    nx = np.array([cross_matrix(nv) for nv in nvec])

    R = np.array([(1.-np.cos(a)) * nTn_i + np.cos(a) * one + np.sin(a) * nx_i
                  for a, nTn_i, nx_i in zip(alpha, nTn, nx)])

    vec = np.array([np.dot(R[0], vec_i.T) for vec_i in vec3])    

    ra = np.arctan2(vec[:, 1], vec[:, 0])
    dec = np.arcsin(vec[:, 2])

    ra += np.where(ra < 0., 2. * np.pi, 0.)

    return ra, dec


def set_ticks(ax, fs=20, galactic=False):
    xnames=(r"24h", r"0h")
    if galactic:
        xnames = (r'180$^{\circ}$', r'-180$^{\circ}$')

    xticks=np.linspace(-180., 180., 7)

    if hasattr(ax, "set_longitude_grid_ends"):
        ax.set_longitude_grid_ends(85.)
        # create labels and grid
        ax.xaxis.set_major_locator(NullLocator())
        xmargin = np.pi/80.
        ax.text(-np.pi - xmargin, 0.,
                         r" {0:s}".format(xnames[0]), size=fs,#"large",
                         weight="semibold", horizontalalignment="right",
                         verticalalignment="center")
        ax.text(np.pi + xmargin, 0.,
                         r" {0:s}".format(xnames[1]), size=fs,#"large",
                         weight="semibold",
                         horizontalalignment="left",
                         verticalalignment="center")

        yticks = ax.yaxis.get_major_ticks()
        yticks[5].label1.set_visible(False)

    ax.yaxis.set_tick_params(labelsize=fs)
    return


# function to add event with angular uncertainty to a skymap
def add_event(ax, ra_i, dec_i, sigma_i, coords='ra', **kwargs):

    if abs(dec_i) > np.radians(70): ext = 20.
    elif abs(dec_i) > np.radians(30): ext = 10.
    else: ext = 3.

    val = ext * np.degrees(sigma_i)

    n = 200
    #get all points on the map for which the angular distance is euqal the 1 sigma level
    ra_bins = np.linspace(np.degrees(ra_i)-val, np.degrees(ra_i)+val, num=n)
    dec_bins = np.linspace(np.degrees(dec_i)-val, np.degrees(dec_i)+val, num=n)
    xx, yy =np.meshgrid(ra_bins, dec_bins, indexing='ij')

    DIST = pyasl.getAngDist(xx,yy,
                                np.degrees(ra_i),np.degrees(dec_i)) - np.degrees(sigma_i)

    c = kwargs.pop('color', 'black')
    d = np.zeros_like(DIST)
    d[DIST==0.] = 1

    if coords=='ra':
        xx,yy=  _trans(np.radians(xx), np.radians(yy))
        ra_i, dec_i = _trans(ra_i,dec_i)

    res = ax.contour(xx, yy, DIST, levels=[ 0.], colors = c,
                       alpha=0.8, **kwargs)
    return res


def add_catalog(ax, cat, n, key='sign_avg', tck=None, vals=None, 
                **kwargs):
    m = cat[key] >= 0
    mb2 = np.abs(cat['Glat']) > 10.
    cat =cat[m&mb2]
    ind = np.argsort(cat[key])[-n:]

    _ra = np.radians(cat[ind]['ra'])
    _dec = np.radians(cat[ind]['dec'])

    _ra, _dec = _trans(_ra, _dec)

    if vals is not None:
        ind = hp.ang2pix(hp.npix2nside(len(vals)),  _dec+np.pi/2., _ra)
        alphas = vals[ind]
        label = kwargs.pop('label', None)
        n=0
        for i, deci in enumerate(_dec):

            alpha = alphas[i] / np.max(alphas)
            kwargs['alpha'] =  alpha
            if alphas[i] == np.max(alphas) and n==0:
                ax.scatter(_ra[i], deci, label=label,**kwargs)
                n+=1
            else:
                ax.scatter(_ra[i], deci, **kwargs)
        return

    ax.scatter(_ra, _dec, **kwargs)
    return

def _trans(ra, dec):
    r''' Transform ra and dec such that they can be plotted on a hammer skymap
    0h right side, 24h left side
    '''
    x, y = np.pi-np.atleast_1d(ra), np.atleast_1d(dec)

    return x, y


def add_plane(ax, coords='ra', color='black', label='Galactic center/plane'):
    c = SkyCoord(frame="galactic", l=0., b=0., unit='deg')
    gc = SkyCoord(l=0*u.degree, b=0*u.degree, frame='galactic')
    _gc= gc.fk5
    gc_ra = _gc.ra.rad
    gc_dec = _gc.dec.rad

    if coords=='ra':
        cra ,cdec = _trans(gc_ra, gc_dec)
    else:
        cra ,cdec = gc_ra, gc_dec

    ax.plot(cra, cdec, marker='o',ms=15, c=color, linestyle='dotted', label=label, 
           alpha=0.8)

    num2 = 50
    gp_ra = np.zeros(num2)
    gp_dec = np.zeros(num2)
    for i,x_i in enumerate(np.linspace(0,360,num2)):
        gc = SkyCoord(l=x_i*u.degree, b=0*u.degree, frame='galactic')
        _gp= gc.fk5
        gp_ra[i] = _gp.ra.rad
        gp_dec[i] = _gp.dec.rad


    ind = np.argmin(gp_dec[gp_ra>np.pi])
    ind = np.where(gp_dec == gp_dec[gp_ra>0][ind])[0][0]
    cra ,cdec = _trans(gp_ra, gp_dec)
    ax.plot(np.append(cra[ind:],cra[:ind]), np.append(cdec[ind:],cdec[:ind]), marker='None', c=color, 
       linestyle='dotted', linewidth=3)

    return True


def add_txs(ax, coords='ra', marker='o', c='red', label='TXS 0506+056'):
    txs_ra = np.radians(77.36)
    txs_dec = np.radians(5.69)

    if coords=='ra':
        txs_ra, txs_dec = _trans(txs_ra, txs_dec)
    else:
        txs_ra, txs_dec = txs_ra, txs_dec
    ax.plot(txs_ra, txs_dec, marker=marker,ms=15, c=c,
       linestyle='None', label=label)

    return


def add_extended_plane(ax, color='black', ngrid=500, **kwargs):
    NSIDE = 2**6
    npix = hp.nside2npix(NSIDE)

    #add also the lowest and upper line
    evals = dict()
    for blat in [-10., 10]:
        num2 = 500
        gp_ra = np.zeros(num2)
        gp_dec = np.zeros(num2)
        for i,x_i in enumerate(np.linspace(0,359,num2)):
            gc = SkyCoord(l=x_i*u.degree, b=blat*u.degree, frame='galactic')
            _gp= gc.fk5
            gp_ra[i] = _gp.ra.rad
            gp_dec[i] = _gp.dec.rad


        cra ,cdec = _trans(gp_ra, gp_dec) 

        _args = np.argsort(cra)  
        spl = UnivariateSpline(cra[_args], cdec[_args], s=1e-4)
        evals[blat] = spl

    fit_vals = np.linspace(-np.pi, np.pi, 100)
    ax.fill_between(fit_vals, evals[-10.](fit_vals), evals[10.](fit_vals), color='red', 
                    alpha=0.5) 

    return True


def shade_sky(ax, sky='all', **kwargs):    
    fit_vals = np.linspace(-np.pi, np.pi, 500)
    if sky == 'all':
        ymin = -np.pi / 2*np.ones_like(fit_vals)
        ymax = np.pi / 2*np.ones_like(fit_vals)
    elif sky == 'north':
        ymin = np.radians(-5.)*np.ones_like(fit_vals)
        ymax = np.pi / 2*np.ones_like(fit_vals)
    elif sky == 'south':
        ymin = -np.pi / 2*np.ones_like(fit_vals)
        ymax = np.radians(-5.)*np.ones_like(fit_vals)

    ax.fill_between(fit_vals, ymin, ymax, **kwargs) 

    return True

