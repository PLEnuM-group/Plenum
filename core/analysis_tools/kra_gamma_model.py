# This class loads the data from the kra gamma model
# for galactic neutrino emission

import numpy as np
import tables, math, os, sys, scipy
import healpy as hp

gaggeroFile = r"../resources/gaggero_KRAgamma_hardening_5e7_neutrino_COGalprop_HIGalprop_healpix.V2_hd5"
class GaggeroMap(object):
    def __init__(self, filename):
        hdf = tables.open_file(filename)
        self.flux = hdf.root.flux.col("value")#.reshape((171, 33619968/171))
        self.nside = 128
        self.emin = 10
        self.emax = 1e8
        self.efactor = 1.1
        self.npix = 12*self.nside**2
        hdf.close()
    def GalToHealpy(self, lon, lat):
        theta = -lat + np.pi/2.
        if lon<0:
            phi = lon+2*np.pi
        else:
            phi = lon
        return theta, phi
    def EnergyToBin(self, energy):
        bin = math.log(energy/self.emin)/math.log(self.efactor)
        rest = bin - int(bin)
        return int(bin), rest
    def BinToEnergy(self, bin):
        return self.emin*self.efactor**bin
    def GetFlux(self, energy, lon, lat):
        if energy < self.emin or energy > self.emax:
            return 0.
        theta, phi = self.GalToHealpy(lon, lat)
        pix = hp.ang2pix(self.nside, theta, phi)
        bin, rest = self.EnergyToBin(energy)
        if rest >= 0.:
            nextBin = bin+1
        else:
            nextBin = max(0, bin-1)
        if self.flux[bin*self.npix + pix]==0. and self.flux[nextBin*self.npix + pix]==0.:
            return 0.
        deltaFlux = (math.log(self.flux[bin*self.npix + pix])-math.log(self.flux[nextBin*self.npix + pix]))
        deltaEnergy = (math.log(self.BinToEnergy(bin))-math.log(self.BinToEnergy(nextBin)))
        if deltaEnergy==0.:
            return self.flux[bin*self.npix + pix]
        expo = deltaFlux/deltaEnergy*(math.log(energy)-math.log(self.BinToEnergy(bin)))
        return math.exp(math.log(self.flux[bin*self.npix + pix]) + expo)


