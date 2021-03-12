# This files contains some functions that can generate signal expectations
# for different scenarios

import abc
import numpy as np
from core.analysis_tools.kra_gamma_model import GaggeroMap,gaggeroFile
from astropy.coordinates import SkyCoord
from scipy.special import expn

class SignalSpectrumKRAgamma(object):
    r'''
    '''
    def __init__(self, kra_gamma_model=GaggeroMap, kra_gamma_file=gaggeroFile):
        r'''
        '''
        self.model = kra_gamma_model(kra_gamma_file)


    def _generate_KRAgamma_skymap(self, ra_mids, sindec_mids, etrue_mids):
        r'''
        '''
        n_etrue = len(etrue_mids)
        xx,yy = np.meshgrid(ra_mids, np.arcsin(sindec_mids), indexing='ij')
        cords = SkyCoord(xx.flatten(), yy.flatten(),  unit='rad')

        glon = cords.galactic.l.rad
        glat = cords.galactic.b.rad

        skymap = np.zeros((len(xx.flatten()),n_etrue),dtype=float)
        for i, (xi,yi) in enumerate(zip(glon,glat)):
            skymap[i,:] = [self.model.GetFlux(energy=ek, lon=xi, lat=yi) 
                 for k,ek in enumerate(etrue_mids)]

        skymap = skymap.reshape((xx.shape+(n_etrue,)))
        return skymap


    def _generate_KRAgamma_skymap_integrated(self, ra_mids, sindec_mids, etrue_mids, 
            etrue_widths):
        r'''
        '''
        skymap = self._generate_KRAgamma_skymap(ra_mids, sindec_mids, etrue_mids)
        return np.sum(skymap * etrue_widths, axis=-1)



class PowerLaw(object, metaclass=abc.ABCMeta):
    r''' Abstract base class for power law functions
    '''
    def __init__(self):
        r'''
        '''
        # Call the super function to allow for multiple class inheritance.
        super(PowerLaw, self).__init__()


    def flux(self):
        r'''
        '''
        pass
    
    def integrated_flux(self):
        r'''
        '''
        pass

    



class SinglePowerLaw(PowerLaw, metaclass=abc.ABCMeta):
    r'''
    '''
    def __init__(self, phi0_bins, gamma_bins, E0=1e5,
            n_fitbins_phi0=40, n_fitbins_gamma=30):
        r'''
        '''
        super(SinglePowerLaw, self).__init__()

        self._E0 = E0
        self.params_names = ['phi0', 'gamma']
        self.params = [phi0_bins,
                gamma_bins]
        self.fit_params = [10**np.linspace(np.log10(phi0_bins[0]),
            np.log10(phi0_bins[-1]),n_fitbins_phi0),
            np.linspace(gamma_bins[0], gamma_bins[-1],
                n_fitbins_gamma)]

        self.params_shape = phi0_bins.shape \
                + gamma_bins.shape

    def flux(self, phi0, gamma, energy):
        r'''
        '''
        return phi0 * (energy / self._E0)**(-gamma)


    def integrated_flux(self, phi0, gamma, emin, emax):
        r'''
        '''
        if gamma!=1:
            res = phi0 * self._E0**(gamma) / (1-gamma) \
                    * (emax**(1-gamma) - emin**(1-gamma))
        else:
            res  = phi0*self._E0**(gamma) * np.log(emax/emin)
        return res


class TwoComponentPowerLaw(PowerLaw, metaclass=abc.ABCMeta):
    r'''
    '''
    def __init__(self, phi0_bins, gamma0_bins, gamma1_bins, E_threshold,
            E0=1e5, n_fitbins_phi0=40, n_fitbins_gamma0=30, n_fitbins_gamma1=30):
        r'''
        '''
        super(TwoComponentPowerLaw, self).__init__()

        self._E0 = E0
        self._E_th = E_threshold
        self.params_names = ['phi0', 'gamma0', 'gamma1']
        self.params = [phi0_bins,
                gamma0_bins, gamma1_bins]
        self.fit_params = [10**np.linspace(np.log10(phi0_bins[0]),
            np.log10(phi0_bins[-1]),n_fitbins_phi0),
            np.linspace(gamma0_bins[0], gamma0_bins[-1],
                n_fitbins_gamma0),
            np.linspace(gamma1_bins[0], gamma1_bins[-1],
                n_fitbins_gamma1)]

        self.params_shape = phi0_bins.shape \
                + gamma0_bins.shape\
                + gamma1_bins.shape

    def flux(self, phi0, gamma0, gamma1, energy):
        r'''
        '''
        return phi0 * ((energy/self._E0)**(-gamma0) \
                + (self._E_th/self._E0)**(gamma1 -gamma0) \
                * (energy/self._E0)**(-gamma1))


    def integrated_flux(self, phi0, gamma0, gamma1, emin, emax):
        r'''
        '''
        def _int_single_powerlaw(_gamma, _emin, _emax):
            if _gamma!=1:
                res = 1. / (1-_gamma) \
                    * (_emax**(1-_gamma) - _emin**(1-_gamma))
            else:
                res  = np.log(_emax/_emin)
            return res

        p0 = self._E0**gamma0 * _int_single_powerlaw(gamma0, emin, emax)
        p1 = (self._E_th/self._E0)**(gamma1-gamma0) \
                * self._E0**gamma1 * _int_single_powerlaw(gamma1, 
                        emin, emax)

        return phi0*(p0+p1)


class CutOffPowerLaw(PowerLaw, metaclass=abc.ABCMeta):
    r'''
    '''
    def __init__(self, phi0_bins, gamma_bins, cutoff_energy_bins, 
            E0=1e5, n_fitbins_phi0=40, n_fitbins_gamma=30, n_fitbins_cutoff=30):
        r'''
        '''
        super(CutOffPowerLaw, self).__init__()

        self._E0 = E0
        self.params_names = ['phi0', 'gamma', 'e_cutoff']
        self.params = [phi0_bins,
                gamma_bins, cutoff_energy_bins]
        self.fit_params = [10**np.linspace(np.log10(phi0_bins[0]),
            np.log10(phi0_bins[-1]),n_fitbins_phi0),
            np.linspace(gamma_bins[0], gamma_bins[-1],
                n_fitbins_gamma),
            10**np.linspace(np.log10(cutoff_energy_bins[0]),
            np.log10(cutoff_energy_bins[-1]),n_fitbins_cutoff)]

        self.params_shape = phi0_bins.shape \
                + gamma_bins.shape\
                + cutoff_energy_bins.shape

    def flux(self, phi0, gamma, e_cutoff, energy):
        r'''
        '''
        return phi0 * (energy/self._E0)**(-gamma) \
                * np.exp(-energy/e_cutoff)

    def integrated_flux(self, phi0, gamma, e_cutoff, emin, emax):
        r'''
        '''
        rmax = -emax * (emax/self._E0)**(-gamma) * expn(gamma,\
                emax/e_cutoff)
        rmin = -emin * (emin/self._E0)**(-gamma) * expn(gamma, \
                emin/e_cutoff)

        return phi0 *(rmax - rmin)
        

