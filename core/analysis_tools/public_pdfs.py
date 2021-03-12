# This file contains a selection of public pdfs that are used within 
# the Plenum notebooks
from core.tools import *
import numpy as np
import scipy


class  EffectiveAreas(object):
    r''' This class loads the public effective areas
    and allows to use it within the analyses
    '''

    def __init__(self, path='resources/effective_areas_av_per_day_fine_binning.npy', 
            log_etrue_bins=np.arange(2,8.1,0.1), sindec_bins = np.linspace(-1.,1.,35) ):
        r'''
        '''

        self.etrue_mids = 10**((log_etrue_bins[1:]+log_etrue_bins[:-1])/2.)
        self.etrue_bins = 10**log_etrue_bins
        self.etrue_bin_widths = np.diff(self.etrue_bins)

        self.sindec_bins = sindec_bins
        self.sindec_mids = get_mids(sindec_bins)
        self.n_sd, self.n_etrue = len(self.sindec_bins)-1, len(self.etrue_bins)-1

        self._eff_area_hists = np.load(path)
        self.exp_keys = [k for k in self._eff_area_hists.dtype.fields.keys() 
                if 'effA' in k]
        self._eff_area_interpolation = self._generate_eff_area_interpolation()

        

    def _generate_eff_area_interpolation(self):
        r''' This function generates a grid interpolator instance 
        for each effective area histogram
        '''

        binmids = (get_mids(self.sindec_bins, ext=True), self.etrue_mids)
        eff_areas = dict()
        keys = self.exp_keys
        for exp_key in keys:
            eff_area_hist = self._eff_area_hists[exp_key]
            eff_areas[exp_key] = scipy.interpolate.RegularGridInterpolator(
                    binmids, np.log(eff_area_hist.reshape((self.n_sd, 
                        self.n_etrue))),
                    method="linear",
                    bounds_error=False,
                    fill_value=-10.
                    )

        return eff_areas


    def get_eff_area_values(self, sindec_vals, etrue_vals, exp_key):
        r''' This function evaluates the effective area at the given 
        phase space position

        Parameters:
        ----------------
        sindec_vals: float or array
            sinus declination values
        etrue_vals: float or array
            true energy values
        exp_key: str
            key of the respective experiment for which the effective area shall
            be evaluated
        '''
        return np.exp(self._eff_area_interpolation[exp_key]((
            sindec_vals, etrue_vals)))



class TrueEnergy2ReconstructedEnergySmearing(object):
    r''' This class handles the conversion from event counts in true neutrino
    energy bins towards reconstructed energy bins. Currently the conversion 
    matrix can be constructed either from public data or icecube internal 
    monte carlo
    '''

    def __init__(self, path_public='resources/IC86_II_smearing.csv',
            mc_path=None):
        r'''
        '''
        public_data_hist = np.genfromtxt(path_public,
                        skip_header=1)
        log_emin, log_emax = public_data_hist[:,0], public_data_hist[:,1]
        self._log_emids = (log_emin+log_emax)/2.
        
        log_ereco_min, log_ereco_max = public_data_hist[:,4], public_data_hist[:,5]
        self._log_ereco_mids = (log_ereco_min+log_ereco_max)/2.

        self._event_counts = public_data_hist[:,-1]
        self._public_smearing_function = self._get_interpolation(
                self._log_emids, self._log_ereco_mids, 
                np.arange(2,8.1,0.5), np.arange(2,8.1,0.5),
                weights=self._event_counts)

        self._mc_smearing_function = None
        if mc_path is not None:
            idata = np.load(mc_path)
            # remove all events that don't have a proper energy proxy
            k_mc = 'muon_energy_entry'
            mask = np.isnan(idata[k_mc])
            idata = idata[~mask]

            log_bins_reco = np.linspace(2, 8.1, 45)
            log_bins_mc = np.linspace(2, 8.2, 50)
            
            mc_weights = idata['ow'] # i'm ingoring the energy 
            # dependence in the energy bins here
            self._mc_smearing_function = self._get_interpolation(
                np.log10(idata['trueE']), np.log10(idata['energy']),
                log_bins_mc, log_bins_reco, weights=mc_weights)


    def _get_interpolation(self, log_etrue_vals, log_ereco_vals, log_bins_etrue, 
            log_bins_ereco, weights):
        r''' This function generates an interpolation instance for the 
        energy smearing

        Parameters:
        ----------------
        log_etrue_vals: array
            contains the log10 of the true energy values

        '''

        bins = (log_bins_etrue, log_bins_ereco)
        h,_,_= np.histogram2d(log_etrue_vals, log_ereco_vals, 
            bins=bins, weights=weights)

        h = h / h.sum(axis=1)[:,np.newaxis]

        binmids = tuple([get_mids(bi, ext=True) for bi in bins])
        spline = scipy.interpolate.RegularGridInterpolator(
                    binmids, np.log(h),
                    method="linear",
                    bounds_error=False,
                    fill_value=-10.
                    )

        return spline

    def evaluate_energy_smearing(self, log_true_energy, log_reco_energy, mc=False):
        r'''
        '''
        if not mc:
            spl_func = self._public_smearing_function
        else:
            spl_func = self._mc_smearing_function

        res = np.exp(spl_func((log_true_energy, log_reco_energy)))
        res = res/res.sum(axis=1)[:,np.newaxis]
        return res





