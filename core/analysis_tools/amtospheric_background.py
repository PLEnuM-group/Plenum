# This file contains the functionality to generate the atmospheric
# background that can be used within the Plenum notebooks. For the esimations 
# MCEq is used.

from core.tools import *
import numpy as np
from scipy.interpolate import splrep, splev

import mceq_config as config
from MCEq.core import MCEqRun
import crflux.models as pm


class AtmosphericBackground(object):
    r'''
    '''
    def __init__(self, interaction_model='SIBYLL23C', 
            theta_angles=np.rad2deg(np.arccos(np.linspace(0.5,0,3))), 
            ):
        r'''
        '''

        self._mceq = MCEqRun(
            interaction_model=interaction_model,
            theta_deg=0.,
            primary_model=(pm.GlobalSplineFitBeta,
                 '/Users/mhuber/AtmosphericShowers/Data/GSF_spline_20171007.pkl.bz2')
            )

        self.e_grid = self._mceq.e_grid
        # get the flux estimates
        self.flux_keys =  ['numu_conv','numu_pr','numu_total',
                           'mu_conv','mu_pr','mu_total',
                           'nue_conv','nue_pr','nue_total',
                           'nutau_pr']

        self.fluxes = self._calc_fluxes(theta_angles)
        self.flux_tcks = dict()

    def _calc_fluxes(self, theta_angles):
        r'''
        '''
        fluxes = dict()
        mceq = self._mceq
        mag=0.
        #Initialize empty grid
        for frac in self.flux_keys:
            fluxes[frac] = np.zeros_like(self.e_grid)

        #Average fluxes, calculated for different angles
        for theta in theta_angles:
            mceq.set_theta_deg(theta)
            mceq.solve()

            # same meaning of prefixes for muon neutrinos as for muons
            fluxes['mu_conv'] += (mceq.get_solution('conv_mu+', mag)
                         + mceq.get_solution('conv_mu-', mag))

            fluxes['mu_pr'] += (mceq.get_solution('pr_mu+', mag)
                       + mceq.get_solution('pr_mu-', mag))

            fluxes['mu_total'] += (mceq.get_solution('total_mu+', mag)
                          + mceq.get_solution('total_mu-', mag))


            # same meaning of prefixes for muon neutrinos as for muons
            fluxes['numu_conv'] += (mceq.get_solution('conv_numu', mag)
                         + mceq.get_solution('conv_antinumu', mag))

            fluxes['numu_pr'] += (mceq.get_solution('pr_numu', mag)
                       + mceq.get_solution('pr_antinumu', mag))

            fluxes['numu_total'] += (mceq.get_solution('total_numu', mag)
                          + mceq.get_solution('total_antinumu', mag))

            # same meaning of prefixes for electron neutrinos as for muons
            fluxes['nue_conv'] += (mceq.get_solution('conv_nue', mag)
                        + mceq.get_solution('conv_antinue', mag))

            fluxes['nue_pr'] += (mceq.get_solution('pr_nue', mag)
                      + mceq.get_solution('pr_antinue', mag))

            fluxes['nue_total'] += (mceq.get_solution('total_nue', mag)
                         + mceq.get_solution('total_antinue', mag))


            # since there are no conventional tau neutrinos, prompt=total
            fluxes['nutau_pr'] += (mceq.get_solution('total_nutau', mag)
                        + mceq.get_solution('total_antinutau', mag))

        #average the results
        for frac in self.flux_keys:
            fluxes[frac] = fluxes[frac]/float(len(theta_angles))

        return fluxes

    def interpolate_fluxes(self, key, **kwargs):
        r'''
        '''
        mceq = self._mceq
        if not 'mu' in key:
            tck = splrep(mceq.e_grid, np.log(self.fluxes[key])  , s=1.e-2,
                    **kwargs)
        else:
            # in order to converge the values at the highest energies
            # must be ignored for muons
            tck = splrep(mceq.e_grid[:-5], np.log(self.fluxes[key])[:-5]  , 
                    s=1.e-2, **kwargs)
        self.flux_tcks[key] = tck
        return

    def empty_interpolations(self, key='all'):
        r'''
        '''
        if key=='all':
            self.flux_tcks = dict()
        elif (key in self.flux_tcks.keys()):
            self.flux_tcks.pop(key)
        else:
            print('No fit provided for {0} yet!'.format(key))
        return

    def evaluate_flux(self, energy, key):
        r'''
        '''
        try:
            tck = self.flux_tcks[key]
        except:
            self.interpolate_fluxes(key)
            tck = self.flux_tcks[key]

        return np.exp(splev(energy, tck))

