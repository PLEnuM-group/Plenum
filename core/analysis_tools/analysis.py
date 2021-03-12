# This file contains the analysis class that is used for most
# of the plenum studies

from core.tools import *
import numpy as np
import abc
from copy import copy, deepcopy
import scipy

from core.progressbar.progressbar import ProgressBar

class Analysis(object, metaclass=abc.ABCMeta):
    ''' Abstract base class
    '''

    def __init__(self, energy_smearing, atm_bckg, effective_areas,
            livetime=1, rng_seed=0):

        # Call the super function to allow for multiple class inheritance.
        super(Analysis, self).__init__()

        # class instance that allows energy smearing
        self._energy_smearing = energy_smearing

        # class instance for atmospheric bckg
        self._atm_bckg = atm_bckg

        # class instance for the effective area
        self._effective_areas = effective_areas
        self._etrue_bins = effective_areas.etrue_bins
        self._etrue_mids = effective_areas.etrue_mids
        self._etrue_bin_width = np.diff(effective_areas.etrue_bins)
        self._n_etrue = len(self._etrue_mids)

        self.lt = livetime * 365.25*86400

        # define a reproducable random number generator
        self.rand = np.random.RandomState(rng_seed)


    def reset_random_number_generator(self, seed):
        r'''
        '''
        self.rand = np.random.RandomState(seed)

    def _generate_atmospheric_background_expectation(self):
        r'''
        '''
        pass

    def _generate_background_expectation(self):
        r'''
        '''
        pass
    def _generate_signal_expectation(self):
        r'''
        '''
        pass

    def _test_statistic_function(self):
        r'''
        '''
        pass

    def do_trials(self, n_trials, signal_kwargs=None, spatial_masks=None,
            bckg_kwargs={'atm_keys':'numu'}, data=None, **kwargs):
        r'''
        '''
        lambda_b = self._generate_background_expectation(**bckg_kwargs)
        if data is None:
            if signal_kwargs is not None:
                lambda_tot = dict()
                lambda_s_inj = self._generate_signal_expectation(
                    **signal_kwargs)

                for exp_key in self._effective_areas.exp_keys:
                    lambda_tot[exp_key] = lambda_b[exp_key] + lambda_s_inj[exp_key][0]
            else:
                lambda_tot = copy(lambda_b)

            data = self.generate_background_data(lambda_tot, n_trials)
        
        return_data = kwargs.pop('return_data', False)
        if return_data:
            return data, self._test_statistic_function(data, lambda_b, 
                spatial_masks=spatial_masks, **kwargs)

        return self._test_statistic_function(data, lambda_b, 
                spatial_masks=spatial_masks, **kwargs)


    def generate_background_data(self, lambda_b, n_trials, med=False):
        r'''
        '''
        if med:
            return lambda_b

        data = dict()
        for exp_key in self._effective_areas.exp_keys:
            out = self.rand.poisson(lambda_b[exp_key].flatten(), size=(n_trials, 
                len(lambda_b[exp_key].flatten())))
            data[exp_key] = out.reshape((n_trials,)+lambda_b[exp_key].shape)

        return data

    def int_powerlaw(self, phi0, gamma, emin, emax, E0=1e5):
        if gamma!=1:
            return phi0*E0**(gamma)*1./(1-gamma) * (emax**(1-gamma) - emin**(1-gamma))
        else:
            return phi0*E0**(gamma)*np.log(emax/emin)


    def _llh_ratio_function(self, data, lambda_b, lambda_s,
            axis=(1,2,3)):
        r''' Function evaluating the llh_ratio between signal
        and background hypothesis
        
        Parameters:
        -------------------
        data: array | (ntrials, bins.shape)
            observed number of events in each energy bin
        lambda_b: array | bins.shape
            expected number of background events in each bin
        lambda_s: array | (Nguess, bins.shape)
            expected number of signal events in each bin
        '''

        r = lambda_s / lambda_b 
        #background expectation should not be 0
        _mask = (np.ones_like(lambda_s)*lambda_b) == 0.
        r[_mask] = 0.

        res = 2 * np.sum(np.log(1+r)[...,np.newaxis] * data - \
                lambda_s[...,np.newaxis],axis=axis)
        return res


class GenericSpatialTemplateAnalysis(Analysis):
    r'''
    '''

    def __init__(self, sindec_bins, ra_bins, log_ereco_bins, energy_smearing, 
            atm_bckg, effective_areas, livetime=1, rng_seed=0):
        r'''
        '''
        super(GenericSpatialTemplateAnalysis, self).__init__(energy_smearing=energy_smearing, 
                atm_bckg=atm_bckg, effective_areas=effective_areas, livetime=livetime, 
                rng_seed=rng_seed)

        self._phi100_astro = 1.44e-18
        self._gamma_astro = 2.28

        # the binning used in this analysis is setup as
        # (ereco, ra, sindec)

        # the spatial binning defines equally sized bins
        # on a skymap with given pixel size
        self.sindec_bins = sindec_bins
        self.sindec_mids = get_mids(sindec_bins)  
        self.sindec_width = np.diff(sindec_bins)

        self.ra_bins = ra_bins 
        self.ra_mids = get_mids(ra_bins)
        self.ra_width = np.diff(ra_bins)
        self.pixel_sizes = (self.ra_width*self.sindec_width[:,np.newaxis]).T

        self.ereco_bins = 10**log_ereco_bins
        self.ereco_mids = 10**get_mids(log_ereco_bins)
        self._n_ereco = len(self.ereco_mids)

        # generate the smearing matrix
        xx,yy = np.meshgrid(np.log10(self._etrue_mids), 
                np.log10(self.ereco_mids), indexing='ij')
        self._energy_smearing_matrix = \
                self._energy_smearing.evaluate_energy_smearing(xx,yy, mc=True).T

        # generate a grid in ra and declination
        self._xx_ra, self._yy_sdec = np.meshgrid(self.ra_mids, self.sindec_mids, 
                indexing='ij')


        self.bckg_expectation_values = dict()


    def _generate_atmospheric_background_expectation(self, atm_key, exp_key):
        r'''
        '''
        eval_bins_sindec = self._yy_sdec.flatten()
        Ntot = np.array([(self._effective_areas.get_eff_area_values(eval_bins_sindec, 
                    ei, exp_key) * self._atm_bckg.evaluate_flux(ei, atm_key)
                    * self._etrue_bin_width[i]) 
                    for i,ei in enumerate(self._etrue_mids)])

        # convert into reco energy bins
        Ntot_reco = np.dot(self._energy_smearing_matrix, Ntot).reshape(
                (self._n_ereco,)+self._yy_sdec.shape)

        return (Ntot_reco*self.pixel_sizes*self.lt)


    def _generate_astro_background_expectation(self, phi100, gamma, 
            exp_key):
        r'''
        '''
        eval_bins_sindec = self._yy_sdec.flatten()
        Ntot = np.array([(self._effective_areas.get_eff_area_values(eval_bins_sindec, 
                    ei, exp_key) * self.int_powerlaw(phi100, gamma, self._etrue_bins[i], 
                        self._etrue_bins[i+1])) 
                    for i,ei in enumerate(self._etrue_mids)])

        # convert into reco energy bins
        Ntot_reco = np.dot(self._energy_smearing_matrix, Ntot).reshape(
                (self._n_ereco,)+self._yy_sdec.shape)

        return (Ntot_reco*self.pixel_sizes*self.lt)

    def _generate_background_expectation(self, atm_keys, astro_bckg=False):
        r'''
        '''
        atm_keys = np.atleast_1d(atm_keys)
        bckg_values = dict()
        for exp_key in self._effective_areas.exp_keys:
            bckg = np.zeros((len(self.ereco_mids),)+self._xx_ra.shape, dtype=float)
            if not exp_key in self.bckg_expectation_values.keys():
                self.bckg_expectation_values[exp_key] = dict()
            for akey in atm_keys:
                try:
                    bckg += self.bckg_expectation_values[exp_key][akey]
                except:
                    self.bckg_expectation_values[exp_key][akey] = \
                            self._generate_atmospheric_background_expectation(akey, 
                                    exp_key)
                    bckg += self.bckg_expectation_values[exp_key][akey]

            if astro_bckg:
                try:
                    bckg += self.bckg_expectation_values[exp_key]['astro']
                except:
                    self.bckg_expectation_values[exp_key]['astro'] = \
                            self._generate_astro_background_expectation(
                                    self._phi100_astro, self._gamma_astro,
                                    exp_key)
                    bckg += self.bckg_expectation_values[exp_key]['astro']

            bckg_values[exp_key] = bckg
        return bckg_values





class KRAgammaAnalysis(GenericSpatialTemplateAnalysis):
    r'''
    '''

    def __init__(self, sindec_bins, ra_bins, log_ereco_bins, energy_smearing, 
            atm_bckg, effective_areas, signal_model, livetime=1, rng_seed=0, 
            phi100_astro=1.44e-18, gamma_astro=2.28, 
            norms=None):
        r'''
        '''
        super(KRAgammaAnalysis, self).__init__(sindec_bins, ra_bins, log_ereco_bins,  
                energy_smearing=energy_smearing, atm_bckg=atm_bckg, 
                effective_areas=effective_areas, livetime=livetime, rng_seed=\
                rng_seed)


        self._phi100_astro = phi100_astro
        self._gamma_astro = gamma_astro

        self.signal_model = signal_model
        self._signal_expectations = None

        if norms is None:
            self._norms = 10**np.arange(-4,2,0.1)


    def _generate_signal_expectation_hist(self, norms):
        r'''
        '''

        kragamma_hist = self.signal_model._generate_KRAgamma_skymap(
                self.ra_mids, self.sindec_mids, self._etrue_mids)
        eval_bins_sindec = self._yy_sdec.flatten()

        self._signal_expectations = dict()
        for exp_key in self._effective_areas.exp_keys:
            xx0,yy0 = np.meshgrid(eval_bins_sindec, self._etrue_mids,
                    indexing='ij')

            effa = self._effective_areas.get_eff_area_values(xx0, 
                    yy0, exp_key).reshape(self._xx_ra.shape+(self._n_etrue,))

            a = (self.pixel_sizes[...,np.newaxis]* effa*kragamma_hist*
                    self._etrue_bin_width).reshape((np.prod(self._xx_ra.shape),
                        self._n_etrue))
            b = np.dot(self._energy_smearing_matrix, a.T).reshape(
                    (self._n_ereco,)+self._xx_ra.shape)

            self._signal_expectations[exp_key] = b[...,np.newaxis]* norms * self.lt
        return

    def _generate_signal_expectation(self, norm, exp_key=None):
        r'''
        '''
        if self._signal_expectations is None:
            self._generate_signal_expectation_hist(self._norms)

        ind = np.argmin(np.abs(self._norms-norm))
        if exp_key is None:
            signal_values = dict()
            for exp_keyi in self._effective_areas.exp_keys:
                signal_values[exp_keyi] = deepcopy(
                    self._signal_expectations[exp_keyi][...,ind])
            return signal_values
        else:
            signal_values = deepcopy(
                    self._signal_expectations[exp_key][...,ind])
            return signal_values

    def _test_statistic_function(self, data, lambda_b, spatial_masks=None):
        r'''
        '''
        out = dict()
        for exp_key in self._effective_areas.exp_keys:
            res = np.zeros(data[exp_key].shape[0], dtype=[('ts',float),
                ('norm',float)])
            llh_vals = np.zeros((data[exp_key].shape[0], len(self._norms)), dtype=float)
            for i,normi in enumerate(self._norms):
                lambda_s = self._generate_signal_expectation(norm=normi, 
                        exp_key=exp_key)
                d, lb, ls = data[exp_key], copy(lambda_b[exp_key]), copy(lambda_s)
                if spatial_masks is not None:
                    for k, lb_k in enumerate(lb):
                        lb_k[~spatial_masks[exp_key]] = 0
                        ls[k][~spatial_masks[exp_key]] = 0

                llh_vals[:,i] = self._llh_ratio_function(d, 
                        lambda_b=lb, lambda_s=ls)
    
            ts = np.max(llh_vals,axis=1)
            ts[ts<0] = 0.

            res['ts'] = ts
            res['norm'] = self._norms[np.argmax(llh_vals, axis=1)]
            out[exp_key] = res
        return out


    def _llh_ratio_function(self, data, lambda_b, lambda_s,
            axis=(1,2,3)):
        r''' Function evaluating the llh_ratio between signal
        and background hypothesis
        
        Parameters:
        -------------------
        data: array | (ntrials, bins.shape)
            observed number of events in each energy bin
        lambda_b: array | bins.shape
            expected number of background events in each bin
        lambda_s: array | (Nguess, bins.shape)
            expected number of signal events in each bin
        '''

        r = lambda_s / lambda_b 
        #background expectation should not be 0
        _mask = (np.ones_like(lambda_s)*lambda_b) == 0.
        r[_mask] = 0.

        res = 2 * np.sum(np.log(1+r)[np.newaxis,...] * data - \
                lambda_s[np.newaxis,...],axis=axis)
        return res






class ReducedSpatialTemplatePowerLawAnalysis(GenericSpatialTemplateAnalysis):
    r'''
    '''

    def __init__(self, sindec_bins, ra_bins, log_ereco_bins, energy_smearing, 
            atm_bckg, effective_areas, spatial_template, signal_model, livetime=1, rng_seed=0):
        r'''
        '''
        super(ReducedSpatialTemplatePowerLawAnalysis, self).__init__(sindec_bins, ra_bins, log_ereco_bins,  
                energy_smearing=energy_smearing, atm_bckg=atm_bckg, 
                effective_areas=effective_areas, livetime=livetime, rng_seed=\
                rng_seed)

        self.signal_model = signal_model
        self._signal_splines = None

        self._spatial_template = spatial_template
        self._spatial_template_mask = \
                self._spatial_template.get_mask(self.ra_mids,
                        self.sindec_mids)

        _, self._template_mask = np.broadcast_arrays(
                np.empty((len(self.ereco_mids),)+self._xx_ra.shape),
                self._spatial_template_mask)

    def _generate_background_expectation(self, atm_keys, astro_bckg=False):
        r'''
        '''
        bckg_values = super(ReducedSpatialTemplatePowerLawAnalysis, 
                self)._generate_background_expectation(atm_keys, astro_bckg)

        # set the expectation value to 0
        reduced_bckg_values = dict()
        for exp_key, bckg in bckg_values.items():
            reduced_bckg_values[exp_key] = \
                    bckg[self._template_mask].reshape((self._n_ereco,
                    np.sum(self._spatial_template_mask)))

        return reduced_bckg_values

    def _generate_signal_expectation_splines(self, sindec=None, 
            integrate=False):
        r'''
        '''
        splines = dict()
        xx,yy = np.meshgrid(sindec,
                self._etrue_mids, indexing='ij')

        shape = self.signal_model.params_shape
        params = np.meshgrid(*self.signal_model.params,
                indexing='ij')
        loop_vals = [pi.flatten() for pi in params]
        ppbar = ProgressBar(len(self._effective_areas.exp_keys),parent=None).start()
        for exp_key in self._effective_areas.exp_keys:
            eff_area = self._effective_areas.get_eff_area_values(xx, 
                    yy, exp_key)

            hist_exp = np.zeros((self._n_ereco, len(sindec), 
                np.prod(shape)), dtype=float)
            pbar = ProgressBar(len(loop_vals[0]), parent=ppbar).start()
            for k, pk in enumerate(zip(*loop_vals)):
                int_flux = np.array([self.signal_model.integrated_flux(*pk, 
                    emin=self._etrue_bins[j], emax=self._etrue_bins[j+1])
                    for j,ej in enumerate(self._etrue_mids)])
                int_flux = int_flux*eff_area
                hist_exp[...,k] = np.dot(self._energy_smearing_matrix,
                        int_flux.T) * self.pixel_sizes[0,0] * self.lt
                pbar.increment()
            pbar.finish()

            hist_exp = hist_exp.reshape((self._n_ereco, len(sindec),)\
                    +shape)
            # set to nonzero value to avoid probelms with the grid 
            # interpolator
            hist_exp[hist_exp==0] = 1e-15

            if not integrate:
                binmids = (self.ereco_mids, sindec, *self.signal_model.params)
                spline_exp = scipy.interpolate.RegularGridInterpolator(
                        binmids, np.log(hist_exp),
                        method="linear",
                        )
            else:
                binmids = (self.ereco_mids, *self.signal_model.params)
                spline_exp = scipy.interpolate.RegularGridInterpolator(
                        binmids, np.log(hist_exp.sum(axis=1)),
                        method="linear",
                        )
            splines[exp_key] = spline_exp
            ppbar.increment()

        ppbar.finish()
        return splines

    def _generate_signal_expectation(self, params, force_update=False, 
            exp_key=None):
        r'''
        '''
        if (self._signal_splines is None) or force_update:
            self._signal_splines = self._generate_signal_expectation_splines(
                    self.sindec_mids)

        xx, yy = np.meshgrid(self.ereco_mids, 
                self._yy_sdec[self._spatial_template_mask],
                indexing='ij')

        spl_params = [np.atleast_1d(pi)[:, np.newaxis] for pi in params]
        if exp_key is None:
            signal_vals = dict()
            for exp_keyi, spline in self._signal_splines.items():
                res = np.exp(spline((xx.flatten(), yy.flatten(),
                    *spl_params)))
                signal_vals[exp_keyi] = res.reshape((
                    len(np.atleast_1d(params[0])),)+xx.shape)
            return signal_vals

        else:
            res = np.exp(self._signal_splines[exp_key]((xx.flatten(), 
                yy.flatten(), *spl_params)))
            signal_vals = res.reshape((len(np.atleast_1d(params[0])),)\
                        +xx.shape)
            return signal_vals

    def _test_statistic_function(self, data, lambda_b, batch_size=None,
            integrate=False, **kwargs):
        r'''
        '''
        fit_params = np.meshgrid(*self.signal_model.fit_params)
        loop_vals = [pi.flatten() for pi in fit_params]
        if integrate: axis0= 1
        else: axis0=(1,2)


        out = dict()
        for exp_key in self._effective_areas.exp_keys:
            dtypes = [('ts', float)] + [(ni, float) for ni in 
                    self.signal_model.params_names]
            res = np.zeros(data[exp_key].shape[0], dtype=dtypes)
            if batch_size is None:
                lambda_s = self._generate_signal_expectation(loop_vals,
                        exp_key=exp_key)
            else:
                nbatches = int(np.ceil(len(loop_vals[0]) / batch_size ))
                if integrate: grid_shape = (self._n_ereco, )
                else: grid_shape = (self._n_ereco, np.sum(self._spatial_template_mask))
                lambda_s = np.zeros(((len(loop_vals[0]),)+grid_shape), 
                        dtype='float32')
                for l in range(nbatches):
                    loop_valsl = [lk[l*batch_size: (l+1)*batch_size] 
                            for lk in loop_vals]
                    lambda_s[l*batch_size: (l+1)*batch_size] = \
                            self._generate_signal_expectation(loop_valsl,
                                    exp_key=exp_key)

            # now bring the data into the correct shape
            di = np.swapaxes(data[exp_key][...,np.newaxis], axis1=0,axis2=-1)
            
            if batch_size is None:
                llh_scan = self._llh_ratio_function(di, lambda_b[exp_key], 
                        lambda_s, axis=axis0).T
            else:
                llh_scan = np.empty((data[exp_key].shape[0],loop_vals[0]))
                for l in range(nbatches):
                    llh_scan[...,l*batch_size: (l+1)*batch_size] = \
                            self._llh_ratio_function(di, lambda_b[exp_key],
                                    lambda_s=lambda_s[l*batch_size:\
                                            (l+1)*batch_size], axis=axis0).T

            llh_scan = llh_scan.reshape((data[exp_key].shape[0],)\
                    +fit_params[0].shape)

            axis = tuple(np.arange(len(self.signal_model.params))+1)
            max_vals = np.max(llh_scan, axis=axis)
            max_vals[max_vals<0] = 0

            res['ts'] = max_vals
            
            # Now reshape the array to get the best fit parameters
            llh_scan = llh_scan.reshape(llh_scan.shape[0],
                    np.prod(llh_scan.shape[1:]))
            m = np.argmax(llh_scan, axis=1)
            for i,ni in enumerate(self.signal_model.params_names):
                res[ni] = loop_vals[i][m]

            out[exp_key] = res
        return out



class ReducedIntegratedSpatialTemplatePowerLawAnalysis(ReducedSpatialTemplatePowerLawAnalysis):
    r'''
    '''

    def __init__(self, sindec_bins, ra_bins, log_ereco_bins, energy_smearing, 
            atm_bckg, effective_areas, spatial_template, signal_model, 
            livetime=1, rng_seed=0):
        r'''
        '''
        super(ReducedIntegratedSpatialTemplatePowerLawAnalysis, self).__init__(sindec_bins, 
                ra_bins, log_ereco_bins, energy_smearing=energy_smearing, 
                atm_bckg=atm_bckg, spatial_template=spatial_template,
                effective_areas=effective_areas, signal_model=signal_model, livetime=livetime, 
                rng_seed=rng_seed)


    def _generate_signal_expectation(self, params, force_update=False, 
            exp_key=None):
        r'''
        '''
        if (self._signal_splines is None) or force_update:
            sindec = self._yy_sdec[self._spatial_template_mask]
            self._signal_splines = self._generate_signal_expectation_splines(
                    sindec, integrate=True)

        xx = self.ereco_mids
        spl_params = [np.atleast_1d(pi)[:, np.newaxis] for pi in params]
        if exp_key is None:
            signal_vals = dict()
            for exp_keyi, spline in self._signal_splines.items():
                res = np.exp(spline((xx,
                    *spl_params)))
                signal_vals[exp_keyi] = res.reshape((
                    len(np.atleast_1d(params[0])),)+xx.shape)
            return signal_vals

        else:
            res = np.exp(self._signal_splines[exp_key]((xx, 
                 *spl_params)))
            signal_vals = res.reshape((len(np.atleast_1d(params[0])),)\
                        +xx.shape)
            return signal_vals



    def _generate_background_expectation(self, atm_keys, astro_bckg=False):
        r'''
        '''
        bckg_values = super(ReducedIntegratedSpatialTemplatePowerLawAnalysis, 
                self)._generate_background_expectation(atm_keys, astro_bckg)

        # integrate over all spatial bins
        reduced_bckg_values = dict()
        for exp_key, bckg in bckg_values.items():
            reduced_bckg_values[exp_key] = \
                    bckg.sum(axis=1)

        return reduced_bckg_values














