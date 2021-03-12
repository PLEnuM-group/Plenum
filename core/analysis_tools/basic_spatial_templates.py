# Here some basic spatial template classes are saved

import numpy as np
from astropy.coordinates import SkyCoord
from core.tools import ang_dist

class Galaxy(object):
    r'''
    '''

    def __init__(self, size_gplane, radius_gcenter):
        r'''
        '''
        self.size_gplane = size_gplane
        self.radius_gcenter = radius_gcenter

        self.pixel_size = None

    def get_mask(self, ra, sindec):
        r'''
        '''
        xx,yy = np.meshgrid(ra, np.arcsin(sindec), 
                indexing='ij')

        cords = SkyCoord(xx.flatten(), yy.flatten(),  
                unit='rad')

        lg = cords.galactic.l.rad
        bg = cords.galactic.b.rad
        
        # get the angular distance of the bins with 
        # respect to the galactic center
        ang_unc = ang_dist(0.,0., lg , bg)

        mask_template_gc = (ang_unc[0] < np.deg2rad(self.radius_gcenter)
                ).reshape(xx.shape)
        mask_template_gp = (np.abs(bg) < np.deg2rad(self.size_gplane)
                ).reshape(xx.shape)

        return mask_template_gc|mask_template_gp

class GalacticCenter(object):
    r'''
    '''

    def __init__(self, radius_gcenter):
        r'''
        '''
        self.radius_gcenter = radius_gcenter
        self.pixel_size = None

    def get_mask(self, ra, sindec):
        r'''
        '''
        xx,yy = np.meshgrid(ra, np.arcsin(sindec), 
                indexing='ij')

        cords = SkyCoord(xx.flatten(), yy.flatten(),  
                unit='rad')

        lg = cords.galactic.l.rad
        bg = cords.galactic.b.rad
        
        # get the angular distance of the bins with 
        # respect to the galactic center
        ang_unc = ang_dist(0.,0., lg , bg)

        mask_template_gc = (ang_unc[0] < np.deg2rad(self.radius_gcenter)
                ).reshape(xx.shape)
        
        return mask_template_gc


class FullSky(object):
    r'''
    '''

    def __init__(self):
        r'''
        '''
        self.pixel_size = None

    def get_mask(self, ra, sindec):
        r'''
        '''
        xx,yy = np.meshgrid(ra, np.arcsin(sindec), 
                indexing='ij')
        mask = np.ones_like(xx, dtype=bool)
        
        return mask


class NorthernHemisphere(object):
    r'''
    '''

    def __init__(self):
        r'''
        '''
        self.pixel_size = None

    def get_mask(self, ra, sindec):
        r'''
        '''
        xx,yy = np.meshgrid(ra, np.arcsin(sindec), 
                indexing='ij')
        mask = yy > np.deg2rad(-5.)
        
        return mask







