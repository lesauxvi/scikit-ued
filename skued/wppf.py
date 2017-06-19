# -*- coding: utf-8 -*-
"""
Routines for whole powder-pattern fitting (WPPF)
"""

import numpy as np
from numpy import pi
from .voigt import pseudo_voigt
from .simulation import powdersim

def rietveld(crystal, s, I):
    """ 
    Extract diffracted intensities from data according to a Rietveld fit.
    
    Parameters
    ----------
    crystal : skued.structure.Crystal instance

    s, I : array_like, shape (N,)
        Scattering length and diffracted intensity data, respectively.
    
    Returns
    -------
    intensities : list of 4-tuples
        List of tuples of the form (intensity, h, k, l) for all reflections (h, k, l)
        that are within the bounds of `s`.
    """
    # TODO: integrate
    s, I = np.array(s), np.array(I)

    # Structure factors from model
    h, k, l = crystal.bounded_reflections(4*np.pi*s.max())
    Gx, Gy, Gz = crystal.scattering_vector(h, k, l)
    SF = crystal.structure_factor( (Gx, Gy, Gz) )

    s_reflections = np.sqrt(Gx**2 + Gy**2 + Gz**2)/(4*pi)
    simulated = powdersim(crystal, s_reflections)

    peak_mult_corr = np.abs(SF)**2/simulated
    extracted_intensities = np.interp(s_reflections, s, I) * peak_mult_corr

    return list(zip(extracted_intensities, h, k, l))

#def lebail(crystal, s, I, max_iter = 100):
#    """
#    Le Bail decomposition of electron powder diffraction pattern.
#
#    Parameters
#    ----------
#    crystal : skued.structure.Crystal instance
#
#    s, I : array_like, shape (N,)
#        Scattering length and diffracted intensity, respectively.
#    max_iter : int
#        Maximum number of iterations.
#
#    Returns
#    -------
#    """
#    s, I = np.array(s), np.array(I)
#    wg, wl = 0.01, 0.02 # FWHM of Gaussian and Lorentzian components of the Voigt
#
#    # Extract the experimental intensities at the peaks
#    # according to the model
#    h, k, l = crystal.bounded_reflections(4*np.pi*s.max())
#    Gx, Gy, Gz = crystal.scattering_vector(h, k, l)
#    s_theo = np.sqrt( Gx**2 + Gy**2 + Gz**2 )/(4*np.pi)
#    exp_intensities = np.interp(s_theo, s, I)
#
#    # Initial decomposed intensities are nonsense
#    calc_intensities = np.ones_like(h, dtype = np.float)
#
#    synthetic = np.empty_like(s_theo, dtype = np.float)
#    for _ in range(max_iter):
#
#        # Compute synthetic pattern from last iteration's intensities
#        synthetic[:] = np.zeros_like(s_theo, dtype = np.float)
#        for center, intensity in zip(s_theo, exp_intensities):
#            synthetic += intensity * pseudo_voigt(s_theo, center, fwhm_g = wg, fwhm_l = wl)
#        
#        # Calculate proportion of peak intensity per reflection
#        calc_intensities[:] = (calc_intensities/synthetic)*exp_intensities
#    
#    return list(zip(calc_intensities, h, k, l))