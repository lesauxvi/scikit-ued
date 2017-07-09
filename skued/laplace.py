# -*- coding: utf-8 -*-
""" 
Numerical Laplace transform 
"""
import numpy as np
from scipy.fftpack import fft, fftshift

def heaviside(x):
    """ 
    Heaviside step function

    Parameters
    ----------
    x : iterable or numerical
    """
    return 1 * (np.array(x) >= 0)

def laplace(x, y, sigma = 0):
    """
    Laplace transform of y = f(x).

    Parameters
    ----------
    x : `~numpy.ndarray`, shape (N,)
        Time-array such that ``y = f(x)``.
    y : `~numpy.ndarray`, shape (N,)
        Input ``y = f(x)``.
    sigma : float or ndarray, optional
        Real-part of the Laplace transform variable s. Default is 0, which makes 
        the Laplace transform equivalent to the Fourier transform.

    Returns
    -------
    out : ndarray, shape (M,N) , dtype complex
        Laplace transform. If sigma is a float, M = 1.
    """
    # Reshape arrays to vectorize fft for multiple sigmas
    # If input sigma is numerical, change to (1,1) array
    # Sigma rows along rows, x and y along colums
    if isinstance(sigma, (int, float)):
        sigma = (sigma,)

    sigma = np.asarray(sigma, dtype = np.float)[:, None] 
    x, y = x[None, :], y[None, :]

    # We include a factor of 2 pi to sigma because fft uses exp(2 pi 1j k t)
    transformed = fft(heaviside(x) * y * np.exp(-2*np.pi*sigma*x), axis = 1)
    return fftshift(transformed, axes = 1)