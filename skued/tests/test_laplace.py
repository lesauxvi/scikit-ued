
import numpy as np
from scipy.fftpack import fft, fftshift
import unittest

from ..laplace import laplace, heaviside

np.random.seed(23)

class TestLaplace(unittest.TestCase):

    def setUp(self):
        self.x = np.linspace(0, 10, 512) # fft is fastest on arrays with size of powers of 2
        self.y = np.random.random_sample(size = self.x.shape)
    
    def test_heaviside_arrays(self):
        """ Test Heaviside function on array arguments"""
        x = np.arange(10, 20, step = 1)
        self.assertTrue(np.allclose(np.ones_like(x), heaviside(x)))

        x2 = np.arange(-10, -1, step = 1)
        self.assertTrue(np.allclose(np.zeros_like(x2), heaviside(x2)))
        
        # Test that heaviside([0 0 0 0 ... ]) = [1 1 1 1 ...]
        x3 = np.zeros(shape = (10,))
        self.assertTrue(np.allclose(np.ones_like(x3), heaviside(x3)))
    
    def test_heaviside_numerical(self):
        """ Test heaviside function on float/int arguments """
        self.assertEqual(heaviside(0), 1)
        self.assertEqual(heaviside(-1), 0)
        self.assertEqual(heaviside(1), 1)

    def test_fourier_equivalent(self):
        """ Test that the Laplace transform with s purely imaginary is
        equivalent to the Fourier transform """
        # x must be >= 0 for this test to pass 
        direct_fft = fftshift(fft(self.y))
        laplace_fft = laplace(self.x, self.y, sigma = 0)
        self.assertTrue(np.allclose(direct_fft, laplace_fft))
    
    def test_fourier_equivalent_sigma_1(self):
        """ Test that laplace transform behaves as expected for sigma = 1 """
        mod_y = heaviside(self.x) * self.y * np.exp(-2*np.pi*self.x)
        direct_fft = fftshift(fft(mod_y))
        self.assertTrue(np.allclose(direct_fft, laplace(self.x, self.y, sigma = 1)))