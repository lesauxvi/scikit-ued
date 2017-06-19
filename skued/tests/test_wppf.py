# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np
import unittest

from ..structure import graphite
from ..simulation import powdersim
from .. import rietveld, pseudo_voigt

def recompose(s, intensities):
	""" Recompose a pattern from list of (intensities, h, k, l) """


class TestRietveldDecomposition(unittest.TestCase):

	def setUp(self):
		self.crystal = deepcopy(graphite)
	
	def test_trivial(self):
		""" Test that Rietveld decomposition on simulated data is correct
		within 5% """
		s = np.linspace(0.1, 1, 512)
		I = powdersim(self.crystal, s)
		I /= I.max()

		decomp = rietveld(self.crystal, s, I)

		pattern = np.zeros_like(s)
		for (intensity, h, k, l) in decomp:
			G = self.crystal.scattering_vector(h, k, l)
			center = np.linalg.norm(G)/(4*np.pi)
			pattern += intensity*pseudo_voigt(s, center, 0.01, 0.02)
		
		# Don't compare the edges due to artifacts
		self.assertTrue(np.allclose(I[50:-50], pattern[50:-50], rtol = 0.05))

if __name__ == '__main__':
	unittest.main()