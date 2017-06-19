# -*- coding: utf-8 -*-
from random import choice, seed
import numpy as np
from .. import Atom, atomic_number
import unittest

seed(23)

# No scattering parameters for Z > 103
ELEMENTS = [key for key in atomic_number.keys() if atomic_number[key] < 104]

class TestAtom(unittest.TestCase):
    
    def test_electron_form_factor_side_effects(self):
        """ Test that arrays passed to Atom.electron_form_factors are unchanged, which
        has been a problem in the past. """
        nG = np.random.random(size = (256,))
        copied = np.copy(nG)

        a = Atom(choice(ELEMENTS), [0,0,0])
        _ = a.electron_form_factor(nG)

        self.assertTrue(np.allclose(nG, copied))