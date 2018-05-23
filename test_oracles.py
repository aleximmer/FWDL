import unittest
import numpy as np
from oracles import LMO_l1, LMO_nuclear


class TestLMOl1(unittest.TestCase):
    def test_feasibility(self):
        for i in range(100):
            grad = np.random.randn(3) * 10 + (np.random.rand(3) - 0.5) * 5
            l1 = np.linalg.norm(grad, ord=1)
            kappa = l1 * 0.9
            s = LMO_l1(grad, kappa=kappa)
            fs = np.linalg.norm(s, ord=1)
            self.assertAlmostEqual(fs, kappa)

    def test_reshaping(self):
        v = np.zeros((10, 10))
        v[3, 5] = 2
        s = LMO_l1(v, kappa=2)
        self.assertEqual(-2, s[3, 5])


class TestLMONuclear(unittest.TestCase):
    def test_feasibility(self):
        grad = np.random.randn(3, 3) * 10 + (np.random.rand(3, 3) - 0.5) * 5
        nuc = np.linalg.norm(grad, ord='nuc')
        kappa = nuc * 0.9
        s = LMO_nuclear(grad, kappa=kappa)
        fs = np.linalg.norm(s, ord='nuc')
        self.assertAlmostEqual(fs, kappa)
