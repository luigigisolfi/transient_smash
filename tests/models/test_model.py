from transient_smash.models.model import *
import unittest
import numpy as np
from scipy.stats import norm
from sbi.utils import process_prior, BoxUniform
import torch
from torch.distributions import Normal, Uniform, Exponential, MultivariateNormal
from torch.distributions import Independent

class TestModel(unittest.TestCase):
    def test_abstract_model(self):
        """Test that the abstract Model class cannot be instantiated."""
        with self.assertRaises(TypeError):
            model = Model()  # This should raise a TypeError since Model is abstract

class TestSimpleModel(unittest.TestCase):
    
    def test_evaluate(self):
        """Test the simple model and its simulator function."""
        simple_model = SimpleModel()
        ### Test in the case of an integer input or an np array
        self.assertEqual(simple_model.evaluate(x=1,a=1,b=1),2)
        self.assertTrue((simple_model.evaluate(x=np.array([1,2,3]),a=1,b=1)==np.array([2,3,4])).all())

    def test_set_input_data(self):
        """Test setting input data for the model."""
        simple_model = SimpleModel()
        x_data = np.array([0, 1, 2, 3])
        simple_model.set_input_data(x_data)
        self.assertTrue((simple_model.x == x_data).all())

    def test_simulator(self):
        """Test the simple model and its simulator function."""
        simple_model = SimpleModel()
        a = 1
        b = 1
        ### Test that simulator raises an error if input data not set
        with self.assertRaises(ValueError):
            simple_model.simulator(a,b)
        ### Test that simulator works correctly after setting input data
        simple_model.set_input_data(np.array([1,2,3]))
        self.assertTrue((simple_model.simulator(a,b)==np.array([2,3,4])).all())

    def test_set_priors(self):

        """Test setting prior distributions for model parameters."""
        simple_model = SimpleModel()
        # Test setting a prior which is not recognized raises an error
        with self.assertRaises(ValueError):
            simple_model.set_priors({'a': ('unknown', 0, 1)})
        # Now set valid priors and test
        priors = {
            'velocity': ("uniform", 1, 0, 2),        # 1D uniform for velocity
            'position': ("normal", 2, 0, 2),         # 2D normal for position
            'orientation': ("exponential", 1)       # 4D normal for orientation
        }

        simple_model.set_priors(priors)

        # Test that the priors are a list of length 2 (for a and b)
        self.assertEqual(len(simple_model.priors.dists), 3)

        # Test that the priors are the expected type
        self.assertIsInstance(simple_model.priors.dists[0], Independent)
        self.assertIsInstance(simple_model.priors.dists[0], torch.distributions.Distribution)


    def test_get_sbi_priors(self):
        """Test getting sbi-compatible prior distributions."""
        simple_model = SimpleModel()
        # Test that getting priors without setting them raises an error
        with self.assertRaises(ValueError):
            simple_model.get_sbi_priors()
        # Now set priors and test getting them
        priors = {
            'velocity': ("uniform", 1, 0, 2),        # 1D uniform for velocity
            'position': ("normal", 2, 0, 2),         # 2D normal for position
            'orientation': ("exponential", 1)       # 4D normal for orientation
        }

        simple_model.set_priors(priors)
        sbi_priors = simple_model.get_sbi_priors()

        # Test that the priors are the expected length and type
        self.assertEqual(len(sbi_priors.dists), len(priors.keys()))
        self.assertIsInstance(sbi_priors.dists[0], BoxUniform)
        self.assertIsInstance(sbi_priors.dists[1], MultivariateNormal)
        self.assertIsInstance(sbi_priors.dists[2], Exponential)


def test_get_sbi_simulator(self):
        """Test getting an sbi-compatible simulator function."""
        simple_model = SimpleModel()
        priors = {
            'velocity': ("uniform", 1, 0, 2),        # 1D uniform for velocity
            'position': ("normal", 2, 0, 2),         # 2D normal for position
            'orientation': ("exponential", 1)       # 4D normal for orientation
        }
        simple_model.set_priors(priors)
        sbi_simulator = simple_model.get_sbi_simulator()
        # Test that the returned simulator is callable
        self.assertTrue(callable(sbi_simulator))
        # Test that the simulator produces expected output for given parameters
    
class TestNoisySimpleModel(unittest.TestCase):

    def test_evaluate_with_noise(self):
        """Test the simple model and its simulator function."""
        simple_model = SimpleModel_PlusSimpleNoise()

        ### Test in the case of an np array with a fixed random seed, 42
        self.assertEqual(simple_model.evaluate(np.array([1]),1,1,0.2),2+norm.rvs(loc=0,scale=0.2,size=1,random_state=42))
        # self.assertTrue((simple_model.simulator(x=np.array([1,2,3]),a=1,b=1)==np.array([2,3,4])).all())

if __name__=='__main__':
    unittest.main()