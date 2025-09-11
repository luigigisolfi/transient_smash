from transient_smash.models.model import *
import unittest
import numpy as np
from scipy.stats import norm

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
        self.assertEqual(simple_model.evaluate(x=1,params=[1,1]),2)
        self.assertTrue((simple_model.evaluate(x=np.array([1,2,3]),params=[1,1])==np.array([2,3,4])).all())

    def test_set_input_data(self):
        """Test setting input data for the model."""
        simple_model = SimpleModel()
        x_data = np.array([0, 1, 2, 3])
        simple_model.set_input_data(x_data)
        self.assertTrue((simple_model.x == x_data).all())

    def test_simulator(self):
        """Test the simple model and its simulator function."""
        simple_model = SimpleModel()
        params = torch.tensor([[1.0, 1.0]])
        ### Test that simulator raises an error if input data not set
        with self.assertRaises(ValueError):
            simple_model.simulator(params)
        ### Test that simulator works correctly after setting input data
        simple_model.set_input_data(np.array([1,2,3]))
        self.assertTrue((simple_model.simulator(params)==np.array([2,3,4])).all())

    def test_set_priors(self):
        from torch.distributions import Uniform, Normal
        """Test setting prior distributions for model parameters."""
        simple_model = SimpleModel()
        # Test setting a prior which is not recognized raises an error
        with self.assertRaises(ValueError):
            simple_model.set_priors({'a': ('unknown', 0., 1.)})
        # Now set valid priors and test
        priors = {'a': ('uniform', 0., 2.), 'b': ('normal', 0., 2.)}
        simple_model.set_priors(priors)
        # Test that the priors are a list of length 2 (for a and b)
        self.assertEqual(len(simple_model.priors.dists), 2)
        # Test that the priors are the expected type
        self.assertIsInstance(simple_model.priors.dists[0], Uniform)
        self.assertIsInstance(simple_model.priors.dists[1], Normal)
        # Test that the priors have the expected parameters
        self.assertEqual(simple_model.priors.dists[0].low, 0.)
        self.assertEqual(simple_model.priors.dists[0].high, 2.)
        self.assertEqual(simple_model.priors.dists[1].loc, 0.)
        self.assertEqual(simple_model.priors.dists[1].scale, 2.)

    def test_get_sbi_priors(self):
        from torch.distributions import Uniform, Normal
        """Test getting sbi-compatible prior distributions."""
        simple_model = SimpleModel()
        # Test that getting priors without setting them raises an error
        with self.assertRaises(ValueError):
            simple_model.get_sbi_priors()
        # Now set priors and test getting them
        priors = {'a': ('uniform', 0., 2.), 'b': ('normal', 0., 2.)}
        simple_model.set_priors(priors)
        sbi_priors = simple_model.get_sbi_priors()
        # Test that the returned priors are a list of length 2 (for a and b)
        self.assertEqual(len(sbi_priors.dists), 2)
        # Test that the priors are the expected type
        self.assertIsInstance(sbi_priors.dists[0], Uniform)
        self.assertIsInstance(sbi_priors.dists[1], Normal)
        # Test that the priors have the expected parameters
        self.assertEqual(sbi_priors.dists[0].low, 0.)
        self.assertEqual(sbi_priors.dists[0].high, 2.)
        self.assertEqual(sbi_priors.dists[1].loc, 0.)
        self.assertEqual(sbi_priors.dists[1].scale, 2.)

    def test_sample_prior(self):
        """Test sampling from the prior distributions."""
        simple_model = SimpleModel()
        prior_info = {'a': ('uniform', 0., 2.), 'b': ('normal', 0., 2.)}
        simple_model.set_priors(prior_info)
        priors = simple_model.get_sbi_priors()
        samples = priors.sample((5,))
        # Test that the samples have the expected shape (5 samples of 2 parameters)
        self.assertEqual(samples.shape, (5, 2))

    def test_get_sbi_simulator(self):
        """Test getting an sbi-compatible simulator function."""
        simple_model = SimpleModel()
        simple_model.set_priors({'a': ('uniform', 0., 2.), 'b': ('uniform', 0., 2.)})
        x = np.array([1, 2, 3])
        simple_model.set_input_data(x)
        sbi_simulator = simple_model.get_sbi_simulator()
        # Test that the returned simulator is callable
        self.assertTrue(callable(sbi_simulator))
        # Test that the simulator produces expected output for given parameters
        y = sbi_simulator(torch.tensor([[1.0, 1.0]]))
        self.assertIsInstance(y, torch.Tensor)

class TestSimpleModelWithNoise(unittest.TestCase):
    def test_evaluate_with_noise(self):
        """Test the simple model with noise and its simulator function."""
        model = SimpleModelWithNoise()
        # Test evaluate method with specific noise parameters
        np.random.seed(42)  # For reproducibility
        params = [1, 1, 0, 1]  # a=1, b=1, noise_mean=0, noise_std=1
        result = model.evaluate(x=np.array([1, 2, 3]), params=params)
        expected = np.array([2, 3, 4]) + norm.rvs(loc=0, scale=1, size=3, random_state=42)
        self.assertTrue(np.allclose(result, expected))

    def test_simulator_with_noise(self):
        """Test the simulator method of the simple model with noise."""
        model = SimpleModelWithNoise()
        model.set_input_data(np.array([1, 2, 3]))
        np.random.seed(42)  # For reproducibility
        params = torch.tensor([[1.0, 1.0, 0.0, 1.0]])  # a=1, b=1, noise_mean=0, noise_std=1
        result = model.simulator(params)
        expected = np.array([2, 3, 4]) + norm.rvs(loc=0, scale=1, size=3, random_state=42)
        self.assertTrue(np.allclose(result, expected))

    def test_noise_generation(self):
        """Test the noise generation method."""
        model = SimpleModelWithNoise()
        np.random.seed(42)
        noise = model.noise(x=np.array([1, 2, 3]), noise_mean=0, noise_std=1)
        expected = norm.rvs(loc=0, scale=1, size=3, random_state=42)
        self.assertTrue(np.allclose(noise, expected))
        self.assertEqual(len(noise), 3)  # Ensure noise length matches input length

class TestSinusoidalModelWithNoise(unittest.TestCase):
    def test_evaluate_sinusoidal(self):
        """Test the sinusoidal model with noise and its simulator function."""
        model = SinusoidalModelWithNoise()
        # Test evaluate method with specific noise parameters
        np.random.seed(42)  # For reproducibility
        params = [1, 2, 0, 0, 0, 1]  # amplitude=1, frequency=2, phase=0, offset=0, noise_mean=0, noise_std=1
        result = model.evaluate(x=np.array([0, 1, 2]), params=params)
        expected = 1 * np.sin(2 * np.pi * 2 * np.array([0, 1, 2]) + 0) + 0
        expected += norm.rvs(loc=0, scale=1, size=3, random_state=42)
        self.assertTrue(np.allclose(result, expected))

    def test_simulator_sinusoidal(self):
        """Test the simulator method of the sinusoidal model with noise."""
        model = SinusoidalModelWithNoise()
        model.set_input_data(np.array([0, 1, 2]))
        np.random.seed(42)  # For reproducibility
        params = torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0, 1.0]])  # amplitude=1, frequency=2, phase=0, offset=0, noise_mean=0, noise_std=1
        result = model.simulator(params)
        expected = 1 * np.sin(2 * np.pi * 2 * np.array([0, 1, 2]) + 0) + 0
        expected += norm.rvs(loc=0, scale=1, size=3, random_state=42)
        self.assertTrue(np.allclose(result, expected))

    def test_noise_generation_sinusoidal(self):
        """Test the noise generation method in the sinusoidal model."""
        model = SinusoidalModelWithNoise()
        np.random.seed(42)
        noise = model.noise(x=np.array([0, 1, 2]), noise_mean=0, noise_std=1)
        expected = norm.rvs(loc=0, scale=1, size=3, random_state=42)
        self.assertTrue(np.allclose(noise, expected))
        self.assertEqual(len(noise), 3)  # Ensure noise length matches input length


if __name__=='__main__':
    unittest.main()