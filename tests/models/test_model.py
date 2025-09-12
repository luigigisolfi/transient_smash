from transient_smash.models.model import Model, SimpleModel, \
SimpleModelWithNoise, SinusoidalModelWithNoise
import unittest
import numpy as np
from scipy.stats import norm
import pytest
import torch
from torch.distributions import Uniform, Normal

class TestModel(unittest.TestCase):
    """Test the abstract Model class."""
    def test_abstract_model(self) -> None:
        """Test that the abstract Model class cannot be instantiated."""
        with pytest.raises(TypeError):
            Model()  # This should raise a TypeError since Model is abstract

class TestSimpleModel(unittest.TestCase):
    """Test the SimpleModel class."""
    def test_evaluate(self) -> None:
        """Test the simple model and its simulator function."""
        simple_model = SimpleModel()
        ### Test in the case of an integer input or an np array
        assert simple_model.evaluate(x=1,params=[1,1]) == 2
        assert (simple_model.evaluate(x=np.array([1,2,3]),params=[1,1]) == 
                np.array([2,3,4])).all()

    def test_set_input_data(self) -> None:
        """Test setting input data for the model."""
        simple_model = SimpleModel()
        x_data = np.array([0, 1, 2, 3])
        simple_model.set_input_data(x_data)
        assert (simple_model.x == x_data).all()

    def test_simulator(self) -> None:
        """Test the simple model and its simulator function."""
        simple_model = SimpleModel()
        params = torch.tensor([[1.0, 1.0]])
        ### Test that simulator raises an error if input data not set
        with pytest.raises(Exception):
            simple_model.simulator(params)
        ### Test that simulator works correctly after setting input data
        simple_model.set_input_data(np.array([1,2,3]))
        assert (simple_model.simulator(params) == np.array([2,3,4])).all()

    def test_set_priors(self) -> None:
        """Test setting prior distributions for model parameters."""
        simple_model = SimpleModel()
        # Test setting a prior which is not recognized raises an error
        with pytest.raises(Exception):
            simple_model.set_priors({'a': ('unknown', 0., 1.)})
        # Now set valid priors and test
        priors = {'a': ('uniform', 0., 2.), 'b': ('normal', 0., 2.)}
        simple_model.set_priors(priors)
        # Test that the priors are a list of length 2 (for a and b)
        assert len(simple_model.priors.dists) == 2
        # Test that the priors are the expected type
        assert isinstance(simple_model.priors.dists[0], Uniform)
        assert isinstance(simple_model.priors.dists[1], Normal)
        # Test that the priors have the expected parameters
        assert simple_model.priors.dists[0].low == 0.
        assert simple_model.priors.dists[0].high == 2.
        assert simple_model.priors.dists[1].loc == 0.
        assert simple_model.priors.dists[1].scale == 2.

    def test_get_sbi_priors(self) -> None:
        """Test getting sbi-compatible prior distributions."""
        simple_model = SimpleModel()
        # Test that getting priors without setting them raises an error
        with pytest.raises(Exception):
            simple_model.get_sbi_priors()
        # Now set priors and test getting them
        priors = {'a': ('uniform', 0., 2.), 'b': ('normal', 0., 2.)}
        simple_model.set_priors(priors)
        sbi_priors = simple_model.get_sbi_priors()
        # Test that the returned priors are a list of length 2 (for a and b)
        assert len(sbi_priors.dists) == 2
        # Test that the priors are the expected type
        assert isinstance(sbi_priors.dists[0], Uniform)
        assert isinstance(sbi_priors.dists[1], Normal)
        # Test that the priors have the expected parameters
        assert sbi_priors.dists[0].low == 0.
        assert sbi_priors.dists[0].high == 2.
        assert sbi_priors.dists[1].loc == 0.
        assert sbi_priors.dists[1].scale == 2.

    def test_sample_prior(self) -> None:
        """Test sampling from the prior distributions."""
        simple_model = SimpleModel()
        prior_info = {'a': ('uniform', 0., 2.), 'b': ('normal', 0., 2.)}
        simple_model.set_priors(prior_info)
        priors = simple_model.get_sbi_priors()
        samples = priors.sample((5,))
        # Test that the samples have the expected shape
        assert samples.shape == (5, 2)

    def test_get_sbi_simulator(self) -> None:
        """Test getting an sbi-compatible simulator function."""
        simple_model = SimpleModel()
        simple_model.set_priors({'a': ('uniform', 0., 2.), 
                                 'b': ('uniform', 0., 2.)})
        x = np.array([1, 2, 3])
        simple_model.set_input_data(x)
        sbi_simulator = simple_model.get_sbi_simulator()
        # Test that the returned simulator is callable
        assert callable(sbi_simulator)
        # Test that the simulator produces expected output for given parameters
        y = sbi_simulator(torch.tensor([[1.0, 1.0]]))
        assert isinstance(y, torch.Tensor)

class TestSimpleModelWithNoise(unittest.TestCase):
    """Test the SimpleModelWithNoise class."""
    def test_evaluate_with_noise(self) -> None:
        """Test the simple model with noise and its simulator function."""
        model = SimpleModelWithNoise()
        # Test evaluate method with specific noise parameters
        np.random.seed(42)  # For reproducibility
        params = [1, 1, 0, 1]  # a=1, b=1, noise_mean=0, noise_std=1
        result = model.evaluate(x=np.array([1, 2, 3]), params=params)
        expected = np.array([2, 3, 4]) + norm.rvs(loc=0, scale=1, size=3, random_state=42)
        assert np.allclose(result, expected)

    def test_simulator_with_noise(self) -> None:
        """Test the simulator method of the simple model with noise."""
        model = SimpleModelWithNoise()
        model.set_input_data(np.array([1, 2, 3]))
        np.random.seed(42)  # For reproducibility
        params = torch.tensor([[1.0, 1.0, 0.0, 1.0]])  
        result = model.simulator(params)
        expected = np.array([2, 3, 4]) + \
            norm.rvs(loc=0, scale=1, size=3, random_state=42)
        assert np.allclose(result, expected)

    def test_noise_generation(self) -> None:
        """Test the noise generation method."""
        model = SimpleModelWithNoise()
        np.random.seed(42)
        noise = model.noise(x=np.array([1, 2, 3]), noise_mean=0, noise_std=1)
        expected = norm.rvs(loc=0, scale=1, size=3, random_state=42)
        assert np.allclose(noise, expected)
        assert len(noise) == 3  # Ensure noise length matches input length

class TestSinusoidalModelWithNoise(unittest.TestCase):
    """Test the SinusoidalModelWithNoise class."""
    def test_evaluate_sinusoidal(self) -> None:
        """Test the sinusoidal model with noise and its simulator function."""
        model = SinusoidalModelWithNoise()
        # Test evaluate method with specific noise parameters
        np.random.seed(42)  # For reproducibility
        params = [1, 2, 0, 0, 0, 1]  
        result = model.evaluate(x=np.array([0, 1, 2]), params=params)
        expected = 1 * np.sin(2 * np.pi * 2 * np.array([0, 1, 2]) + 0) + 0
        expected += norm.rvs(loc=0, scale=1, size=3, random_state=42)
        assert np.allclose(result, expected)

    def test_simulator_sinusoidal(self) -> None:
        """Test the simulator method of the sinusoidal model with noise."""
        model = SinusoidalModelWithNoise()
        model.set_input_data(np.array([0, 1, 2]))
        np.random.seed(42)  # For reproducibility
        params = torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0, 1.0]])  
        result = model.simulator(params)
        expected = 1 * np.sin(2 * np.pi * 2 * np.array([0, 1, 2]) + 0) + 0
        expected += norm.rvs(loc=0, scale=1, size=3, random_state=42)
        assert np.allclose(result, expected)

    def test_noise_generation_sinusoidal(self) -> None:
        """Test the noise generation method in the sinusoidal model."""
        model = SinusoidalModelWithNoise()
        np.random.seed(42)
        noise = model.noise(x=np.array([0, 1, 2]), noise_mean=0, noise_std=1)
        expected = norm.rvs(loc=0, scale=1, size=3, random_state=42)
        assert np.allclose(noise, expected)
        assert len(noise) == 3  # Ensure noise length matches input length


if __name__=='__main__':
    unittest.main()