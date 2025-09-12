import numpy as np
import pytest
import torch
from scipy.stats import norm
from torch.distributions import Normal, Uniform

from transient_smash.models.model import (
    Model,
    SimpleModel,
    SimpleModelWithNoise,
    SinusoidalModelWithNoise,
)


# --------------------------
# Abstract base Model
# --------------------------

def test_abstract_model():
    """Test that the abstract Model class cannot be instantiated."""
    with pytest.raises(TypeError):
        Model()  # This should raise a TypeError since Model is abstract


# --------------------------
# SimpleModel
# --------------------------

def test_evaluate_simple_model():
    simple_model = SimpleModel()
    # integer input
    assert simple_model.evaluate(x=1, params=[1, 1]) == 2
    # numpy array input
    assert np.all(
        simple_model.evaluate(x=np.array([1, 2, 3]), params=[1, 1])
        == np.array([2, 3, 4])
    )


def test_set_input_data_simple_model():
    simple_model = SimpleModel()
    x_data = np.array([0, 1, 2, 3])
    simple_model.set_input_data(x_data)
    assert np.all(simple_model.x == x_data)


def test_simulator_simple_model():
    simple_model = SimpleModel()
    params = torch.tensor([[1.0, 1.0]])
    # should fail if input not set
    with pytest.raises(Exception):
        simple_model.simulator(params)
    # works after input set
    simple_model.set_input_data(np.array([1, 2, 3]))
    assert np.all(simple_model.simulator(params) == np.array([2, 3, 4]))


def test_set_priors_simple_model():
    simple_model = SimpleModel()
    # bad prior
    with pytest.raises(Exception):
        simple_model.set_priors({"a": ("unknown", 0.0, 1.0)})
    # good priors
    priors = {"a": ("uniform", 0.0, 2.0), "b": ("normal", 0.0, 2.0)}
    simple_model.set_priors(priors)
    assert len(simple_model.priors.dists) == 2
    assert isinstance(simple_model.priors.dists[0], Uniform)
    assert isinstance(simple_model.priors.dists[1], Normal)
    assert simple_model.priors.dists[0].low == 0.0
    assert simple_model.priors.dists[0].high == 2.0
    assert simple_model.priors.dists[1].loc == 0.0
    assert simple_model.priors.dists[1].scale == 2.0


def test_get_sbi_priors_simple_model():
    simple_model = SimpleModel()
    with pytest.raises(Exception):
        simple_model.get_sbi_priors()
    priors = {"a": ("uniform", 0.0, 2.0), "b": ("normal", 0.0, 2.0)}
    simple_model.set_priors(priors)
    sbi_priors = simple_model.get_sbi_priors()
    assert len(sbi_priors.dists) == 2
    assert isinstance(sbi_priors.dists[0], Uniform)
    assert isinstance(sbi_priors.dists[1], Normal)
    assert sbi_priors.dists[0].low == 0.0
    assert sbi_priors.dists[0].high == 2.0
    assert sbi_priors.dists[1].loc == 0.0
    assert sbi_priors.dists[1].scale == 2.0


def test_sample_prior_simple_model():
    simple_model = SimpleModel()
    simple_model.set_priors({"a": ("uniform", 0.0, 2.0), "b": ("normal", 0.0, 2.0)})
    priors = simple_model.get_sbi_priors()
    samples = priors.sample((5,))
    assert samples.shape == (5, 2)


def test_get_sbi_simulator_simple_model():
    simple_model = SimpleModel()
    simple_model.set_priors({"a": ("uniform", 0.0, 2.0), "b": ("uniform", 0.0, 2.0)})
    x = np.array([1, 2, 3])
    simple_model.set_input_data(x)
    sbi_simulator = simple_model.get_sbi_simulator()
    assert callable(sbi_simulator)
    y = sbi_simulator(torch.tensor([[1.0, 1.0]]))
    assert isinstance(y, torch.Tensor)


# --------------------------
# SimpleModelWithNoise
# --------------------------

def test_evaluate_with_noise_simple_model():
    model = SimpleModelWithNoise()
    params = [1, 1, 0, 1]  # a=1, b=1, noise_mean=0, noise_std=1
    rng = np.random.default_rng(42)
    result = model.evaluate(x=np.array([1, 2, 3]), params=params)
    expected = np.array([2, 3, 4]) + rng.normal(loc=0, scale=1, size=3)
    assert np.allclose(result, expected)


def test_simulator_with_noise_simple_model():
    model = SimpleModelWithNoise()
    model.set_input_data(np.array([1, 2, 3]))
    rng = np.random.default_rng(42)
    params = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    result = model.simulator(params)
    expected = np.array([2, 3, 4]) + rng.normal(loc=0, scale=1, size=3)
    assert np.allclose(result, expected)


def test_noise_generation_simple_model():
    model = SimpleModelWithNoise()
    rng = np.random.default_rng(42)
    noise = model.noise(x=np.array([1, 2, 3]), noise_mean=0, noise_std=1)
    expected = rng.normal(loc=0, scale=1, size=3)
    assert np.allclose(noise, expected)
    assert len(noise) == 3


# --------------------------
# SinusoidalModelWithNoise
# --------------------------

def test_evaluate_sinusoidal_model():
    model = SinusoidalModelWithNoise()
    params = [1, 2, 0, 0, 0, 1]
    rng = np.random.default_rng(42)
    result = model.evaluate(x=np.array([0, 1, 2]), params=params)
    expected = 1 * np.sin(2 * np.pi * 2 * np.array([0, 1, 2]) + 0) + 0
    expected += rng.normal(loc=0, scale=1, size=3)
    assert np.allclose(result, expected)


def test_simulator_sinusoidal_model():
    model = SinusoidalModelWithNoise()
    model.set_input_data(np.array([0, 1, 2]))
    rng = np.random.default_rng(42)
    params = torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0, 1.0]])
    result = model.simulator(params)
    expected = 1 * np.sin(2 * np.pi * 2 * np.array([0, 1, 2]) + 0) + 0
    expected += rng.normal(loc=0, scale=1, size=3)
    assert np.allclose(result, expected)


def test_noise_generation_sinusoidal_model():
    model = SinusoidalModelWithNoise()
    rng = np.random.default_rng(42)
    noise = model.noise(x=np.array([0, 1, 2]), noise_mean=0, noise_std=1)
    expected = rng.normal(loc=0, scale=1, size=3)
    assert np.allclose(noise, expected)
    assert len(noise) == 3
