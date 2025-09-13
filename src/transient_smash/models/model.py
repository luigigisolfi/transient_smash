from abc import ABC, abstractmethod

import numpy as np
import torch
from sbi.utils import MultipleIndependent, process_simulator
from scipy.stats import norm
from torch.distributions import Normal, Uniform

from .exceptions import ModelError


class Model(ABC):
    """Abstract base class for models."""

    @abstractmethod
    def evaluate(self, x: np.ndarray, *args: np.ndarray) -> np.ndarray:
        """
        Evaluate the model given input data and model parameters.

        Args:
            x: Input data for the model (i.e. time points at which to simulate).
            *args: Additional arguments for the evaluation containing model parameters.

        Returns:
            Model output evaluated at the given input data.

        """
        ...

    def simulator(self, *args: np.ndarray) -> np.ndarray:
        """
        Generate simulated data given the pre-set input data and model parameters.

        Args:
            *args: Additional arguments for the simulation containing model parameters.

        Returns:
            Simulated output of the model.

        """
        # Check if input data has been set
        if not hasattr(self, "x"):
            err_str = (
                "Input data 'x' has not been set. Please use the "
                "'set_input_data' method to set it before calling the simulator."
            )
            raise ModelError(err_str)
        return self.evaluate(self.x, *args)

    def set_input_data(self, x: np.ndarray) -> None:
        """
        Set input data for the model.

        Args:
            x: Input data for the model (i.e. time points at which to simulate).

        Returns:
            None

        """
        self.x = x

    def get_sbi_simulator(self) -> callable:
        """Get a simulator function compatible with the sbi package."""
        priors = self.get_sbi_priors()
        is_numpy_simulator = True
        return process_simulator(self.simulator, priors, is_numpy_simulator)

    def set_priors(self, priors: dict) -> None:
        """
        Set prior distributions for model parameters.

        Args:
            priors (dict): A dictionary containing prior distribution information for each parameter.

        Returns:
            None

        """
        priors_list = []
        for param in priors:
            dist = priors[param][0]
            if dist.lower() == "uniform":
                priors_list.append(
                    Uniform(
                        torch.tensor([priors[param][1]]),
                        torch.tensor([priors[param][2]]),
                    )
                )
            elif dist.lower() == "normal":
                priors_list.append(
                    Normal(
                        torch.tensor([priors[param][1]]),
                        torch.tensor([priors[param][2]]),
                    )
                )
            else:
                err_str = (
                    f"Invalid distribution selected: {dist}. "
                    f"Please choose either 'uniform' or 'normal'."
                )
                raise ModelError(err_str)
        self.priors = MultipleIndependent(priors_list)

    def get_sbi_priors(self) -> MultipleIndependent:
        """
        Get prior distributions for model parameters.

        Returns:
            Distribution-like priors for each parameter.

        """
        if not hasattr(self, "priors"):
            err_str = (
                "Priors have not been set. Please use the "
                "'set_priors' method to set them before calling this method."
            )
            raise ModelError(err_str)

        return self.priors


class SimpleModel(Model):
    """A simple linear model: y = a * x + b."""

    def evaluate(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Evaluate the simple linear model.

        Args:
            x: np.ndarray of shape (M,)
            params: np.ndarray of shape (N, 2)

        Returns:
            np.ndarray of shape (N, M)

        """
        params = np.array(params)  # handle np/torch uniformly

        if params.ndim == 1:  # single (a, b)
            a, b = params
            return a * x + b

        if params.ndim == 2:  # multiple (a, b) pairs
            a = params[:, 0][:, None]  # shape (N, 1)
            b = params[:, 1][:, None]  # shape (N, 1)
            return a * x[None, :] + b

        err_str = f"Unexpected params shape: {params.shape}"
        raise ModelError(err_str)


class SimpleModelWithNoise(Model):
    """A simple linear model with Gaussian noise: y = a * x + b + noise."""

    def evaluate(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Evaluate the simple linear model with noise.

        Args:
            x: Input data for the model.
            params: Model parameters (a, b, noise_mean, noise_std).

        Returns:
            Model output evaluated at the given input data plus noise.

        """
        params = np.array(params)  # handle both np/torch uniformly
        if params.ndim == 1:  # single (a, b, noise_mean, noise_std)
            a, b, noise_mean, noise_std = params
        elif params.ndim == 2:  # multiple (a, b, noise_mean, noise_std) pairs
            a = params[:, 0][:, None]  # shape (N, 1)
            b = params[:, 1][:, None]  # shape (N, 1)
            noise_mean = params[:, 2][:, None]  # shape (N, 1)
            noise_std = params[:, 3][:, None]
        else:
            err_str = f"Unexpected params shape: {params.shape}"
            raise ModelError(err_str)

        return a * x + b + self.noise(x, noise_mean, noise_std)

    def noise(
        self, x: np.ndarray, noise_mean: np.ndarray, noise_std: np.ndarray
    ) -> np.ndarray:
        """
        Simulate the noise to apply to the model.

        Args:
            x: Input data for the model.
            noise_mean: Mean of the noise distribution.
            noise_std: Standard deviation of the noise distribution.

        """
        return norm.rvs(loc=noise_mean, scale=noise_std, size=len(x))


class SinusoidalModelWithNoise(Model):
    """
    A sinusoidal model with Gaussian noise: y = A * sin(2 * pi * f * x +
    phi) + b + noise.
    """

    def evaluate(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Evaluate the sinusoidal model with noise.

        Args:
            x: Input data for the model.
            params: Model parameters (A, f, phi, b, noise_mean, noise_std).

        Returns:
            Model output evaluated at the given input data plus noise.

        """
        params = np.array(params)  # handle both np/torch uniformly
        if params.ndim == 1:  # single (A, f, phi, b, noise_mean, noise_std)
            a, f, phi, b, noise_mean, noise_std = params
        elif params.ndim == 2:  # (A, f, phi, b, noise_mean, noise_std) pairs
            a = params[:, 0][:, None]  # shape (N, 1)
            f = params[:, 1][:, None]  # shape (N, 1)
            phi = params[:, 2][:, None]  # shape (N, 1)
            b = params[:, 3][:, None]  # shape (N, 1)
            noise_mean = params[:, 4][:, None]  # shape (N, 1)
            noise_std = params[:, 5][:, None]
        else:
            err_str = f"Unexpected params shape: {params.shape}"
            raise ModelError(err_str)

        return (
            a * np.sin(2 * np.pi * f * x + phi)
            + b
            + self.noise(x, noise_mean, noise_std)
        )

    def noise(
        self, x: np.ndarray, noise_mean: np.ndarray, noise_std: np.ndarray
    ) -> np.ndarray:
        """
        Simulate the noise to apply to the model.

        Args:
            x: Input data for the model.
            noise_mean: Mean of the noise distribution.
            noise_std: Standard deviation of the noise distribution.

        """
        return norm.rvs(loc=noise_mean, scale=noise_std, size=len(x))
