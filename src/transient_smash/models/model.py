from abc import ABC, abstractmethod
from scipy.stats import norm,uniform
import torch
from torch.distributions import Normal,Uniform, Independent
from sbi.utils import process_prior, process_simulator
import numpy as np


class Model(ABC):

    @abstractmethod
    def evaluate(self, x, *args):
        """Evaluate the model given input data and model parameters.
        
        Args:
            x: Input data for the model (i.e. time points at which to simulate).
            *args: Additional arguments for the evaluation containing model parameters.

        Returns:
            Output of the model.
            
        """
        ...


    def simulator(self, *args: np.ndarray) -> np.ndarray:
        """Generate simulated data given the pre-set input data and model parameters.
        
        Args:
            *args: Additional arguments for the simulation containing model parameters.

        Returns:
            Simulated output of the model.
            
        """
        # Check if input data has been set
        if not hasattr(self, 'x'):
            raise ValueError("Input data 'x' has not been set. Please use the 'set_input_data' method to set it before calling the simulator.")
        return self.evaluate(self.x, *args)

    def set_input_data(self, x: np.ndarray):
        """Set input data for the model.
        
        Args:
            x: Input data for the model (i.e. time points at which to simulate).

        Returns:
            None
            
        """
        self.x = x
        return

    def get_sbi_simulator(self) -> callable:
        """Get a simulator function compatible with the sbi package.
        """
        priors = self.get_sbi_priors()
        is_numpy_simulator = True
        sbi_simulator = process_simulator(self.simulator, priors, is_numpy_simulator)
        return sbi_simulator


    def set_priors(self, priors):
        """Set prior distributions for model parameters.
        
        Args:
            prior_info: A dictionary containing prior distribution information for each parameter.

        Returns:
            None
            
        """
        num_dims = len(priors.keys())
        sampled_priors = []
        for param in priors.keys():
            dist = priors[param][0]
            if dist.lower()=='uniform':
                
                sampled_priors.append(
                        Uniform(torch.tensor(priors[param][1]),torch.tensor(priors[param][2])))
            elif dist.lower()=='normal':
                sampled_priors.append(
                        Normal(torch.tensor(priors[param][1]),torch.tensor(priors[param][2])))
            else:
                raise ValueError(f"Invalid distribution selected: {dist}")
        self.priors = sampled_priors
    
    def get_sbi_priors(self):
        """Get prior distributions for model parameters.
        
        Returns:
            PyTorch distribution-like priors for each parameter.
            
        """
        if not hasattr(self,'priors'):
            raise ValueError("Priors have not been set. Please use the 'set_priors' method to set them before calling this method.")
        # return process_prior(self.priors)
        return self.priors
            


class SimpleModel(Model):
    def evaluate(self, x, params):
        """
        x: np.ndarray of shape (M,)
        params: np.ndarray of shape (N, 2)
        Returns: shape (N, M)
        """
        params = np.array(params)  # if you want to handle both np/torch uniformly
    
        if params.ndim == 1:  # single (a, b)
            a, b = params
            return a * x + b
        
        elif params.ndim == 2:  # multiple (a, b) pairs
            a = params[:, 0][:, None]  # shape (N, 1)
            b = params[:, 1][:, None]  # shape (N, 1)
            return a * x[None, :] + b
        
        else:
            raise ValueError(f"Unexpected params shape: {params.shape}")
    
class SimpleModelWithNoise(Model):

    def evaluate(self, x, params):
        """A simple linear model: y = a * x + b."""
        params = np.array(params)  # if you want to handle both np/torch uniformly
        if params.ndim == 1:  # single (a, b, noise_mean, noise_std)
            a, b, noise_mean, noise_std = params
        elif params.ndim == 2:  # multiple (a, b, noise_mean, noise_std) pairs
            a = params[:, 0][:, None]  # shape (N, 1)
            b = params[:, 1][:, None]  # shape (N, 1)
            noise_mean = params[:, 2][:, None]  # shape (N, 1)
            noise_std = params[:, 3][:, None]
        else:
            raise ValueError(f"Unexpected params shape: {params.shape}")

        return a * x + b + self.noise(x, noise_mean, noise_std)

    def noise(self, x, noise_mean, noise_std):
        """Simulate the noise to apply to the model.
        
        Args:
            x: Input data for the model.
            noise_mean: Mean of the noise distribution.
            noise_std: Standard deviation of the noise distribution.
        """
        return norm.rvs(loc=noise_mean, scale=noise_std, size=len(x))

class SinusoidalModelWithNoise(Model):
    
    def evaluate(self, x, params):
        """A sinusoidal model: y = A * sin(2 * pi * f * x + phi) + b."""
        params = np.array(params)  # if you want to handle both np/torch uniformly
        if params.ndim == 1:  # single (A, f, phi, b, noise_mean, noise_std)
            A, f, phi, b, noise_mean, noise_std = params
        elif params.ndim == 2:  # multiple (A, f, phi, b, noise_mean, noise_std) pairs
            A = params[:, 0][:, None]  # shape (N, 1)
            f = params[:, 1][:, None]  # shape (N, 1)
            phi = params[:, 2][:, None]  # shape (N, 1)
            b = params[:, 3][:, None]  # shape (N, 1)
            noise_mean = params[:, 4][:, None]  # shape (N, 1)
            noise_std = params[:, 5][:, None]
        else:
            raise ValueError(f"Unexpected params shape: {params.shape}")

        return A * np.sin(2 * np.pi * f * x + phi) + b + self.noise(x, noise_mean, noise_std)

    def noise(self, x, noise_mean, noise_std):
        """Simulate the noise to apply to the model.
        
        Args:
            x: Input data for the model.
            noise_mean: Mean of the noise distribution.
            noise_std: Standard deviation of the noise distribution.
        """
        return norm.rvs(loc=noise_mean, scale=noise_std, size=len(x))