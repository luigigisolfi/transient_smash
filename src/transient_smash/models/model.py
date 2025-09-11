from abc import ABC, abstractmethod
from scipy.stats import norm
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
        self.priors = priors
        return
    
    def get_sbi_priors(self):
        """Get prior distributions for model parameters.
        
        Returns:
            PyTorch distribution-like priors for each parameter.
            
        """
        return process_prior(self.priors)


class SimpleModel(Model):
    def simulator(self, x, a, b):
        """A simple linear model: y = a * x + b."""
        return a * x + b
    
class SimpleModel_PlusNoise(Model):

    def simulator(self, x, a, b, *args):
        """A simple linear model: y = a * x + b."""
        return a * x + b + self.noise(x, *args)

    @abstractmethod
    def noise(self, x, *args):
        """Simulate the noise to apply to the model.
        
        Args:
            x: Input data for the model.
            *args: Additional arguments for the noise distribution.

        Returns:
            Simulated noise on the model.
            
        """
        ...

class SimpleModel_PlusSimpleNoise(SimpleModel_PlusNoise):

    def noise(self,x,sigma,seed=42):
        """Add Gaussian noise to a dataset with defined variance, sigma"""
        return norm.rvs(loc=0,scale=sigma,size=len(x),random_state=seed)