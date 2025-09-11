from abc import ABC, abstractmethod
from scipy.stats import norm


class Model(ABC):

    @abstractmethod
    def simulator(self, x, *args):
        """Generate simulated data given input data and model parameters.
        
        Args:
            x: Input data for the model (i.e. time points at which to simulate).
            *args: Additional arguments for the simulation containing model parameters.

        Returns:
            Simulated output of the model.
            
        """
        ...

    def get_sbi_simulator(self):
        """Get a simulator function compatible with the sbi package.
        """
        pass

    def set_priors(self, prior_info):
        """Set prior distributions for model parameters.
        
        Args:
            prior_info: A dictionary containing prior distribution information for each parameter.

        Returns:
            None
            
        """
        self.prior_info = prior_info
        return
    
    def get_sbi_priors(self):
        """Get prior distributions for model parameters.
        
        Returns:
            PyTorch distribution-like priors for each parameter.
            
        """
        pass


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