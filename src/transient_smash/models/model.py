from abc import ABC, abstractmethod
from scipy.stats import norm


class Model(ABC):

    @abstractmethod
    def simulator(self, x, *args):
        """Simulate the model at the given input x.
        
        Args:
            x: Input data for the model.
            *args: Additional arguments for the simulation containing model parameters.

        Returns:
            Simulated output of the model.
            
        """
        ...

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