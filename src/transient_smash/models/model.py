from abc import ABC, abstractmethod


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
