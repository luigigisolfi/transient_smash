from abc import ABC, abstractmethod
from scipy.stats import norm,uniform
import torch
from sbi.utils import process_prior, process_simulator
import numpy as np
from torch.distributions import Normal, Uniform, Exponential, Independent
from sbi.utils import MultipleIndependent
import torch

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


    def set_priors_emma(self, priors: dict) -> None:
        """Set prior distributions for model parameters.
        Args:
            priors: A dictionary containing prior distribution information
            for each parameter.
        Returns:
            None
        """
        priors_list = []
        for param in priors:
            dist = priors[param][0]
            if dist.lower()=='uniform':
                priors_list.append(
                    Uniform(torch.tensor([priors[param][1]]),
                            torch.tensor([priors[param][2]])))
            elif dist.lower()=='normal':
                priors_list.append(
                    Normal(torch.tensor([priors[param][1]]),
                           torch.tensor([priors[param][2]])))
            else:
                err_str = f"Invalid distribution selected: {dist}. " \
                          f"Please choose either 'uniform' or 'normal'."
                raise Exception(err_str)
        self.priors = MultipleIndependent(priors_list)

    def set_priors(self, priors: dict) -> None:
        """Set prior distributions for model parameters.

        Creates distributions compatible with SBI's MultipleIndependent.
        """
        import torch
        from torch.distributions import MultivariateNormal, Exponential
        from sbi.utils import MultipleIndependent, BoxUniform

        sampled_priors = []

        for param_name, spec in priors.items():
            print(f"Processing parameter: {param_name}")

            # Handle both single specs and lists of specs
            if isinstance(spec[0], str):
                specs_to_process = [spec]
            else:
                specs_to_process = spec

            for i, single_spec in enumerate(specs_to_process):
                dist_type = single_spec[0].lower()
                print(f"  Creating {dist_type} distribution {i+1}")

                if dist_type == "uniform":
                    n, low, high = single_spec[1], single_spec[2], single_spec[3]
                    print(f'    {n}D uniform prior, range [{low}, {high}]')

                    # BoxUniform handles multi-dimensional uniform correctly
                    low_tensor = torch.full((n,), low, dtype=torch.float32)
                    high_tensor = torch.full((n,), high, dtype=torch.float32)

                    dist = BoxUniform(low_tensor, high_tensor)
                    print(f"    Created BoxUniform with event_shape: {dist.event_shape}, batch_shape: {dist.batch_shape}")
                    sampled_priors.append(dist)

                elif dist_type == "normal":
                    n, mean, std = single_spec[1], single_spec[2], single_spec[3]
                    print(f'    {n}D normal prior, mean={mean}, std={std}')

                    # Always use MultivariateNormal, even for 1D case
                    mean_tensor = torch.full((n,), mean, dtype=torch.float32)
                    cov_matrix = torch.eye(n, dtype=torch.float32) * (std ** 2)
                    dist = MultivariateNormal(mean_tensor, cov_matrix)

                    print(f"    Created MultivariateNormal with event_shape: {dist.event_shape}, batch_shape: {dist.batch_shape}")
                    sampled_priors.append(dist)

                elif dist_type == "exponential":
                    rate = single_spec[1]
                    print(f'Exponential prior, rate={rate}')

                    if rate <= 0:
                        raise ValueError("Exponential rate must be > 0")

                    # Use Exponential with tensor rate for any dimensionality
                    rate_tensor = torch.full((1,), rate, dtype=torch.float32)
                    dist = Exponential(rate_tensor)

                    print(f"    Created Exponential with event_shape: {dist.event_shape}, batch_shape: {dist.batch_shape}")
                    sampled_priors.append(dist)

                else:
                    print(f"ERROR: Invalid distribution type: {dist_type}")
                    raise ValueError(f"Invalid distribution type: {dist_type}")

        print(f"Created {len(sampled_priors)} prior distributions")

        # Debug: Check all batch shapes before creating MultipleIndependent
        for i, dist in enumerate(sampled_priors):
            print(f"Distribution {i}: batch_shape={dist.batch_shape}, event_shape={dist.event_shape}")
            if dist.batch_shape not in (torch.Size([]), torch.Size([1])):
                raise ValueError(f"Distribution {i} has invalid batch_shape: {dist.batch_shape}. Must be () or (1,)")

        total_dims = sum(dist.event_shape.numel() for dist in sampled_priors)
        print(f"Total parameter dimensions: {total_dims}")

        self.priors = MultipleIndependent(sampled_priors)
        print(f"Successfully created MultipleIndependent with {len(self.priors.dists)} distributions")


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
    def evaluate(self, x, a, b):
        """A simple linear model: y = a * x + b."""
        return a * x + b
    
class SimpleModel_PlusNoise(Model):

    def evaluate(self, x, a, b, *args):
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