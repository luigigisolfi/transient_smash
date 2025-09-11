from transient_smash.sbi_interface.sbi_interface import NLESBI, NPESBI
import unittest
from sbi.inference import NLE, NPE, NRE
import torch
from sbi.utils import BoxUniform

class TestNLESBI(unittest.TestCase):
    """
    Concrete implementation of the SBI interface using the `sbi` library.

    Provides methods for creating inference objects, training them with
    simulations, drawing posterior samples, and visualizing results.
    """

    def test_create_inference_object(

            self,
            prior  = None # example prior
    ):

        nre_sbi = NLESBI()
        inference_object = nre_sbi.create_inference_object(prior)

        if prior is not None:
            self.assertIsInstance(prior, torch.distributions.Distribution)

        self.assertIsInstance(inference_object, NLE)

    def test_compute_distribution(

            self,
            inference_object = None,
            theta = torch.randn(10, 3),
            x = torch.randn(10, 3),
    ):

        nre_sbi = NLESBI()

        # test
        _ = torch.manual_seed(42)
        from sbi.utils import BoxUniform

        lower_bound = torch.as_tensor([0.05, 0.01, 0.005, 0.005])
        upper_bound = torch.as_tensor([0.15, 0.03, 0.03, 0.15])
        prior = BoxUniform(low=lower_bound, high=upper_bound)

        # test
        inference_object = nre_sbi.create_inference_object(prior)

        estimator, distribution = nre_sbi.compute_distribution(inference_object, theta, x)

        print(distribution)


if __name__=='__main__':
    unittest.main()