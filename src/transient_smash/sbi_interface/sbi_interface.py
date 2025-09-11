"""Simulation-Based Inference (SBI) interface for transient_smash."""

from abc import ABC, abstractmethod
from typing import Literal

import torch
from sbi.analysis import pairplot
from sbi.inference import NLE, NPE, NRE
from sbi.inference.base import Inference
from sbi.inference.posteriors import Posterior
from torch import Tensor


class AbstractSBIInterface(ABC):
    """Abstract base class for Simulation-Based Inference (SBI)."""

    @abstractmethod
    def create_inference_object(
        self,
        prior: torch.distributions.Distribution | None,
        estimation_type: Literal["NRE", "NLE", "NPE"] = "NRE",
    ) -> Inference:
        """Create an SBI inference object based on the specified estimation type."""

    @abstractmethod
    def compute_distribution(
        self,
        inference_object: Inference,
        theta: Tensor,
        x: Tensor,
    ) -> Posterior:
        """Train inference object and build posterior distribution."""

    @abstractmethod
    def sample_distribution(
        self,
        distribution: Posterior,
        theta: Tensor,
        x: Tensor,
    ) -> Tensor:
        """Draw samples from the posterior distribution."""

    @abstractmethod
    def plot_posterior(self, distribution_theta: Tensor) -> None:
        """Plot the posterior samples using pairwise plots."""


class SBIInterface(AbstractSBIInterface):
    """Concrete SBI interface using the sbi library."""

    def create_inference_object(
        self,
        prior: torch.distributions.Distribution | None = None,
        estimation_type: Literal["NRE", "NLE", "NPE"] = "NRE",
    ) -> Inference:
        """Create an SBI inference object."""
        if estimation_type == "NRE":
            inference_object: Inference = NRE(prior)
        elif estimation_type == "NLE":
            inference_object = NLE(prior)
        elif estimation_type == "NPE":
            inference_object = NPE(prior)
        else:
            error_msg = (
                "Model type not supported. "
                "Please select a supported model (NRE, NPE, NLE). "
                "Aborting..."
            )
            raise ValueError(error_msg)
        return inference_object

    def compute_distribution(
        self,
        inference_object: Inference,
        theta: Tensor,
        x: Tensor,
    ) -> Posterior:
        """Train inference object and build posterior distribution."""
        _ = inference_object.append_simulations(theta, x).train()
        distribution: Posterior = inference_object.build_posterior()
        return distribution

    def sample_distribution(
        self,
        distribution: Posterior,
        theta: Tensor,
        _x: Tensor,  # unused
    ) -> Tensor:
        """Sample from the posterior distribution."""
        theta_shape = theta.shape  # e.g., torch.Size([2000, 3])
        distribution_theta: Tensor = distribution.sample(sample_shape=theta_shape)
        return distribution_theta

    def plot_posterior(self, distribution_theta: Tensor) -> None:
        """Visualize the posterior samples using a pairplot."""
        pairplot(distribution_theta)
