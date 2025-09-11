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
    """
    Abstract base class for Simulation-Based Inference (SBI).

    This interface defines the required methods for creating inference
    objects, computing posteriors, sampling from distributions, and
    visualizing results.
    """

    @abstractmethod
    def create_inference_object(
        self,
        prior: torch.distributions.Distribution | None,
        estimation_type: Literal["NRE", "NLE", "NPE"] = "NRE",
    ) -> Inference:
        """
        Create an SBI inference object.

        Parameters
        ----------
        prior : torch.distributions.Distribution or None
            Prior distribution over model parameters.
        estimation_type : {"NRE", "NLE", "NPE"}, optional
            Inference method to use. Defaults to "NRE".

        Returns
        -------
        Inference
            The SBI inference object.

        """

    @abstractmethod
    def compute_distribution(
        self,
        inference_object: Inference,
        theta: Tensor,
        x: Tensor,
    ) -> Posterior:
        """
        Train inference object and build a posterior distribution.

        Parameters
        ----------
        inference_object : Inference
            An initialized SBI inference object.
        theta : torch.Tensor
            Parameters sampled from the prior.
        x : torch.Tensor
            Observed or simulated data corresponding to theta.

        Returns
        -------
        Posterior
            The trained posterior distribution.

        """

    @abstractmethod
    def sample_distribution(
        self,
        distribution: Posterior,
        theta: Tensor,
        x: Tensor,
    ) -> Tensor:
        """
        Sample from the posterior distribution.

        Parameters
        ----------
        distribution : Posterior
            Posterior distribution returned by `compute_distribution`.
        theta : torch.Tensor
            Parameter samples, used to define the sample shape.
        x : torch.Tensor
            Observed or simulated data (may be unused).

        Returns
        -------
        torch.Tensor
            Samples drawn from the posterior distribution.

        """

    @abstractmethod
    def plot_posterior(self, distribution_theta: Tensor) -> None:
        """
        Plot posterior samples.

        Parameters
        ----------
        distribution_theta : torch.Tensor
            Samples drawn from the posterior distribution.

        Returns
        -------
        None

        """


class SBIInterface(AbstractSBIInterface):
    """
    Concrete implementation of the SBI interface using the `sbi` library.

    Provides methods for creating inference objects, training them with
    simulations, drawing posterior samples, and visualizing results.
    """

    def create_inference_object(
        self,
        prior: torch.distributions.Distribution | None = None,
        estimation_type: Literal["NRE", "NLE", "NPE"] = "NRE",
    ) -> Inference:
        """
        Create an SBI inference object.

        Parameters
        ----------
        prior : torch.distributions.Distribution or None, optional
            Prior distribution over model parameters.
        estimation_type : {"NRE", "NLE", "NPE"}, optional
            Inference method to use. Defaults to "NRE".

        Returns
        -------
        Inference
            The SBI inference object.

        Raises
        ------
        ValueError
            If the estimation type is not supported.

        """
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
        """
        Train inference object and build posterior distribution.

        Parameters
        ----------
        inference_object : Inference
            An initialized SBI inference object.
        theta : torch.Tensor
            Parameters sampled from the prior.
        x : torch.Tensor
            Observed or simulated data corresponding to theta.

        Returns
        -------
        Posterior
            The trained posterior distribution.

        """
        _ = inference_object.append_simulations(theta, x).train()
        distribution: Posterior = inference_object.build_posterior()
        return distribution

    def sample_distribution(
        self,
        distribution: Posterior,
        theta: Tensor,
        _x: Tensor,  # unused
    ) -> Tensor:
        """
        Sample from the posterior distribution.

        Parameters
        ----------
        distribution : Posterior
            Posterior distribution returned by `compute_distribution`.
        theta : torch.Tensor
            Parameter samples, used to define the sample shape.
        _x : torch.Tensor
            Observed or simulated data (not used in this implementation).

        Returns
        -------
        torch.Tensor
            Samples drawn from the posterior distribution.

        """
        theta_shape = theta.shape  # e.g., torch.Size([2000, 3])
        distribution_theta: Tensor = distribution.sample(sample_shape=theta_shape)
        return distribution_theta

    def plot_posterior(self, distribution_theta: Tensor) -> None:
        """
        Plot posterior samples.

        Parameters
        ----------
        distribution_theta : torch.Tensor
            Samples drawn from the posterior distribution.

        Returns
        -------
        None

        """
        pairplot(distribution_theta)
