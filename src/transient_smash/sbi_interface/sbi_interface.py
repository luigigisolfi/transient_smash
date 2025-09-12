"""Simulation-Based Inference (SBI) interface for transient_smash."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from sbi.analysis import pairplot
from sbi.inference import NLE, NPE
from torch import Tensor


class SBIInterface(ABC):
    """
    Abstract base class for Simulation-Based Inference (SBI).

    This interface defines the required methods for creating inference
    objects, computing posteriors, sampling from distributions, and
    visualizing results.
    """

    @abstractmethod
    def create_inference_object(
        self,
        prior: list(torch.distributions.Distribution) | None = None,
    ) -> Inference:
        """
        Create an SBI inference object.

        Parameters
        ----------
        prior : torch.distributions.Distribution or None
            Prior distribution over model parameters.

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


class NLESBI(SBIInterface):
    """
    Concrete SBI interface using the sbi library.

    Provides implementations for creating inference objects, training,
    sampling from posteriors, and visualizing posterior samples.
    """

    def create_inference_object(
        self, prior: list(torch.distributions.Distribution) | None = None
    ) -> Inference:
        """
        Create an SBI inference object.

        Parameters
        ----------
        prior : torch.distributions.Distribution or None, optional
            Prior distribution over parameters. Defaults to None.
        estimation_type : {"NRE", "NLE", "NPE"}, optional
            Inference algorithm to use. Defaults to "NRE".

        Returns
        -------
        Inference
            An SBI inference object (NRE, NLE, or NPE).

        Raises
        ------
        ValueError
            If the specified estimation_type is not supported.

        """
        inference_object = NLE(prior)

        return inference_object

    def compute_distribution(
        self,
        inference_object: Inference,
        theta: Tensor,
        x: Tensor,
    ) -> Posterior:
        """
        Train the inference object and build a posterior distribution.

        Parameters
        ----------
        inference_object : Inference
            An initialized SBI inference object.
        theta : torch.Tensor
            Parameters sampled from the prior.
        x : torch.Tensor
            Simulated or observed data corresponding to `theta`.

        Returns
        -------
        Posterior
            The posterior distribution trained on the simulations.

        """
        estimator = inference_object.append_simulations(theta, x).train()
        distribution = inference_object.build_posterior()

        return estimator, distribution

    def sample_distribution(
        self,
        distribution: Posterior,
        theta: Tensor,
        _x: Tensor,
    ) -> Tensor:
        """
        Sample from the posterior distribution.

        Parameters
        ----------
        distribution : Posterior
            Posterior distribution returned by `compute_distribution`.
        theta : torch.Tensor
            Parameters sampled from the prior. Used to determine sample shape.
        _x : torch.Tensor
            Unused. Kept for interface consistency.

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
        Visualize posterior samples with a pairplot.

        Parameters
        ----------
        distribution_theta : torch.Tensor
            Samples drawn from the posterior distribution.

        Returns
        -------
        None

        """
        pairplot(distribution_theta)


class NPESBI(SBIInterface):
    """
    Concrete SBI interface using the sbi library.

    Provides implementations for creating inference objects, training,
    sampling from posteriors, and visualizing posterior samples.
    """

    def create_inference_object(
        self, prior: torch.distributions.Distribution | None = None
    ) -> Inference:
        """
        Create an SBI inference object.

        Parameters
        ----------
        prior : torch.distributions.Distribution or None, optional
            Prior distribution over parameters. Defaults to None.
        estimation_type : {"NRE", "NLE", "NPE"}, optional
            Inference algorithm to use. Defaults to "NRE".

        Returns
        -------
        Inference
            An SBI inference object (NRE, NLE, or NPE).

        Raises
        ------
        ValueError
            If the specified estimation_type is not supported.

        """
        inference_object = NPE(prior)

        return inference_object

    def compute_distribution(
        self,
        inference_object: Inference,
        theta: Tensor,
        x: Tensor,
    ) -> Posterior:
        """
        Train the inference object and build a posterior distribution.

        Parameters
        ----------
        inference_object : Inference
            An initialized SBI inference object.
        theta : torch.Tensor
            Parameters sampled from the prior.
        x : torch.Tensor
            Simulated or observed data corresponding to `theta`.

        Returns
        -------
        Posterior
            The posterior distribution trained on the simulations.

        """
        _ = inference_object.append_simulations(theta, x).train()
        distribution: Posterior = inference_object.build_posterior()
        return distribution

    def sample_distribution(
        self,
        distribution: Posterior,
        theta: Tensor,
        _x: Tensor,
    ) -> Tensor:
        """
        Sample from the posterior distribution.

        Parameters
        ----------
        distribution : Posterior
            Posterior distribution returned by `compute_distribution`.
        theta : torch.Tensor
            Parameters sampled from the prior. Used to determine sample shape.
        _x : torch.Tensor
            Unused. Kept for interface consistency.

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
        Visualize posterior samples with a pairplot.

        Parameters
        ----------
        distribution_theta : torch.Tensor
            Samples drawn from the posterior distribution.

        Returns
        -------
        None

        """
        pairplot(distribution_theta)
