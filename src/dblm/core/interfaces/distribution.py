from __future__ import annotations
from typing import Sequence
import abc
import torch


class Distribution(abc.ABC):

    @abc.abstractmethod
    def likelihood_function(self, assignment: tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def log_likelihood_function(self, assignment: tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def condition_on(self, observation: dict[int, int]) -> Distribution:
        raise NotImplementedError()

class LocallyNormalizedDistribution(Distribution):

    @abc.abstractmethod
    def parent_indices(self) -> tuple[int,...]:
        raise NotImplementedError()

    @abc.abstractmethod
    def child_indices(self) -> tuple[int,...]:
        raise NotImplementedError()


class GloballyNormalizedDistribution(Distribution):

    @abc.abstractmethod
    def unnormalized_likelihood_function(self, assignment: tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def partition_function(self) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def log_unnormalized_likelihood_function(self, assignment: tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def log_partition_function(self) -> torch.Tensor:
        raise NotImplementedError()

    def likelihood_function(self, assignment: tuple[int, ...]) -> torch.Tensor:
        return self.unnormalized_likelihood_function(assignment) / self.partition_function()

    def log_likelihood_function(self, assignment: tuple[int, ...]) -> torch.Tensor:
        return self.log_unnormalized_likelihood_function(assignment) - self.log_partition_function()

class IncompleteLikelihoodMixin(abc.ABC):

    @abc.abstractmethod
    def incomplete_likelihood_function(self, assignment: Sequence[tuple[int, int]]) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def incomplete_log_likelihood_function(self, assignment: Sequence[tuple[int, int]]) -> torch.Tensor:
        raise NotImplementedError()
