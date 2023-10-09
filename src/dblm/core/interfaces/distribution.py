from __future__ import annotations
from typing import Sequence
import abc
import torch


class Distribution(abc.ABC):

    @abc.abstractmethod
    def probability(self, assignment: tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def log_probability(self, assignment: tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def condition_on(self, observation: dict[int, int] | dict[int, torch.Tensor]) -> Distribution:
        raise NotImplementedError()

class ConditionalDistribution(Distribution):

    @abc.abstractmethod
    def parent_indices(self) -> tuple[int,...]:
        raise NotImplementedError()

    @abc.abstractmethod
    def child_indices(self) -> tuple[int,...]:
        raise NotImplementedError()

class UnnormalizedDistribution(Distribution):

    @abc.abstractmethod
    def unnormalized_probability(self, assignment: tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def normalization_constant(self) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def energy(self, assignment: tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def log_normalization_constant(self) -> torch.Tensor:
        raise NotImplementedError()

    def probability(self, assignment: tuple[int, ...]) -> torch.Tensor:
        return self.unnormalized_probability(assignment) / self.normalization_constant()

    def log_probability(self, assignment: tuple[int, ...]) -> torch.Tensor:
        return self.energy(assignment) - self.log_normalization_constant()

class MarginalMixin(abc.ABC):

    @abc.abstractmethod
    def marginal_probability(self, assignment: Sequence[tuple[int, int]]) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def log_marginal_probability(self, assignment: Sequence[tuple[int, int]]) -> torch.Tensor:
        raise NotImplementedError()

class NormalizedDistribution(UnnormalizedDistribution):

    def normalization_constant(self):
        return 1

    def log_normalization_constant(self):
        return 0
