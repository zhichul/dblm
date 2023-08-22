from __future__ import annotations
import abc
from typing import Sequence


class Distribution(abc.ABC):

    @abc.abstractmethod
    def likelihood_function(self, assignment):
        raise NotImplementedError()

    @abc.abstractmethod
    def log_likelihood_function(self, assignment):
        raise NotImplementedError()

class LocallyNormalizedDistribution(Distribution):
    ...

class GloballyNormalizedDistribution(Distribution):

    @abc.abstractmethod
    def unnormalized_likelihood_function(self, assignment):
        raise NotImplementedError()

    @abc.abstractmethod
    def partition_function(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def log_unnormalized_likelihood_function(self, assignment):
        raise NotImplementedError()

    @abc.abstractmethod
    def log_partition_function(self):
        raise NotImplementedError()

    def likelihood_function(self, assignment):
        return self.unnormalized_likelihood_function(assignment) / self.partition_function()

    def log_likelihood_function(self, assignment):
        return self.log_unnormalized_likelihood_function(assignment) - self.log_partition_function()

class IncompleteLikelihoodMixin(abc.ABC):

    @abc.abstractmethod
    def incomplete_likelihood_function(self, assignment: Sequence[tuple[int, int]]):
        raise NotImplementedError()

    @abc.abstractmethod
    def incomplete_log_likelihood_function(self, assignment: Sequence[tuple[int, int]]):
        raise NotImplementedError()
