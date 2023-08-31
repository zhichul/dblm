from __future__ import annotations
import abc
from typing import Sequence
import torch
from dblm.core.interfaces import distribution

"""
Classes that hold the structure and parameter of PGMs.
They should really be treated as immutables.
Though the implementation doesn't prevent mutation of their fields.
"""
class MultivariateFunction:

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._nvars: int = 0
        self._nvals: list = []

    @property
    def nvals(self) -> list[int]:
        return self._nvals

    @property
    def nvars(self) -> int:
        return self._nvars

class ProbabilisticGraphicalModel(MultivariateFunction, abc.ABC):

    @abc.abstractmethod
    def graph(self) -> graph.Graph:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_factor_graph_model(self) -> FactorGraphModel:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_probability_table(self) -> ProbabilityTable:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_potential_table(self) -> PotentialTable:
        raise NotImplementedError()

class FactorGraphModel(ProbabilisticGraphicalModel, distribution.UnnormalizedDistribution):

    def to_factor_graph_model(self) -> FactorGraphModel:
        return self

    @abc.abstractmethod
    def factor_variables(self) -> list[tuple[int,...]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def factor_functions(self) -> list[PotentialTable]:
        raise NotImplementedError()

    @abc.abstractmethod
    def conditional_factor_variables(self, observation: dict[int, int] | dict[int, torch.Tensor]) -> list[int]:
        raise NotImplementedError()

    @abc.abstractmethod
    def conditional_factor_functions(self, observation: dict[int, int] | dict[int, torch.Tensor]) -> list[PotentialTable]:
        raise NotImplementedError()


class BayesianNetwork(ProbabilisticGraphicalModel, distribution.ConditionalDistribution, distribution.NormalizedDistribution):

    @abc.abstractmethod
    def local_distributions(self) -> list[distribution.Distribution]:
        raise NotImplementedError()

    @abc.abstractmethod
    def topological_order(self) -> list[int]:
        raise NotImplementedError()

    @abc.abstractmethod
    def local_variables(self) -> list[tuple[int,...]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def local_parents(self) -> list[tuple[int,...]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def local_children(self) -> list[tuple[int,...]]:
        raise NotImplementedError()

    @property
    def parent_indices(self):
        return tuple()

    @property
    def child_indices(self):
        return tuple(range(self.nvars))

class MarkovRandomField(ProbabilisticGraphicalModel, distribution.UnnormalizedDistribution):

    @abc.abstractmethod
    def local_potentials(self) -> list[PotentialTable]:
        ...

    @abc.abstractmethod
    def local_variables(self) -> list[tuple[int,...]]:
        ...

class Batchable:

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._batch_dims: int = 0
        self._batch_size: tuple[int,...] = tuple()

    @property
    def batch_dims(self) -> int:
        return self._batch_dims # type:ignore

    @property
    def batch_size(self) -> tuple[int, ...]:
        return self._batch_size # type:ignore

    def expand_batch_dimensions_meta_(self, batch_sizes: tuple[int, ...]) -> None:
        self._batch_size = batch_sizes + self._batch_size
        self._batch_dims = len(batch_sizes) + self._batch_dims

    @abc.abstractmethod
    def expand_batch_dimensions_(self, batch_sizes: tuple[int, ...]) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def expand_batch_dimensions(self, batch_sizes: tuple[int, ...]) -> Batchable:
        raise NotImplementedError()
class PotentialTable(Batchable, MultivariateFunction, abc.ABC):

    @abc.abstractmethod
    def potential_table(self) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def log_potential_table(self) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def potential_value(self, assignment) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def log_potential_value(self, assignment) -> torch.Tensor:
        raise NotImplementedError()

    def to_potential_table(self) -> PotentialTable:
        return self

    @abc.abstractmethod
    def condition_on(self, observation: dict[int, int] | dict[int, torch.Tensor]) -> PotentialTable:
        raise NotImplementedError()

    @abc.abstractmethod
    def marginalize_over(self, variables: Sequence[int]) -> PotentialTable:
        raise NotImplementedError()

    @abc.abstractmethod
    def renormalize(self) -> ProbabilityTable:
        raise NotImplementedError

class ProbabilityTable(PotentialTable, ProbabilisticGraphicalModel, distribution.ConditionalDistribution, distribution.NormalizedDistribution):

    def graph(self) -> graph.Graph:
        raise NotImplementedError()

    @abc.abstractmethod
    def probability_table(self) -> torch.Tensor:
        raise ValueError

    @abc.abstractmethod
    def log_probability_table(self) -> torch.Tensor:
        raise ValueError

    @abc.abstractmethod
    def to_bayesian_network(self) -> BayesianNetwork:
        raise ValueError

    def renormalize(self) -> ProbabilityTable:
        return self

    def to_probability_table(self) -> ProbabilityTable:
        return self
