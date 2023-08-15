from __future__ import annotations
import abc
import torch
from dblm.core.interfaces import distribution

"""
Classes that hold the structure and parameter of PGMs.
They should really be treated as immutables.
Though the implementation doesn't prevent mutation of their fields.
"""
class ProbabilisticGraphicalModel(abc.ABC):

    def __init__(self) -> None:
        self._nvars = 0
        self._nvals = []

    def nvals(self) -> list[int]:
        return self._nvals

    def nvars(self) -> int:
        return self._nvars

    @abc.abstractmethod
    def graph(self) -> graph.Graph:
        ...

    @abc.abstractmethod
    def to_factor_graph_model(self) -> FactorGraphModel:
        ...

    @abc.abstractmethod
    def to_probability_table(self) -> ProbabilityTable:
        ...

class FactorGraphModel(distribution.GloballyNormalizedDistribution, ProbabilisticGraphicalModel):

    def to_factor_graph_model(self) -> FactorGraphModel:
        return self

    @abc.abstractmethod
    def factor_variables(self) -> list[list[int]]:
        ...

    @abc.abstractmethod
    def factor_functions(self) -> list[distribution.Distribution]:
        ...


class BayesianNetwork(distribution.LocallyNormalizedDistribution, ProbabilisticGraphicalModel):

    @abc.abstractmethod
    def local_distributions(self) -> list[distribution.Distribution]:
        ...

    @abc.abstractmethod
    def topological_order(self) -> list[int]:
        ...

    @abc.abstractmethod
    def local_variables(self) -> list[distribution.Distribution]:
        ...


class MarkovRandomField(distribution.GloballyNormalizedDistribution, ProbabilisticGraphicalModel):

    @abc.abstractmethod
    def local_potentials(self) -> list[distribution.Distribution]:
        ...

    @abc.abstractmethod
    def local_variables(self) -> list[distribution.Distribution]:
        ...


class ProbabilityTable(distribution.LocallyNormalizedDistribution, ProbabilisticGraphicalModel):

    def graph(self) -> graph.Graph:
        raise NotImplementedError()

    @abc.abstractmethod
    def probability_table(self) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def log_probability_table(self) -> torch.Tensor:
        ...

class PotentialTable(abc.ABC):

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
