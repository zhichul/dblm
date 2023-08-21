from __future__ import annotations
import abc
import torch
from dblm.core.interfaces import distribution

"""
Classes that hold the structure and parameter of PGMs.
They should really be treated as immutables.
Though the implementation doesn't prevent mutation of their fields.
"""
class MultivariateFunction:

    def __init__(self) -> None:
        self._nvars = 0
        self._nvals = []

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

    @abc.abstractmethod
    def fix_variables(self, observation: dict[int, int]) -> ProbabilisticGraphicalModel:
        raise NotImplementedError()

class FactorGraphModel(distribution.GloballyNormalizedDistribution, ProbabilisticGraphicalModel):

    def to_factor_graph_model(self) -> FactorGraphModel:
        return self

    @abc.abstractmethod
    def factor_variables(self) -> list[list[int]]:
        ...

    @abc.abstractmethod
    def factor_functions(self) -> list[distribution.Distribution]:
        ...

    @abc.abstractmethod
    def fix_variables(self, observation: dict[int, int]) -> FactorGraphModel:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_conditional_factors(self, observation: dict[int, int]) -> tuple[list[tuple[int]], list[distribution.LocallyNormalizedDistribution | distribution.GloballyNormalizedDistribution | PotentialTable]]:
        raise NotImplementedError()

class BayesianNetwork(distribution.LocallyNormalizedDistribution, ProbabilisticGraphicalModel):

    @abc.abstractmethod
    def local_distributions(self) -> list[distribution.Distribution]:
        ...

    @abc.abstractmethod
    def topological_order(self) -> list[int]:
        ...

    @abc.abstractmethod
    def local_variables(self) -> list[tuple[int]]:
        ...

    @abc.abstractmethod
    def local_parents(self) -> list[tuple[int]]:
        ...

    @abc.abstractmethod
    def local_children(self) -> list[tuple[int]]:
        ...

    @abc.abstractmethod
    def fix_variables(self, observation: dict[int, int]) -> BayesianNetwork:
        raise NotImplementedError()
class MarkovRandomField(distribution.GloballyNormalizedDistribution, ProbabilisticGraphicalModel):

    @abc.abstractmethod
    def local_potentials(self) -> list[distribution.Distribution]:
        ...

    @abc.abstractmethod
    def local_variables(self) -> list[distribution.Distribution]:
        ...

    @abc.abstractmethod
    def fix_variables(self, observation: dict[int, int]) -> MarkovRandomField:
        raise NotImplementedError()

class PotentialTable(MultivariateFunction, abc.ABC):

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
    def fix_variables(self, observation: dict[int, int]) -> PotentialTable:
        raise NotImplementedError()

    @abc.abstractmethod
    def marginalize_over(self, variables) -> PotentialTable:
        raise NotImplementedError()

    @abc.abstractmethod
    def renormalize(self) -> ProbabilityTable:
        raise NotImplementedError

class ProbabilityTable(distribution.LocallyNormalizedDistribution, PotentialTable, ProbabilisticGraphicalModel):

    def graph(self) -> graph.Graph:
        raise NotImplementedError()

    @abc.abstractmethod
    def probability_table(self) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def log_probability_table(self) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def to_bayesian_network(self) -> BayesianNetwork:
        ...

    def renormalize(self) -> ProbabilityTable:
        return self
