from __future__ import annotations
from dblm.core.modeling import distribution


class ProbabilisticGraphicalModel(distribution.Distribution):

    def graph(self):
        ...

    def to_factor_graph_model(self) -> FactorGraphModel:
        ...

    def to_probability_table(self) -> ProbabilityTable:
        ...


class FactorGraphModel(ProbabilisticGraphicalModel):
    def to_factor_graph_model(self):
        return self

    def graph(self):
        ...

    def factors(self):
        ...


class BayesianNetwork(ProbabilisticGraphicalModel):
    def to_factor_graph_model(self) -> FactorGraphModel:
        ...

    def graph(self):
        ...

    def local_distributions(self):
        ...

    def topological_order(self):
        ...


class MarkovRandomField(ProbabilisticGraphicalModel):
    def to_factor_graph_model(self) -> FactorGraphModel:
        ...

    def graph(self):
        ...

    def local_potentials(self):
        ...


class ProbabilityTable(ProbabilisticGraphicalModel):
    def to_factor_graph_model(self) -> FactorGraphModel:
        ...

    def graph(self):
        ...

    def probability_table(self):
        ...
    
    def log_probability_table(self):
        ...

class LocalPotentialFunction:

    def potential_table(self):
        ...
    
    def log_potential_table(self):
        ...
