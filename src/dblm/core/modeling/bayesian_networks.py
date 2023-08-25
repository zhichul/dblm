
from __future__ import annotations

import torch
from dblm.core.interfaces import pgm
from dblm.core.modeling import factor_graphs, probability_tables, utils

import torch.nn as nn


class BayesianNetwork(nn.Module, pgm.BayesianNetwork):

    def __init__(self, nvars: int,
                 nvals: list[int],
                 local_variables: list[tuple[int,...]],
                 local_parents: list[tuple[int,...]],
                 local_children: list[tuple[int,...]],
                 local_functions: list[pgm.ProbabilityTable],
                 topological_order: list[int]):
        super().__init__()
        self._factor_variables = local_variables
        self._factor_functions = nn.ModuleList(local_functions) # type:ignore
        self._parents = local_parents
        self._children = local_children
        self._nvars = nvars
        self._nvals = nvals
        self._topological_order = topological_order

    # BayesianNetwork
    def topological_order(self) -> list[int]:
        return self._topological_order

    def local_variables(self) -> list[tuple[int,...]]:
        return self._factor_variables

    def local_distributions(self) -> list[pgm.ProbabilityTable]:
        return self._factor_functions # type:ignore

    def local_parents(self):
        return self._parents

    def local_children(self):
        return self._children

    # ProbabilisticGraphicalModel
    def graph(self):
        # TODO: implement a directed graph in dblm.core.graph and use it to represent a bayes net
        raise NotImplementedError()

    def to_factor_graph_model(self) -> pgm.FactorGraphModel:
        return factor_graphs.FactorGraph(self.nvars, self.nvals, self._factor_variables, self._factor_functions) # type:ignore

    def to_probability_table(self) -> pgm.ProbabilityTable:
        return probability_tables.LogLinearProbabilityTable.joint_from_factors(self.nvars, self.nvals, self._factor_variables, self._factor_functions) # type:ignore

    def to_potential_table(self) -> pgm.PotentialTable:
        return probability_tables.LogLinearPotentialTable.joint_from_factors(self.nvars, self.nvals, self._factor_variables, self._factor_functions) # type:ignore

    def condition_on(self, observation):
        raise NotImplementedError()

    # LocallyNormalizedDistribution
    def parent_indices(self) -> tuple[int, ...]:
        children_indices = set(self.child_indices())
        return tuple(i for i in range(self.nvars) if i not in children_indices)

    def child_indices(self) -> tuple[int, ...]:
        children_indices = set()
        for children in self.local_children():
            children_indices.update(children)
        return tuple(children_indices)

    def likelihood_function(self, assignment):
        return self.log_likelihood_function(assignment).exp()

    def log_likelihood_function(self, assignment):
        log_likelihood = torch.tensor(0.0)
        for i in self.topological_order():
            factor_vars, factor_function = self._factor_variables[i], self._factor_functions[i]
            factor_assignment = tuple(assignment[var] for var in factor_vars)
            factor_potential = factor_function.log_likelihood_function(factor_assignment) # type:ignore
            log_likelihood = log_likelihood + factor_potential
        return log_likelihood

    @staticmethod
    def join(bayes1, bayes2, shared):
        if not isinstance(bayes1, pgm.BayesianNetwork) or not isinstance(bayes2, pgm.BayesianNetwork):
            raise ValueError("BaysianNetwork.join can only join pgm.BayesianNetwork")

        size_1 = bayes1.nvars
        size_2 = bayes2.nvars
        nvars_o = size_1 + size_2 - len(shared)

        # variable index map
        b2_to_b1 = utils.map_21(size_1, size_2, shared)

        # construct nvals and factor_variables
        nvals_o = list(bayes1.nvals) + [nval for i, nval in enumerate(bayes2.nvals) if i not in shared] # append the non-shared nodes' nvals from graph 2
        local_variables_o = list(bayes1.local_variables()) + [tuple(b2_to_b1[var] for var in vars) for vars in bayes2.local_variables()] # append the factor vars from graph 2 but mapped to new indices
        local_parents_o = list(bayes1.local_parents()) + [tuple(b2_to_b1[var] for var in vars) for vars in bayes2.local_parents()] # append the parent vars from graph 2 but mapped to new indices
        local_children_o = list(bayes1.local_children()) + [tuple(b2_to_b1[var] for var in vars) for vars in bayes2.local_children()] # append the children vars from graph 2 but mapped to new indices
        local_distributions_o = list(bayes1.local_distributions()) + list(bayes2.local_distributions()) # append the factor functions from graph 2
        topological_order_o = bayes1.topological_order()
        topological_order_o = topological_order_o + [i + len(topological_order_o) for i in bayes2.topological_order()]
        o = BayesianNetwork(nvars_o, nvals_o, local_variables_o, local_parents_o, local_children_o, local_distributions_o, topological_order_o) # type:ignore
        return o
