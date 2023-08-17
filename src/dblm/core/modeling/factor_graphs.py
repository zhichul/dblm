from __future__ import annotations
import itertools

import tqdm
from dblm.core import graph
from dblm.core.modeling import probability_tables
from dblm.core.interfaces import distribution, pgm
import torch.nn as nn

class FactorGraph(nn.Module, pgm.FactorGraphModel):

    def __init__(self, nvars: int,
                 nvals: list[int],
                 factor_variables: list[tuple[int]],
                 factor_functions: list[distribution.LocallyNormalizedDistribution | distribution.GloballyNormalizedDistribution | pgm.PotentialTable]) -> None:
        """factor_functions is required to be a subclass of both nn.Module too."""
        super().__init__()
        self._nvars = nvars
        self._nvals = nvals
        self._factor_variables = factor_variables
        self._factor_functions = nn.ModuleList()
        self._factor_functions.extend(factor_functions) # type: ignore
        self._graph = graph.FactorGraph(nvars, len(factor_functions))
        for i in range(len(factor_functions)):
            for var in factor_variables[i]:
                self._graph.add_edge(i, var)

        self._partition_function_cache = None
        self._log_partition_function_cache = None

    # ProbabilisticGraphicalModel
    def graph(self):
        return self._graph

    def to_probability_table(self) -> pgm.ProbabilityTable:
        return probability_tables.LogLinearProbabilityTable.joint_from_factors(self._nvars, self._nvals, self._factor_variables, self._factor_functions) # type: ignore

    def to_potential_table(self) -> pgm.PotentialTable:
        return probability_tables.LogLinearPotentialTable.joint_from_factors(self._nvars, self._nvals, self._factor_variables, self._factor_functions) # type: ignore

    # FactorGraph
    def factor_variables(self) -> list[tuple[int]]:
        return self._factor_variables

    def factor_functions(self) -> list[distribution.Distribution]:
        return self._factor_functions # type: ignore

    # GloballyNormalizedDistribution
    def unnormalized_likelihood_function(self, assignment):
        u = 1
        for factor_vars, factor_function in zip(self._factor_variables, self._factor_functions):
            factor_assignment = tuple(assignment[var] for var in factor_vars)
            if isinstance(factor_function, distribution.GloballyNormalizedDistribution):
                factor_potential = factor_function.unnormalized_likelihood_function(factor_assignment)
            elif isinstance(factor_function, distribution.LocallyNormalizedDistribution):
                factor_potential = factor_function.likelihood_function(factor_assignment)
            elif isinstance(factor_function, pgm.PotentialTable):
                factor_potential = factor_function.potential_value(factor_assignment)
            else:
                raise AssertionError(f"unexpected factor type: {type(factor_function)}")
            u = u * factor_potential
        return u

    def log_unnormalized_likelihood_function(self, assignment):
        lu = 0
        for factor_vars, factor_function in zip(self._factor_variables, self._factor_functions):
            factor_assignment = tuple(assignment[var] for var in factor_vars)
            if isinstance(factor_function, distribution.GloballyNormalizedDistribution):
                factor_potential = factor_function.log_unnormalized_likelihood_function(factor_assignment)
            elif isinstance(factor_function, distribution.LocallyNormalizedDistribution):
                factor_potential = factor_function.log_likelihood_function(factor_assignment)
            elif isinstance(factor_function, pgm.PotentialTable):
                factor_potential = factor_function.log_potential_value(factor_assignment)
            else:
                raise AssertionError(f"unexpected factor type: {type(factor_function)}")
            lu = lu + factor_potential
        return lu

    def partition_function(self, use_cache=False):
        if self._partition_function_cache is None or not use_cache:
            z = 0
            for assignment in tqdm.tqdm(itertools.product(*[list(range(nval)) for nval in self._nvals])):
                z = z * self.unnormalized_likelihood_function(assignment)
            self._partition_function_cache = z
        return self._partition_function_cache

    def log_partition_function(self, use_cache=False):
        if self._log_partition_function_cache is None or not use_cache:
            lz = 0
            for assignment in tqdm.tqdm(itertools.product(*[list(range(nval)) for nval in self._nvals])):
                lz = lz + self.log_unnormalized_likelihood_function(assignment)
        return self._log_partition_function_cache

    # Self
    @staticmethod
    def join(factor_graph_1, factor_graph_2, shared: dict[int, int]) -> FactorGraph:
        """Always shifts the indices of the second graph, so better to join from the right.
        `shared` is a dict from id in second graph to id in first.
        """
        # TODO: the shared argument is not friendly to a chain of joins since every join updates some indices.
        if not (isinstance(factor_graph_1, FactorGraph) and isinstance(factor_graph_2, FactorGraph)):
            raise ValueError
        size_1 = factor_graph_1.nvars
        size_2 = factor_graph_2.nvars
        nvars_o = size_1 + size_2 - len(shared)

        # map
        f2_to_f1 = dict()
        next_node_id = size_1
        for i in range(size_2):
            if i not in shared:
                f2_to_f1[i] = next_node_id
                next_node_id += 1
            else:
                f2_to_f1[i] = shared[i]

        # construct nvals and factor_variables
        nvals_o = list(factor_graph_1.nvals) + [nval for i, nval in enumerate(factor_graph_2.nvals) if i not in shared] # append the non-shared nodes' nvals from graph 2
        factor_variables_o = list(factor_graph_1.factor_variables()) + [tuple(f2_to_f1[var] for var in vars) for vars in factor_graph_2.factor_variables()] # append the factor vars from graph 2 but mapped to new indices
        factor_functions_o = list(factor_graph_1.factor_functions()) + list(factor_graph_2.factor_functions()) # append the factor functions from graph 2

        o = FactorGraph(nvars_o, nvals_o, factor_variables_o, factor_functions_o) # type: ignore
        return o

    # TODO: implement saving and loading