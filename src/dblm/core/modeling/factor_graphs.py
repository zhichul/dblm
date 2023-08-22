from __future__ import annotations
import code
import itertools
from typing import Sequence
import torch

import tqdm
from dblm.core import graph
from dblm.core.inferencers import belief_propagation
from dblm.core.modeling import probability_tables, utils
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
        self._factor_functions = nn.ModuleList(factor_functions) # type: ignore
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

    def fix_variables(self, observation: dict[int, int]) -> FactorGraph:
        factor_variables, factor_functions = self.get_conditional_factors(observation)
        return FactorGraph(self.nvars, self.nvals,factor_variables, factor_functions)

    def get_conditional_factors(self, observation) -> tuple[list[tuple[int]], list[distribution.LocallyNormalizedDistribution | distribution.GloballyNormalizedDistribution | pgm.PotentialTable]]:
        # modify the factor graph to kill observed variables
        factor_variables = list(self._factor_variables)
        factor_functions = list(self._factor_functions)
        for i in range(len(self._factor_variables)):
            # convert to potential table
            if not isinstance(factor_functions[i], pgm.PotentialTable):
                factor_functions[i] = factor_functions[i].to_potential_table() # type:ignore
            factor_vars = factor_variables[i]
            new_factor_vars = []
            # check for local observation and update factor var list / factor function
            local_observation = dict()
            for local_name, global_name in enumerate(factor_vars):
                if global_name in observation:
                    local_observation[local_name] = observation[global_name]
                else:
                    new_factor_vars.append(global_name)

            if len(local_observation) > 0:
                factor_variables[i] = tuple(new_factor_vars) # type:ignore overwrite with new factor var list
                factor_functions[i] = factor_functions[i].fix_variables(local_observation) # type:ignore overwrite factor function
        return factor_variables, factor_functions # type:ignore

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
            print(factor_potential)
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

        # variable index map
        f2_to_f1 = utils.map_21(size_1, size_2, shared)

        # construct nvals and factor_variables
        nvals_o = list(factor_graph_1.nvals) + [nval for i, nval in enumerate(factor_graph_2.nvals) if i not in shared] # append the non-shared nodes' nvals from graph 2
        factor_variables_o = list(factor_graph_1.factor_variables()) + [tuple(f2_to_f1[var] for var in vars) for vars in factor_graph_2.factor_variables()] # append the factor vars from graph 2 but mapped to new indices
        factor_functions_o = list(factor_graph_1.factor_functions()) + list(factor_graph_2.factor_functions()) # append the factor functions from graph 2

        o = FactorGraph(nvars_o, nvals_o, factor_variables_o, factor_functions_o) # type: ignore
        return o

    # TODO: implement saving and loading

class BPAutoregressiveIncompleteLikelihoodFactorGraph(FactorGraph, distribution.IncompleteLikelihoodMixin):

    def __init__(self, nvars: int, nvals: list[int], factor_variables: list[tuple[int]], factor_functions: list[distribution.LocallyNormalizedDistribution | distribution.GloballyNormalizedDistribution | pgm.PotentialTable], observable: list[tuple[int, int]]) -> None:
        """`observable` is a list of pairs of which the first is the factor index, and the second is the variable index.
        Assumes that observable is sorted w.r.t. the factor index, and that the factor graph represents a autoregressive model,
        such that when computing p(x_i | prefix), the marginalization over x>i can be skipped (since an autoregressive model is
        locally normalized). It also assumes any factor that generates z/x conditioned on x_i is ordered after the factor for x_i
        in the factor_functions list.
        """
        super().__init__(nvars, nvals, factor_variables, factor_functions)
        self.observable = observable
        self.observable_factors = {k for k,_ in observable}
        self.observable_variables = {v for _,v in observable}
        self.observable_variables_to_factors = {v:k for k,v in observable}

    def incomplete_likelihood_function(self, assignment: Sequence[tuple[int, int]]):
        return self.incomplete_log_likelihood_function(assignment).exp() # type:ignore

    def incomplete_log_likelihood_function(self, assignment: Sequence[tuple[int, int]], iterations=10):
        """Assumes assignment is sorted according to the order of self.observable."""
        assignment = list(assignment)
        if dict(assignment).keys() != self.observable_variables:
            raise ValueError(f"can only evaluate the incomplete log likelihood of the prespecified observables: {self.observable_variables} got {dict(assignment).keys()}")
        bp = belief_propagation.FactorGraphBeliefPropagation()
        log_likelihood = torch.tensor(0.0)
        for i, (var, val) in enumerate(assignment):
            factor = self.observable_variables_to_factors[var]
            sub_model = FactorGraph(var+1, self.nvals[:var+1], self._factor_variables[:factor+1], self._factor_functions[:factor+1]) #type:ignore
            inference_results = bp.inference(sub_model, dict(assignment[:i]), [var], iterations=iterations) # observe the previous, query the ith
            log_conditional_likelihood = inference_results.query_marginals[0].log_probability_table()[val]
            log_likelihood = log_likelihood + log_conditional_likelihood
        return log_likelihood

    @staticmethod
    def from_factor_graph(fg: pgm.FactorGraphModel, observable: list[tuple[int, int]]):
        return BPAutoregressiveIncompleteLikelihoodFactorGraph(fg.nvars, fg.nvals, fg.factor_variables(), fg.factor_functions(), observable) # type:ignore

class AutoRegressiveBayesNetMixin:
    """This class gives factor graphs that are really autoregressive bayes nets the bayes net interface.
    Assumes whoever subclasses this implements pgm.FactorGraphModel
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # Bayes Net interface
    def local_distributions(self):
        return self.factor_functions() # type:ignore

    def topological_order(self):
        return list(range(len(self.factor_functions()))) # type: ignore

    def local_variables(self):
        return self.factor_variables() # type:ignore

    def local_parents(self):
        return [vars[:-1] for vars in self.factor_variables()] # type:ignore

    def local_children(self):
        return [vars[-1:] for vars in self.factor_variables()] # type:ignore
