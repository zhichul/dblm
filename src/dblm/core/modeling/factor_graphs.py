from __future__ import annotations
import code
import itertools
import sys
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
                 factor_functions: list[pgm.PotentialTable]) -> None:
        """factor_functions is required to be a subclass of both nn.Module too."""
        self.call_super_init = True
        super().__init__()
        self._nvars = nvars # _nvars must be set for MultivariateFunction
        self._nvals = nvals # _nvals must be set for MultivariateFunction
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
        self._factor_functions: Sequence[pgm.PotentialTable]
        return probability_tables.LogLinearProbabilityTable.joint_from_factors(self._nvars, self._nvals, self._factor_variables, self._factor_functions)

    def to_potential_table(self) -> pgm.PotentialTable:
        self._factor_functions: Sequence[pgm.PotentialTable]
        return probability_tables.LogLinearPotentialTable.joint_from_factors(self._nvars, self._nvals, self._factor_variables, self._factor_functions)

    # FactorGraphModel
    def factor_variables(self) -> list[tuple[int]]:
        return self._factor_variables

    def factor_functions(self) -> list[distribution.Distribution]:
        return self._factor_functions # type: ignore

    def conditional_factor_variables(self, observation: dict[int, int] | dict[int, torch.Tensor]) -> list[tuple[int]]:
        return [tuple(var for var in factor_vars if var not in observation) for factor_vars in self._factor_variables]

    def conditional_factor_functions(self, observation: dict[int, int] | dict[int, torch.Tensor]) -> list[pgm.PotentialTable]:
        new_factor_functions = []
        for factor_vars, factor_function in zip(self._factor_variables, self._factor_functions):
            factor_function : pgm.PotentialTable
            local_observation: dict[int,int] | dict[int, torch.Tensor] = {local_name: observation[global_name] for local_name, global_name in enumerate(factor_vars) if global_name in observation} # type:ignore
            if len(local_observation) > 0:
                new_factor_functions.append(factor_function.condition_on(local_observation))
            else:
                if isinstance(next(observation.values().__iter__()), torch.Tensor):
                    new_factor_functions.append(factor_function.expand_batch_dimensions(tuple(next(observation.values().__iter__()).size()))) # type:ignore
                else:
                    new_factor_functions.append(factor_function)
        return new_factor_functions

    # GloballyNormalizedDistribution
    def unnormalized_likelihood_function(self, assignment: tuple[int,...] | tuple[torch.Tensor,...]) -> torch.Tensor:
        return self.log_unnormalized_likelihood_function(assignment).exp()

    def log_unnormalized_likelihood_function(self, assignment: tuple[int,...] | tuple[torch.Tensor,...]) -> torch.Tensor:
        lu = torch.tensor(0.0)
        for factor_vars, factor_function in zip(self._factor_variables, self._factor_functions):
            factor_assignment = tuple(assignment[var] for var in factor_vars)
            factor_potential = factor_function.log_potential_value(factor_assignment)
            lu = lu + factor_potential
        return lu

    def partition_function(self, use_cache=False) -> torch.Tensor:
        return self.log_partition_function(use_cache=use_cache).exp()

    def log_partition_function(self, use_cache=False) -> torch.Tensor:
        # TODO: this is brute force, we should probably turn it into a giant matrix first and use matrix operations but that would blowup memory
        if self._log_partition_function_cache is None or not use_cache:
            lz = torch.tensor(0.0)
            for assignment in tqdm.tqdm(itertools.product(*[list(range(nval)) for nval in self._nvals])):
                lz = lz + self.log_unnormalized_likelihood_function(assignment)
            self._log_partition_function_cache = lz
        return self._log_partition_function_cache

    # Distribution
    def condition_on(self, observation: dict[int, int] | dict[int, torch.Tensor]) -> FactorGraph:
        factor_variables = self.conditional_factor_variables(observation)
        factor_functions = self.conditional_factor_functions(observation)
        return FactorGraph(self.nvars, self.nvals,factor_variables, factor_functions)

    # Self
    @staticmethod
    def join(factor_graph_1: pgm.FactorGraphModel, factor_graph_2: pgm.FactorGraphModel, shared: dict[int, int]) -> FactorGraph:
        """Always shifts the indices of the second graph, so better to join from the right.
        `shared` is a dict from id in second graph to id in first.
        """
        # TODO: the shared argument is not friendly to a chain of joins since every join updates some indices.
        if not (isinstance(factor_graph_1, pgm.FactorGraphModel) and isinstance(factor_graph_2, pgm.FactorGraphModel)):
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

    def __init__(self, nvars: int, nvals: list[int], factor_variables: list[tuple[int]], factor_functions: list[pgm.PotentialTable], observable: list[tuple[int, int]]) -> None:
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

    def incomplete_likelihood_function(self, assignment: Sequence[tuple[int, int]] | Sequence[tuple[int, torch.Tensor]]) -> torch.Tensor:
        return self.incomplete_log_likelihood_function(assignment).exp() # type:ignore

    def incomplete_log_likelihood_function(self, assignment: Sequence[tuple[int, int]] | Sequence[tuple[int, torch.Tensor]], iterations=10) -> torch.Tensor:
        """Assumes assignment is sorted according to the order of self.observable."""
        assignment = list(assignment) # type:ignore
        if dict(assignment).keys() != self.observable_variables:
            raise ValueError(f"can only evaluate the incomplete log likelihood of the prespecified observables: {self.observable_variables} got {dict(assignment).keys()}")
        bp = belief_propagation.FactorGraphBeliefPropagation()
        log_likelihood = torch.tensor(0.0)
        for i, (var, val) in enumerate(assignment):
            factor = self.observable_variables_to_factors[var]
            sub_model = FactorGraph(var+1, self.nvals[:var+1], self._factor_variables[:factor+1], self._factor_functions[:factor+1]) #type:ignore
            inference_results = bp.inference(sub_model, dict(assignment[:i]), [var], iterations=iterations) # type:ignore observe the previous, query the ith
            log_conditional_likelihood = inference_results.query_marginals[0].log_probability_table()[val]
            log_likelihood = log_likelihood + log_conditional_likelihood
        return log_likelihood

    @staticmethod
    def from_factor_graph(fg: pgm.FactorGraphModel, observable: list[tuple[int, int]]):
        return BPAutoregressiveIncompleteLikelihoodFactorGraph(fg.nvars, fg.nvals, fg.factor_variables(), fg.factor_functions(), observable) # type:ignore

class AutoRegressiveBayesNetMixin:
    """This class gives factor graphs that are really autoregressive
    bayes nets the bayes net interface. Assumes whoever subclasses
    this implements pgm.FactorGraphModel, and that the factors are all
    really LocallyNormalizedDistributions (and since FactorGraphModels
    must have PotentialTables as factors, this together means these
    factors must be ProbabilityTables), with the last varible being the
    children. This means the number of factors should match the number of
    variables in the model.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for factor_function in self.factor_functions(): # type:ignore
            if not isinstance(factor_function, pgm.ProbabilityTable):
                raise ValueError(f"{type(factor_function)}")
            if factor_function.parent_indices() != tuple(range(factor_function.nvars - 1)):
                print(f"not a autoregressive factor {factor_function} "
                      f"parents={factor_function.parent_indices()} nvars={factor_function.nvars}", file=sys.stderr)
                code.interact(local=locals()) # bad parent indices of autoregressive bayes
                raise ValueError(f"not a autoregressive factor {factor_function}")
        # TODO: sortedness of the factors (w.r.t. the variable order) is not tested

    # BayesianNetwork
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
