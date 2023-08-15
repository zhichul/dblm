from __future__ import annotations
from typing import Sequence
from dblm.core.interfaces import pgm
from dblm.core.modeling import factor_graphs
from dblm.core.modeling import constants
import torch.nn as nn
import torch

from dblm.core.interfaces.pgm import FactorGraphModel, ProbabilityTable


class LogLinearTable(nn.Module):

    def __init__(self, size, initializer: constants.TensorInitializer | torch.Tensor, requires_grad:bool=True) -> None:
        """If initializer is a tensor, requires_grad is ignored."""
        super().__init__()
        # Store as ndarray
        if initializer is constants.TensorInitializer.CONSTANT:
            self.logits = nn.Parameter(torch.zeros(size), requires_grad=requires_grad)
        elif initializer is constants.TensorInitializer.UNIFORM:
            self.logits = nn.Parameter(torch.rand(size), requires_grad=requires_grad)
        elif isinstance(initializer, torch.Tensor):
            if list(initializer.size()) != list(size):
                raise ValueError(f"Expected initializer size ({initializer.size()}) to match decalred size ({size})")
            self.logits = initializer
        else:
            raise NotImplementedError()


class LogLinearPotentialTable(LogLinearTable, pgm.PotentialTable):

    def potential_table(self) -> torch.Tensor:
        return self.logits.exp()

    def log_potential_table(self) -> torch.Tensor:
        return self.logits

    def potential_value(self, assignment):
        return self.potential_table()[assignment]

    def log_potential_value(self, assignment):
        return self.log_potential_table()[assignment]


class LogLinearProbabilityTable(LogLinearTable, pgm.ProbabilityTable, pgm.PotentialTable):

    def __init__(self, size, parents: list[int], initializer: constants.TensorInitializer | torch.Tensor, requires_grad:bool=True) -> None:
        super().__init__(size, initializer, requires_grad=requires_grad)
        self._parents = parents
        self._children = [i for i in range(len(size)) if i not in parents]
        self._permute = self._parents + self._children
        self._reverse_permute = [self._permute.index(i) for i in range(len(size))] #TODO fixed this n^2 slowness
        self._probability_table_cache = None
        self._log_probability_table_cache = None

    def probability_table(self, use_cache=False) -> torch.Tensor:
        """This function normalizes over the children dimension, as in a pa -> ch graph."""
        if self._probability_table_cache is None or not use_cache:
            permuted_logits = self.logits.permute(*self._permute)
            original_size = permuted_logits.size()
            flattened_size = original_size[:len(self._parents)] + torch.Size([-1])
            flattened_logits = permuted_logits.reshape(*flattened_size)
            probs = flattened_logits.softmax(dim=-1)
            probs = probs.reshape(original_size)
            probs = probs.permute(self._reverse_permute)
            self._probability_table_cache = probs
        return self._probability_table_cache

    def log_probability_table(self, use_cache=False) -> torch.Tensor: #TODO fix duplicate code with probability_table!
        if self._log_probability_table_cache is None or not use_cache:
            permuted_logits = self.logits.permute(*self._permute)
            original_size = permuted_logits.size()
            flattened_size = original_size[:len(self._parents)] + torch.Size([-1])
            flattened_logits = permuted_logits.reshape(*flattened_size)
            log_probs = flattened_logits.log_softmax(dim=-1)
            log_probs = log_probs.reshape(original_size)
            log_probs = log_probs.permute(self._reverse_permute)
            self._log_probability_table_cache = log_probs
        return self._log_probability_table_cache

    def likelihood_function(self, assignment):
        return self.probability_table()[assignment]

    def log_likelihood_function(self, assignment):
        return self.log_probability_table()[assignment]

    def potential_value(self, assignment):
        return self.likelihood_function(assignment)

    def log_potential_value(self, assignment):
        return self.likelihood_function(assignment)

    def potential_table(self) -> torch.Tensor:
        """IMPORTANT this returns a normalized table!"""
        return self.probability_table()

    def log_potential_table(self) -> torch.Tensor:
        """IMPORTANT this returns a normalized table!"""
        return self.log_probability_table()

    def to_probability_table(self) -> ProbabilityTable:
        return self

    def to_factor_graph_model(self) -> FactorGraphModel:
        nvars = self.nvars()
        nvals = self.nvals()
        factor_vars = [list(range(nvars))]
        factor_functions = [self]
        return factor_graphs.FactorGraph(nvars, nvals, factor_vars, factor_functions) # type: ignore

    def nvars(self) -> int:
        return len(self.logits.size())

    def nvals(self) -> list[int]:
        return list(self.logits.size())

    @staticmethod
    def joint_from_factors(nvars: int, nvals: list[int], factor_variables: Sequence[tuple[int]], factor_functions: Sequence[pgm.PotentialTable]):
        """
        IMPORTANT: assumes the variables in the factor functions are in numerical order!
        e.g. if the set of variables range from v1 to v10, assumes the factor table
        on the set {v2 v5 v3} is always ordered with axis corresponding to v2 v3 v5.

        The sizes of the factors must respect the declared sizes encoded by nvals.
        """
        table = None
        for factor_vars, factor_fn in zip(factor_variables, factor_functions):
            size = [1] * nvars
            local_factor = factor_fn.log_potential_table()
            for var in factor_vars:
                size[var] = nvals[var]
            local_factor = local_factor.reshape(*size)
            table = local_factor if table is None else table + local_factor
        assert table is not None
        return LogLinearProbabilityTable(nvals, [], table)

