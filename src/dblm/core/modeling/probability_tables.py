from __future__ import annotations
from typing import Sequence
from dblm.core.interfaces import pgm, featurizer
from dblm.core.modeling import factor_graphs
from dblm.core.modeling import constants
import torch.nn as nn
import torch


class LogLinearFeaturizedTable(nn.Module):

    def __init__(self, size, featurizer: featurizer.Featurizer, initializer: constants.TensorInitializer, requires_grad:bool=True) -> None:
        super().__init__()
        self.assignments = nn.Parameter(torch.cartesian_prod(*(torch.arange(s) for s in size[:-1])), requires_grad=False)
        self.featurizer = featurizer
        self.layer = nn.Linear(self.featurizer.out_features, 1, bias=False)
        self._nvars = len(size)
        self._nvals = list(size)
        if isinstance(initializer, constants.TensorInitializer):
            if initializer == constants.TensorInitializer.UNIFORM:
                nn.init.uniform_(self.layer.weight, 0, 0.1)
            else:
                raise NotImplementedError()
        else:
            raise ValueError("Only TensorInitializer is allowed as initializer.")

    @property
    def logits(self):
        features = self.featurizer(self.assignments)
        return self.layer(features).reshape(*self._nvals)

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
        self._nvars = len(self.logits.size())
        self._nvals = list(self.logits.size())

class LogLinearTableInferenceMixin:
    """This mixin assumes the inheriter implements interfaces of pgm.ProbabilisticGraphicalModel and pgm.PotentialTable.
    The output it produces is communicated as a LogLinearPotentialTable.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def fix_variables(self, observation: dict[int, int]):
        index = [slice(None,None,None)] * self.nvars # type:ignore
        for k,v in observation.items():
            index[k] = v # type:ignore
        logits = self.log_potential_table()[index] #type:ignore
        return LogLinearPotentialTable(logits.size(), logits)

    def marginalize_over(self, variables):
        logits = self.log_potential_table() # type:ignore
        for var in sorted(variables, reverse=True):
            logits = logits.logsumexp(dim=var)
        return LogLinearPotentialTable(logits.size(), logits)

class LogLinearPotentialMixin:

    def potential_table(self) -> torch.Tensor:
        return self.logits.exp() # type:ignore

    def log_potential_table(self) -> torch.Tensor:
        return self.logits # type:ignore

    def potential_value(self, assignment):
        return self.potential_table()[assignment]

    def log_potential_value(self, assignment):
        return self.log_potential_table()[assignment]

    def renormalize(self):
        return LogLinearProbabilityTable(self.logits.size(), [], self.logits) # type:ignore

class LogLinearPotentialTable(LogLinearPotentialMixin, LogLinearTableInferenceMixin, LogLinearTable, pgm.PotentialTable):

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
        return LogLinearPotentialTable(nvals, table)

class LogLinearProbabilityMixin:

    def __init__(self, size, parents, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._parents = parents
        self._children = [i for i in range(len(size)) if i not in parents]
        self._permute = self._parents + self._children
        self._reverse_permute = [self._permute.index(i) for i in range(len(size))] #TODO fixed this n^2 slowness
        self._probability_table_cache = None
        self._log_probability_table_cache = None

    def _permute_children_to_last_and_flatten(self):
        permuted_logits = self.logits.permute(*self._permute) # type:ignore
        permuted_size = permuted_logits.size()
        flattened_size = permuted_size[:len(self._parents)] + torch.Size([-1])
        flattened_logits = permuted_logits.reshape(*flattened_size)
        return flattened_logits, permuted_size

    def _deflatten_and_permute_children_to_original(self, normalized_flattened_table, permuted_size):
        normalized_flattened_table = normalized_flattened_table.reshape(permuted_size)
        normalized_flattened_table = normalized_flattened_table.permute(self._reverse_permute)
        return normalized_flattened_table

    def probability_table(self, use_cache=False) -> torch.Tensor:
        """This function normalizes over the children dimension, as in a pa -> ch graph.
        The axes that are normalized can be counterintuitive if ch variables appear interleaved with pa,
        since it's the ch axes that are normalized over. Sometimes this cannot be avoided because
        the variable order of some larger graph has to be respected so that ch variable has to appear before
        some pa variable.
        """
        if self._probability_table_cache is None or not use_cache:
            flattened_logits, permuted_size = self._permute_children_to_last_and_flatten()
            probs = flattened_logits.softmax(dim=-1)
            probs = self._deflatten_and_permute_children_to_original(probs, permuted_size)
            self._probability_table_cache = probs
        return self._probability_table_cache

    def log_probability_table(self, use_cache=False) -> torch.Tensor: #TODO fix duplicate code with probability_table!
        if self._log_probability_table_cache is None or not use_cache:
            flattened_logits, permuted_size = self._permute_children_to_last_and_flatten()
            log_probs = flattened_logits.log_softmax(dim=-1)
            log_probs = self._deflatten_and_permute_children_to_original(log_probs, permuted_size)
            self._log_probability_table_cache = log_probs
        return self._log_probability_table_cache

    def likelihood_function(self, assignment):
        return self.probability_table()[assignment]

    def log_likelihood_function(self, assignment):
        return self.log_probability_table()[assignment]

    def potential_table(self) -> torch.Tensor:
        """IMPORTANT this returns a normalized table!"""
        return self.probability_table()

    def log_potential_table(self) -> torch.Tensor:
        """IMPORTANT this returns a normalized table!"""
        return self.log_probability_table()

    def potential_value(self, assignment):
        return self.potential_table()[assignment]

    def log_potential_value(self, assignment):
        return self.log_potential_table()[assignment]

    def to_probability_table(self):
        return self

    def renormalize(self):
        return self

    def to_factor_graph_model(self) -> pgm.FactorGraphModel:
        nvars = self.nvars # type:ignore assumes whoever is using the mixin is implementing MultivariateFunction
        nvals = self.nvals # type:ignore assumes whoever is using the mixin is implementing MultivariateFunction
        factor_vars = [list(range(nvars))]
        factor_functions = [self]
        return factor_graphs.FactorGraph(nvars, nvals, factor_vars, factor_functions) # type: ignore

class LogLinearProbabilityTable(LogLinearProbabilityMixin, LogLinearTableInferenceMixin, LogLinearTable, pgm.ProbabilityTable, pgm.PotentialTable):
    def __init__(self, size, parents: list[int], initializer: constants.TensorInitializer | torch.Tensor, requires_grad:bool=True) -> None:
        super().__init__(size, parents, size, initializer, requires_grad=requires_grad)
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


class LogLinearFeaturizedProbabilityTable(LogLinearProbabilityMixin, LogLinearTableInferenceMixin, LogLinearFeaturizedTable, pgm.ProbabilityTable, pgm.PotentialTable):

    def __init__(self, size, parents, feature_extractor, initializer, requires_grad=True) -> None:
        super().__init__(size, parents, size, feature_extractor, initializer, requires_grad=requires_grad)
