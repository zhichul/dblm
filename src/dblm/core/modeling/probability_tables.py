from __future__ import annotations
from functools import reduce
from typing import Sequence
from dblm.core.interfaces import pgm, featurizer
from dblm.core.modeling import factor_graphs, bayesian_networks
from dblm.core.modeling import constants
import torch.nn as nn
import torch

from dblm.utils import safe_operations

"""
The implementations of the behavior of PotentialTable and ProbabilityTable is split to three units:

1) A class that defines a .logits property, which is essentially the table itself.
    LogLinearFeaturizedTable implements the computation of the logits using a single linear
        layer with no bias, according to some feature function, which computes a different
        value for each assignment of the variables.
    LogLinearTable implements the .logts directly as a parameter tensor.

2) A class that implements the conditionalization and marginalization behavior.
    LogLinearTableInferenceMixin implements this by indexing and logsumexp respectively.

3) A class that computes values of interest from the internal .logits properties to implement the
    parts of PotentialTable or ProbabilityTable interfaces that return tables / values from the table.
    LogLinearPotentialMixin implements the PotentialTable interface.
    LogLinearProbabilityMixin implements the ProbabilityTable interface.

They combine into three concrete types of tables:

a) LogLinearPotentialTable, with direct parametrization of logits.
    This class also has a joint_from_factors method that allows one to materialize into a table from factor-graph like representations, in that case
    the sum of all the log factors form the logits rather than a direct parametrization.
b) LogLinearProbabilityTable, with direct parametrization of logits.
    This class also has a joint_from_factors method that allows one to materialize into a table from factor-graph like representations, in that case
    the sum of all the log factors form the logits rather than a direct parametrization.
c) LogLinearFeaturizedProbabilityTable, with indrect parametrization of logits through a linear function of features of assignments.
"""
class LogLinearFeaturizedTable(nn.Module):

    def __init__(self, size, featurizer: featurizer.Featurizer, initializer: constants.TensorInitializer, requires_grad:bool=True) -> None:
        self.call_super_init = True
        super().__init__()
        self.assignments = nn.Parameter(torch.cartesian_prod(*(torch.arange(s) for s in size)), requires_grad=requires_grad)
        self.featurizer = featurizer
        self.layer = nn.Linear(self.featurizer.out_features, 1, bias=False)
        if isinstance(initializer, constants.TensorInitializer):
            if initializer == constants.TensorInitializer.UNIFORM:
                nn.init.uniform_(self.layer.weight, 0, 0.1)
            else:
                raise NotImplementedError()
        else:
            raise ValueError("Only TensorInitializer is allowed as initializer.")

        # batch dimensions are always assumed to be at the front
        self._nvars = len(size)
        self._nvals = list(size)
        self._batch_dims = 0
        self._batch_size = tuple()

    @property
    def logits(self):
        features = self.featurizer(self.assignments)
        return self.layer(features).reshape(*self._nvals)

class LogLinearTable(nn.Module):

    def __init__(self, size, initializer: constants.TensorInitializer | torch.Tensor, requires_grad:bool=True, batch_dims=0) -> None:
        """If initializer is a tensor, requires_grad is ignored."""
        self.call_super_init = True
        super().__init__()
        # Store as ndarray
        if isinstance(initializer, constants.TensorInitializer) and batch_dims != 0:
            raise ValueError("not allowed to initialize LogLinearTable with batch_dims > 0")
        if initializer is constants.TensorInitializer.CONSTANT:
            self._logits = nn.Parameter(torch.zeros(size), requires_grad=requires_grad)
        elif initializer is constants.TensorInitializer.UNIFORM:
            self._logits = nn.Parameter(torch.rand(size), requires_grad=requires_grad)
        elif isinstance(initializer, (nn.Parameter, torch.Tensor)):
            if list(initializer.size()) != list(size):
                raise ValueError(f"Expected initializer size ({initializer.size()}) to match decalred size ({size})")
            self._logits = initializer
        else:
            raise NotImplementedError()

        # batch dimensions are always assumed to be at the front
        self._nvars = len(self._logits.size()) - batch_dims
        self._nvals = list(self._logits.size())[batch_dims:]
        self._batch_dims = batch_dims
        self._batch_size = tuple(size)[:batch_dims]

    @property
    def logits(self):
        return self._logits


class LogLinearTableInferenceMixin:
    """This mixin assumes the inheriter implements interfaces of pgm.ProbabilisticGraphicalModel and pgm.PotentialTable.
    The output it produces is communicated as a LogLinearPotentialTable.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def condition_on(self, observation: dict[int, int] |  dict[int, torch.Tensor], cartesian=True) -> LogLinearPotentialTable:
        """When supplied a tensor of assignments for the observations,
        returns a LogLinearPotentialTable with ADDITIONAL batch dimensions corresponding
        to the dimensions of the observations. This method "unsqueezes" new batch dimensions.
        """
        is_single_observation = isinstance(next(observation.values().__iter__()), int)
        new_batch_dims = 0 if (is_single_observation or not cartesian) else len(next(observation.values().__iter__()).size()) # type:ignore
        index = [slice(None,None,None)] * (self.batch_dims + self.nvars) # type:ignore
        for k,v in observation.items():
            index[self.batch_dims + k] = v # type:ignore
        if cartesian or is_single_observation:
            logits = self.log_potential_table()[tuple(index)] #type:ignore
        else:
            batch_total = reduce(int.__mul__, self.batch_size, 1) # type:ignore
            logits = self.log_potential_table().reshape(-1, *self.nvals)[(list(range(batch_total)), *index[self.batch_dims:])] # type:ignore
            logits = logits.reshape(*(*self.batch_size, *logits.size()[1:])) # type:ignore
            assert len(logits.size()) == len(self.logits.size()) - len(observation) # type:ignore
        if not is_single_observation and cartesian:
            # permute the cartesian prod batch size to the back of the front, this may be costly
            min_obs_var_index = min(observation.keys()) + self.batch_dims # type:ignore
            permutation = list(range(len(logits.size())))
            old_batch = permutation[:self.batch_dims] # type:ignore
            new_batch = permutation[min_obs_var_index:min_obs_var_index + new_batch_dims]
            front = permutation[self.batch_dims:min_obs_var_index] # type:ignore
            back = permutation[min_obs_var_index + new_batch_dims:] # type:ignore
            permutation = old_batch + new_batch + front + back
            logits = logits.permute(*permutation)
        return LogLinearPotentialTable(logits.size(), logits, batch_dims=self.batch_dims + new_batch_dims) # type:ignore

    def marginalize_over(self, variables: Sequence[int]):
        """This method "squeezes" away variables dimensions."""
        if len(variables) == 0:
            return self
        logits = self.log_potential_table() # type:ignore
        for var in sorted(variables, reverse=True):
            logits = safe_operations.logsumexp(logits, dim=self.batch_dims + var) # type:ignore
        return LogLinearPotentialTable(logits.size(), logits, batch_dims=self.batch_dims) # type:ignore

class LogLinearPotentialMixin:

    def potential_table(self) -> torch.Tensor:
        return self.logits.exp() # type:ignore

    def log_potential_table(self) -> torch.Tensor:
        return self.logits # type:ignore

    def _index(self, assignment, table_fn):
        if len(assignment) > 0 and isinstance(assignment[0], torch.Tensor):
            assignment = tuple(a.view(-1) for a in assignment) # type:ignore
            batch_size: tuple[int,...] = self.batch_size # type:ignore
            batch_total = reduce(int.__mul__, batch_size, 1)
            if batch_total == 1 and self.batch_dims == 0: # type:ignore
                # IMPORTANT: only this case we implicitly expand the batch dimensions to make it easier
                # other cases must keep the size of assignment same as self.batch_size
                # used to raise ValueError("If batch_total = 1 use integer indices not tensor")
                return table_fn()[assignment]
            else:
                return table_fn().view(-1, *self.nvals)[(list(range(batch_total)),*assignment)].reshape(*batch_size) # type:ignore
        else:
            return table_fn()[assignment]

    def potential_value(self, assignment):
        return self._index(assignment, self.potential_table)

    def log_potential_value(self, assignment):
        return self._index(assignment, self.log_potential_table)

    def renormalize(self):
        return LogLinearProbabilityTable(self.logits.size(), [], self.logits, batch_dims=self.batch_dims) # type:ignore

class LogLinearPotentialTable(LogLinearPotentialMixin, LogLinearTableInferenceMixin, LogLinearTable, pgm.PotentialTable):

    @staticmethod
    def joint_from_factors(nvars: int, nvals: list[int], factor_variables: Sequence[tuple[int, ...]], factor_functions: Sequence[pgm.PotentialTable]):
        """
        IMPORTANT: assumes the variables in the factor functions are in numerical order!
        e.g. if the set of variables range from v1 to v10, assumes the factor table
        on the set {v2 v5 v3} is always ordered with axis corresponding to v2 v3 v5.

        The sizes of the factors must respect the declared sizes encoded by nvals.
        """
        if any(factor_function.batch_size != factor_functions[0].batch_size for factor_function in factor_functions):
            raise ValueError(f"Can only join PotentialTables with the same batch_size: {[factor_function.batch_size for factor_function in factor_functions]}")
        table = None
        batch_dims = factor_functions[0].batch_dims
        batch_size = factor_functions[0].batch_size
        for factor_vars, factor_fn in zip(factor_variables, factor_functions):
            size = [1] * (nvars)
            local_factor = factor_fn.log_potential_table()
            for var in factor_vars:
                size[var] = nvals[var]
            local_factor = local_factor.reshape(*(*batch_size, *size))
            table = local_factor if table is None else table + local_factor
        assert table is not None
        return LogLinearPotentialTable((*batch_size, *nvals), table, batch_dims=batch_dims)

    def expand_batch_dimensions_(self, batch_sizes: tuple[int, ...]) -> LogLinearPotentialTable:
        self.expand_batch_dimensions_meta_(batch_sizes) # type:ignore
        self._logits.data = self.logits.data.expand((*self.batch_size, *self.nvals))
        return self

    def expand_batch_dimensions(self, batch_sizes: tuple[int, ...]) -> LogLinearPotentialTable:
        table = LogLinearPotentialTable(self.logits.size(), self.logits.clone(), batch_dims=self.batch_dims)
        table.expand_batch_dimensions_(batch_sizes) # type:ignore
        return table


class LogLinearProbabilityMixin(LogLinearPotentialMixin):

    def __init__(self, nvars, parents, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._parents = tuple(parents)
        self._children = tuple(i for i in range(nvars) if i not in parents) # type:ignore
        if len(parents) > nvars: # type:ignore
            raise ValueError(f"number of parents {parents} can't be larger than number of variables {nvars}")
        self._permute = self._parents + self._children
        self._reverse_permute = [self._permute.index(i) for i in range(nvars)] #type:ignore TODO fixed this n^2 slowness
        self._probability_table_cache = None
        self._log_probability_table_cache = None

    def parent_indices(self):
        return self._parents

    def child_indices(self):
        return self._children

    @property
    def permutation(self):
        return list(range(self.batch_dims)) + [self.batch_dims + i for i in self._permute] # type:ignore

    @property
    def reverse_permutation(self):
        return list(range(self.batch_dims)) + [self.batch_dims + i for i in self._reverse_permute] # type:ignore

    def _permute_children_to_last_and_flatten(self):
        permuted_logits = self.logits.permute(*(self.permutation)) # type:ignore
        permuted_size = permuted_logits.size()
        flattened_size = permuted_size[:(self.batch_dims + len(self._parents))] + torch.Size([-1]) # type:ignore
        flattened_logits = permuted_logits.reshape(*flattened_size)
        return flattened_logits, permuted_size

    def _deflatten_and_permute_children_to_original(self, normalized_flattened_table, permuted_size):
        normalized_flattened_table = normalized_flattened_table.reshape(permuted_size)
        normalized_flattened_table = normalized_flattened_table.permute(self.reverse_permutation)
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
        return self._index(assignment, self.probability_table)

    def log_likelihood_function(self, assignment):
        return self._index(assignment, self.log_probability_table)

    def potential_table(self) -> torch.Tensor:
        """IMPORTANT this returns a normalized table!"""
        return self.probability_table()

    def log_potential_table(self) -> torch.Tensor:
        """IMPORTANT this returns a normalized table!"""
        return self.log_probability_table()

    def potential_value(self, assignment):
        return self.likelihood_function(assignment)

    def log_potential_value(self, assignment):
        return self.log_likelihood_function(assignment)

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

    def to_bayesian_network(self) -> pgm.FactorGraphModel:
        nvars = self.nvars # type:ignore assumes whoever is using the mixin is implementing MultivariateFunction
        nvals = self.nvals # type:ignore assumes whoever is using the mixin is implementing MultivariateFunction
        factor_vars = [list(range(nvars))]
        parents = [self._parents]
        children = [self._children]
        factor_functions = [self]
        topo_order = [0]
        return bayesian_networks.BayesianNetwork(nvars, nvals, factor_vars, parents, children, factor_functions, topo_order) # type:ignore

    def condition_on(self, observation: dict[int, int] | dict[int, torch.Tensor], cartesian=True) -> LogLinearPotentialTable | LogLinearProbabilityTable:
        is_single_observation = isinstance(next(observation.values().__iter__()), int)
        new_batch_dims = 0 if (is_single_observation or not cartesian) else len(next(observation.values().__iter__()).size()) # type:ignore
        index = [slice(None,None,None)] * (self.batch_dims + self.nvars) # type:ignore
        for k,v in observation.items():
            index[self.batch_dims + k] = v # type:ignore
        if cartesian or is_single_observation:
            logits = self.log_probability_table()[tuple(index)] #type:ignore
        else:
            batch_total = reduce(int.__mul__, self.batch_size, 1) # type:ignore
            logits = self.log_probability_table().reshape(-1, *self.nvals)[(list(range(batch_total)), *index[self.batch_dims:])] # type:ignore
            logits = logits.reshape(*(*self.batch_size, *logits.size()[1:])) # type:ignore
            assert len(logits.size()) == len(self.logits.size()) - len(observation) # type:ignore
        if not is_single_observation:
            # permute the batch to the front, this may be costly
            min_obs_var_index = min(observation.keys()) + self.batch_dims # type:ignore
            permutation = list(range(len(logits.size())))
            old_batch = permutation[:self.batch_dims] # type:ignore
            new_batch = permutation[min_obs_var_index:min_obs_var_index + new_batch_dims]
            front = permutation[self.batch_dims:min_obs_var_index] # type:ignore
            back = permutation[min_obs_var_index + new_batch_dims:] # type:ignore
            permutation = old_batch + new_batch + front + back
            logits = logits.permute(*permutation)
        children_set = set(self._children)
        parents_set = set(self._parents)
        if all(k in observation for k in children_set):
            # it doesn't make sense when children are all conditioned on to treat this as
            # any sort of distribution, so make that explicit by returning a PotentialTable rather
            # than a ProbabilityTable
            return LogLinearPotentialTable(logits.size(), logits, batch_dims=self.batch_dims + new_batch_dims) # type:ignore
        else:
            new_parents = []
            new_par_index = 0
            for i in range(self.nvars): # type:ignore assumes nvars is implemented by whoever inherits this mixin
                if i not in observation and i in parents_set:
                    new_parents.append(new_par_index)
                    new_par_index += 1
            return LogLinearProbabilityTable(logits.size(), new_parents, logits, batch_dims=self.batch_dims + new_batch_dims) # type:ignore


class LogLinearProbabilityTable(LogLinearProbabilityMixin, LogLinearTableInferenceMixin, LogLinearTable, pgm.ProbabilityTable):
    def __init__(self, size, parents: list[int] | tuple[int,...], initializer: constants.TensorInitializer | torch.Tensor, requires_grad:bool=True, batch_dims=0) -> None:
        super().__init__(len(size) - batch_dims, parents, size, initializer, requires_grad=requires_grad, batch_dims=batch_dims)

    @staticmethod
    def joint_from_factors(nvars: int, nvals: list[int], factor_variables: Sequence[tuple[int, ...]], factor_functions: Sequence[pgm.PotentialTable]):
        """
        IMPORTANT: assumes the variables in the factor functions are in numerical order!
        e.g. if the set of variables range from v1 to v10, assumes the factor table
        on the set {v2 v5 v3} is always ordered with axis corresponding to v2 v3 v5.

        The sizes of the factors must respect the declared sizes encoded by nvals.
        """
        if any(factor_function.batch_size != factor_functions[0].batch_size for factor_function in factor_functions):
            raise ValueError(f"Can only join PotentialTables with the same batch_size: {[factor_function.batch_size for factor_function in factor_functions]}")

        table = None
        batch_dims = factor_functions[0].batch_dims
        batch_size = factor_functions[0].batch_size
        for factor_vars, factor_fn in zip(factor_variables, factor_functions):
            size = [1] * nvars
            local_factor = factor_fn.log_potential_table()
            for var in factor_vars:
                size[var] = nvals[var]
            local_factor = local_factor.reshape(*(*batch_size, *size))
            table = local_factor if table is None else table + local_factor
        assert table is not None
        return LogLinearProbabilityTable(table.size(), [], table, batch_dims=batch_dims)

    def expand_batch_dimensions_(self, batch_sizes: tuple[int, ...]) -> LogLinearProbabilityTable:
        """This override is just to give the type checker some more hints."""
        self.expand_batch_dimensions_meta_(batch_sizes) # type:ignore
        self._logits.data = self.logits.data.expand((*self.batch_size, *self.nvals))
        return self

    def expand_batch_dimensions(self, batch_sizes: tuple[int, ...]) -> LogLinearProbabilityTable:
        """This override is just to give the type checker some more hints."""
        table = LogLinearProbabilityTable(self.logits.size(), self._parents, self.logits.clone(), batch_dims=self.batch_dims)
        table.expand_batch_dimensions_(batch_sizes) # type:ignore
        return table

class LogLinearFeaturizedProbabilityTable(LogLinearProbabilityMixin, LogLinearTableInferenceMixin, LogLinearFeaturizedTable, pgm.ProbabilityTable):

    def __init__(self, size, parents, feature_extractor, initializer, requires_grad=True, batch_dims=0) -> None:
        super().__init__(len(size) - batch_dims, parents, size, feature_extractor, initializer, requires_grad=requires_grad, batch_dims=batch_dims)

if __name__ == "__main__":
    table = LogLinearPotentialTable((3,4,5,6), constants.TensorInitializer.CONSTANT)
    print(table.batch_dims, table.batch_size, table.logits.size())
    indices = torch.tensor([[0,1,2],[1,2,3]])
    cpt = table.condition_on({i: indices[i] for i in range(indices.size(0))})
    print(cpt.batch_dims, cpt.batch_size, cpt.logits.size())
    print(cpt.log_potential_value((torch.tensor([2,3,4]),torch.tensor([3,4,5]))).size())
    cpt = cpt.condition_on({0: torch.tensor([2,3])})
    print(cpt.batch_dims, cpt.batch_size, cpt.logits.size())
    print(cpt.log_potential_value((torch.tensor([[1,2,3],[3,4,5]]),)).size())
    other_cpt = LogLinearPotentialTable((6,7), constants.TensorInitializer.CONSTANT, batch_dims=0).expand_batch_dimensions_((3,2))
    joint = LogLinearPotentialTable.joint_from_factors(2, [6,7], [(0,), (0,1)], [cpt, other_cpt])
    print(joint.batch_dims, joint.batch_size, joint.logits.size())
