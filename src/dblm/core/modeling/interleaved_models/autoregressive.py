
import torch
from dblm.core.interfaces import pgm
from dblm.core.modeling import factor_graphs, bayesian_networks
import torch.nn as nn

class AutoregressiveInterleavedModel(factor_graphs.AutoRegressiveBayesNetMixin, factor_graphs.FactorGraph, pgm.BayesianNetwork):

    def __init__(self, z0_nvars, z0_nvals, length, encoder) -> None:
        nvars = z0_nvars + length * 2
        nvals = z0_nvals + [z0_nvars, sum(z0_nvals)] * length
        factor_variables = [list(range(z0_nvars + i + 1)) for i in range(length)]
        factor_functions = [AutoregressiveFactor(len(vars), [nvals[var] for var in vars], encoder) for vars in factor_variables] # type:ignore
        super().__init__(nvars, nvals, factor_variables, factor_functions)


class AutoregressiveFactor(nn.Module, pgm.ProbabilityTable):

    def __init__(self, nvars, nvals, encoder) -> None:
        self.call_super_init = True
        super().__init__()
        self.transformer = encoder
        self._nvars = nvars
        self._nvals = nvals
        self._observation = None

    def condition_on(self, observation):
        if observation.keys() != set(range(self.nvars - 1)):
            raise ValueError(f"AutoregressiveFactor has to condition on the entire prefix, expected {list(range(self.nvars - 1))} got {sorted(observation.keys())}")
        self._observation = observation
        return self

    def probability_table(self):
        return self.log_probability_table().exp() # type:ignore

    def log_probability_table(self):
        if self._observation is None:
            raise NotImplementedError()
        input_ids = torch.stack([self._observation[i] for i in range(self.nvars - 1)], dim=1) # type:ignore
        output = self.transformer(input_ids)
        logits = output.logits[:, -1 ,:]
        return logits.log_softmax(-1)

    def to_bayesian_network(self):
        nvars = self.nvars
        nvals = self.nvals
        factor_vars = [list(range(nvars))]
        parents = [self._parents]
        children = [self._children]
        factor_functions = [self]
        topo_order = [0]
        return bayesian_networks.BayesianNetwork(nvars, nvals, factor_vars, parents, children, factor_functions, topo_order) # type:ignore

    @property
    def parent_indices(self):
        return tuple(range(self.nvars - 1))

    @property
    def child_indices(self):
        return (self.nvars-1,)
