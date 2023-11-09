from __future__ import annotations
import code
import math
from typing import Sequence
import torch
from dblm.core.inferencers import belief_propagation
from dblm.core.interfaces import distribution, pgm
from dblm.core.modeling import bayesian_networks, factor_graphs
import torch.nn as nn

class AutoregressiveXModel(nn.Module, pgm.ProbabilityTable):

    def __init__(self, z0_nvars, z0_nvals, length, transformer, x_vocab_indices, bos_vocab_index) -> None:
        self.call_super_init = True
        super().__init__()
        self.z0_nvars = z0_nvars
        self.z0_nvals = z0_nvals
        self._nvars = z0_nvars + length
        self._nvals = z0_nvals + [sum(z0_nvals)] * length
        self.length = length
        self.transformer = transformer
        self.x_vocab_indices = x_vocab_indices
        self.bos_vocab_index = bos_vocab_index
        self.register_buffer("x_mask", torch.ones(self.transformer.config.vocab_size, requires_grad=False).fill_(-math.inf))
        self.x_mask[self.x_vocab_indices] = 0 # type:ignore

    def condition_on(self, observation: dict[int, int] | dict[int, torch.Tensor]) -> pgm.PotentialTable:
        return ConditionalAutoregressiveXModel(observation,
                                                self.z0_nvars,
                                                self.z0_nvals,
                                                self.length,
                                                self.transformer,
                                                self.x_vocab_indices,
                                                self.bos_vocab_index)

    def to_factor_graph_model(self) -> pgm.FactorGraphModel:
        nvars = self.nvars # type:ignore assumes whoever is using the mixin is implementing MultivariateFunction
        nvals = self.nvals # type:ignore assumes whoever is using the mixin is implementing MultivariateFunction
        factor_vars = [list(range(nvars))]
        factor_functions = [self]
        return factor_graphs.FactorGraph(nvars, nvals, factor_vars, factor_functions) # type: ignore

    def to_bayesian_network(self) -> pgm.BayesianNetwork:
        nvars = self.nvars # type:ignore assumes whoever is using the mixin is implementing MultivariateFunction
        nvals = self.nvals # type:ignore assumes whoever is using the mixin is implementing MultivariateFunction
        factor_vars = [list(range(nvars))]
        parents = [self.parent_indices()]
        children = [self.child_indices()]
        factor_functions = [self]
        topo_order = [0]
        return bayesian_networks.BayesianNetwork(nvars, nvals, factor_vars, parents, children, factor_functions, topo_order) # type:ignore

    def expand_batch_dimensions(self, batch_sizes: tuple[int, ...]) -> pgm.Batchable:
        other = AutoregressiveXModel(self.z0_nvars,
                                                self.z0_nvals,
                                                self.length,
                                                self.transformer,
                                                self.x_vocab_indices,
                                                self.bos_vocab_index)
        other.expand_batch_dimensions_(batch_sizes=batch_sizes)
        return other

    def expand_batch_dimensions_(self, batch_sizes: tuple[int, ...]) -> None:
        self.expand_batch_dimensions_meta_(batch_sizes)

    def parent_indices(self) -> tuple[int, ...]:
        return tuple(range(self.z0_nvars))

    def child_indices(self) -> tuple[int, ...]:
        return tuple(range(self.z0_nvars, self.nvars))

    def unnormalized_probability(self, assignment: tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError()

    def energy(self, assignment: tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError()

    def potential_table(self) -> torch.Tensor:
        raise NotImplementedError()

    def log_potential_table(self) -> torch.Tensor:
        raise NotImplementedError()

    def potential_value(self, assignment) -> torch.Tensor:
        raise NotImplementedError()

    def log_potential_value(self, assignment) -> torch.Tensor:
        raise NotImplementedError()

    def marginalize_over(self, variables: Sequence[int]) -> pgm.PotentialTable:
        raise NotImplementedError()

    def probability(self, assignment: tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError()

    def probability_table(self) -> torch.Tensor:
        raise NotImplementedError()

    def log_probability_table(self) -> torch.Tensor:
        raise NotImplementedError()

    def log_probability(self, assignment: tuple[int, ...]) -> torch.Tensor:
        raise NotImplementedError()


class ConditionalAutoregressiveXModel(AutoregressiveXModel):

    def __init__(self, observation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation = observation

    def sample(self):
        z0_offsets = [sum(self.z0_nvals[:i]) for i in range(self.z0_nvars)]
        input_ids = torch.stack([self.observation[i] + z0_offsets[i] for i in range(self.z0_nvars)] + [torch.ones_like(self.observation[0]).fill_(self.bos_vocab_index)], dim=1)
        past_key_values = None
        for _ in range(self.length):
            input = input_ids[:,-1:] if past_key_values is not None else input_ids
            output = self.transformer(input, past_key_values=past_key_values)
            logits = output.logits[:,-1,:]
            past_key_values = output.past_key_values
            logits = logits + self.x_mask # type:ignore
            new_input = torch.multinomial(nn.functional.softmax(logits, dim=-1), num_samples=1) # type:ignore
            input_ids = torch.cat([input_ids, new_input], dim=1)
        return input_ids[:, self.z0_nvars:]

class LatentWorldAutoregressiveXModel(factor_graphs.FactorGraph, distribution.MarginalMixin):

    def __init__(self, pgmz0: pgm.FactorGraphModel, pgmxt: AutoregressiveXModel):
        fg = factor_graphs.FactorGraph.join(pgmz0, pgmxt.to_factor_graph_model(), shared=dict(enumerate(range(pgmz0.nvars))))
        self.call_super_init = True
        super().__init__(fg.nvars, fg.nvals, fg.factor_variables(), fg.factor_functions())
        self.pgmz0 = pgmz0
        self.pgmxt = pgmxt
        self.bp = belief_propagation.FactorGraphBeliefPropagation()

    def marginal_probability(self, assignment: Sequence[tuple[int, int]]):
        return super().marginal_probability(assignment)

    def log_marginal_probability(self, assignment: Sequence[tuple[int, int]]):
        self.forward_propagation([None] * self.pgmz0.nvars, assignment) # type:ignore
        return super().log_marginal_probability(assignment)

    def forward_propagation(self, z0_backward_messages: list[pgm.PotentialTable], assignment: Sequence[tuple[int, int]]):
        z0_marginals = self.latent_variable_inference(z0_backward_messages)
        self.forward_x_propagation(z0_marginals, assignment) # type:ignore

    def forward_x_propagation(self, z0_marginals: list[pgm.ProbabilityTable], assignment: Sequence[tuple[int, int]]):
        log_marginals = [m.log_probability_table() for m in z0_marginals]
        assert False

    def latent_variable_inference(self, z0_backward_messages: list[pgm.PotentialTable]):
        if all(m is None for m in z0_backward_messages):
            return self.bp.inference(self.pgmz0, dict(), list(range(self.pgmz0.nvars))).query_marginals
        else:
            batch_size = self.check_batch_size(z0_backward_messages)
            model = factor_graphs.FactorGraph(
                self.pgmz0.nvars,
                self.pgmz0.nvals,
                self.pgmz0.factor_variables() + [(i,) for i, m in enumerate(z0_backward_messages) if m is not None],
                [f.expand_batch_dimensions(batch_size) for f in self.pgmz0.factor_functions()]  # type:ignore
                + [m for m in z0_backward_messages if m is not None] # type:ignore
            )
            return self.bp.inference(model, dict(), list(range(self.pgmz0.nvars))).query_marginals

    def check_batch_size(self, messages):
        batch_size = None
        for m in messages:
            if m is None:
                continue
            if batch_size is None:
                batch_size = m.batch_size
            else:
                assert batch_size == m.batch_size
        return batch_size
