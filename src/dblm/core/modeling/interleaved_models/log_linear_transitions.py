from __future__ import annotations
import math

from dblm.core.featurizers import x_term_frequency
from dblm.core.inferencers import belief_propagation
from dblm.core.interfaces import pgm
from dblm.core.modeling import bayesian_networks, constants, factor_graphs, markov_networks, probability_tables, switching_tables
import torch.nn as nn
import torch

from dblm.core.samplers import ancestral
from dblm.utils import seeding


class LogLinearLatentMarkovTransitionBase(nn.Module):

    def __init__(self, nstate:int, vocab_size:int, order:int=1, initializer=constants.TensorInitializer.UNIFORM) -> None:
        self.call_super_init = True
        super().__init__()
        self.featurizer = x_term_frequency.XTermFrequency(vocab_size)
        self.vocab_size = vocab_size
        self.nstate = nstate
        self.order = order
        self.layer = nn.Linear(self.featurizer.out_features, (nstate ** (order + 1)), bias=True) # bias is for modeling transition directly
        if initializer == constants.TensorInitializer.CONSTANT:
            nn.init.zeros_(self.layer.weight) # intialize at uniform
        elif initializer == constants.TensorInitializer.UNIFORM:
            nn.init.uniform_(self.layer.weight,0,0.1) # intialize at uniform
        elif isinstance(initializer, constants.UniformlyRandomInitializer):
            nn.init.uniform_(self.layer.weight, initializer.min, initializer.max)
            nn.init.uniform_(self.layer.bias, initializer.min, initializer.max)
        elif isinstance(initializer, constants.GaussianInitializer):
            if initializer.min is None and initializer.max is None:
                nn.init.normal_(self.layer.weight, initializer.mean, initializer.std)
                nn.init.normal_(self.layer.bias, initializer.mean, initializer.std)
            else:
                nn.init.trunc_normal_(self.layer.weight, initializer.mean, initializer.std, initializer.min, initializer.max) # type: ignore
                nn.init.trunc_normal_(self.layer.bias, initializer.mean, initializer.std, initializer.min, initializer.max) # type: ignore
        else:
            raise NotImplementedError()
        nn.init.zeros_(self.layer.bias) # intialize at uniform think of this as the shape of (nstate, ..., nstate) each row corresponding to a ngram prefix

    def logits(self, x_assignment):
        batch_dims = len(x_assignment.size()) - 1
        features = self.featurizer(x_assignment)
        transition = self.layer(features)
        return transition.reshape(*(x_assignment.size()[:batch_dims] + tuple(self.nstate for _ in range(self.order + 1))))

    def generate_wrapper(self, ntokens):
        return LogLinearLatentMarkovTransitionWrapper(self, [self.nstate] * (self.order + 1), ntokens)

class LogLinearLatentMarkovTransitionWrapper(nn.Module, pgm.ProbabilityTable):
    """This class is intentionally not implemented, as it should never be directly used, but rather should be used in fixed_variable form."""

    error = NotImplementedError("This class is only a wrapper and cannot be evaluated. To evaluate first call fix_variables with assignments to all x's.")

    def __init__(self, base: LogLinearLatentMarkovTransitionBase, conditional_size, ntokens, batch_dims=0, batch_size=tuple()):
        self.call_super_init = True
        super().__init__()
        self.base = base
        self.conditional_size = conditional_size
        self._nvars = ntokens + base.order + 1
        self._nvals = [base.vocab_size] * (ntokens - base.order) + [base.nstate, base.vocab_size] * (base.order) + [base.nstate]
        self._x_indices = list(range(ntokens - base.order)) + [ntokens - base.order + i + 1 for i in range(base.order)]
        self._x_indices_set = set(self._x_indices)
        self._map_to_non_x = {j:i for i, j in enumerate([i for i in range(self._nvars) if i not in self._x_indices_set])}
        self._batch_dims = batch_dims
        self._batch_size = batch_size
        self._ntokens = ntokens

    def expand_batch_dimensions_(self, batch_sizes: tuple[int, ...]) -> LogLinearLatentMarkovTransitionWrapper:
        super().expand_batch_dimensions_meta_(batch_sizes)
        return self

    def expand_batch_dimensions(self, batch_sizes: tuple[int, ...]) -> LogLinearLatentMarkovTransitionWrapper:
        table = LogLinearLatentMarkovTransitionWrapper(self.base, self.conditional_size, self._ntokens, batch_dims=self.batch_dims, batch_size=self.batch_size)
        table.expand_batch_dimensions_meta_(batch_sizes)
        return table

    def condition_on(self, observation: dict[int, int] | dict[int, torch.Tensor], cartesian=None, device="cpu"):
        if not set(observation.keys()).issuperset(self._x_indices_set):
            print(observation, self._x_indices, list(range(self.nvars)))
            raise NotImplementedError("For efficiency, we only allow LogLinearLatentMarkovTransition fully conditioned on X")
        if isinstance(next(observation.values().__iter__()), int):
            x_observation = torch.tensor([observation[i] for i in self._x_indices], device=device)
        else:
            x_observation = torch.stack([observation[i] for i in self._x_indices], dim=1) # type:ignore
        logits = self.base.logits(x_observation)
        if cartesian:
            logits = logits.expand((*self.batch_size, *logits.size()))
        o = probability_tables.LogLinearProbabilityTable(logits.size(), list(range(self.base.order)), logits, batch_dims=len(x_observation.size())-1)
        if len(observation.keys()) == len(self._x_indices):
            return o
        else:
            sub_observation = {self._map_to_non_x[k]:v for k,v in observation.items() if k not in self._x_indices_set}
            o = o.condition_on(sub_observation, cartesian=False) # type:ignore
            return o

    def potential_table(self) -> torch.Tensor:
        raise self.error

    def log_potential_table(self) -> torch.Tensor:
        raise self.error

    def potential_value(self, assignment) -> torch.Tensor:
        return self.condition_on(dict(enumerate(assignment))).potential_table() # type:ignore

    def log_potential_value(self, assignment) -> torch.Tensor:
        return self.condition_on(dict(enumerate(assignment))).log_potential_table() # type:ignore

    def marginalize_over(self, variables):
        raise self.error

    def probability_table(self):
        raise self.error

    def log_probability_table(self):
        raise self.error

    def to_factor_graph_model(self):
        raise self.error

    def to_bayesian_network(self):
        raise self.error

    def energy(self, assignment):
        return self.condition_on(dict(enumerate(assignment))).log_potential_table().reshape(-1).item() # type:ignore this fixation will first create a probability table, then get the assignment's probability, represented as a potential value

    def unnormalized_probability(self, assignment):
        raise self.log_probability(assignment).exp() # type:ignore

    def parent_indices(self):
        return tuple(range(self.nvars - 1))

    def child_indices(self):
        return (self.nvars - 1,)

class FixedLengthDirectedChainWithLogLinearTermFrequencyTransitionAndDeterministicEmission(factor_graphs.AutoRegressiveBayesNetMixin, factor_graphs.FactorGraph, pgm.BayesianNetwork):

    def __init__(self, z0_nvars, z0_nvals, length: int, initializer=constants.TensorInitializer.UNIFORM, transition=None, requires_grad=True):
        nvars = z0_nvars + length * 2 # z0 z0 z0 z0 z0 z0 z0 z1 x1 z2 x2 z3 x3 ... zl xl
        nvals = z0_nvals + [z0_nvars, sum(z0_nvals)] * length
        transition = LogLinearLatentMarkovTransitionBase(z0_nvars, sum(z0_nvals), order=1, initializer=initializer) if transition is None else transition

        initial_factor = probability_tables.LogLinearProbabilityTable((z0_nvars,), [], initializer, requires_grad=requires_grad)

        factor_variables : list[tuple[int]] = [(z0_nvars,)]
        factor_functions = nn.ModuleList([initial_factor])

        switch = switching_tables.SwitchingTable(z0_nvars, z0_nvals)

        # TODO need to add BOS nodes for order > 1
        for i in range(length):
            factor_variables.append((*tuple(range(z0_nvars)), z0_nvars + i * 2, z0_nvars + i * 2 + 1)) # type:ignore deterministic emissions for x: z0_nvars zi xi
            factor_functions.append(switch) # deterministic emissions for x: z0_nvars zi xi
            if i < length - 1:
                factor_variables.append((*sum(([z0_nvars + j * 2, z0_nvars + j * 2 + 1] if j >= i else [z0_nvars + j*2+1] for j in range(i+1)), []), z0_nvars + (i+1) * 2)) # type:ignore conditional transitinos: x<i zi xi zi+1
                factor_functions.append(transition.generate_wrapper(i+1)) # conditional transitinos: x<i zi xi zi+1
        super().__init__(nvars, nvals, factor_variables, factor_functions) # type: ignore
        self.transition = transition
        self.z0_nvars = z0_nvars
        self.z0_nvals = z0_nvals
        self.length = length
        self.__requires_grad = requires_grad

    # Bayes Net interface
    def local_distributions(self):
        return self._factor_functions

    def topological_order(self):
        return list(range(len(self._factor_functions)))

    def local_variables(self):
        return self._factor_variables

    def local_parents(self) -> list[tuple[int,...]]:
        return [vars[:-1] for vars in self._factor_variables]

    def local_children(self) -> list[tuple[int,...]]:
        return [vars[-1:] for vars in self._factor_variables]

    def condition_on(self, observation: dict[int, int]):
        # this overrides the default fix variables that turns everything into potential tables which will not support the bayesnet interface anymore
        model = FixedLengthDirectedChainWithLogLinearTermFrequencyTransitionAndDeterministicEmission(self.z0_nvars, self.z0_nvals, self.length, transition=self.transition, requires_grad=self.__requires_grad)
        # model.transition = self.transition
        fv = self.conditional_factor_variables(observation)
        ff = self.conditional_factor_functions(observation)
        model._factor_variables, model._factor_functions = fv, nn.ModuleList(ff) # type:ignore
        return model

    def parent_indices(self) -> tuple[int,...]:
        raise NotImplementedError()

    def child_indices(self) -> tuple[int,...]:
        raise NotImplementedError()

class FixedLengthDirectedChainWithLogLinearTermFrequencyTransitionAndDeterministicEmissionWithNoiseInZ(factor_graphs.AutoRegressiveBayesNetMixin, factor_graphs.FactorGraph, pgm.BayesianNetwork):

    def __init__(self, z0_nvars: int, z0_nvals: list[int], length: int, initializer=constants.TensorInitializer.UNIFORM, transition=None, noise=constants.DiscreteNoise.UNIFORM, separate_noise_distribution_per_state=True, mixture_ratio=(4.0,1.0), requires_grad=True):
        nvars = z0_nvars + length * 5 # z0 z0 z0 z0 z0 z0 z0 z1 z1' z1s z1o x1 ... zl zl' zls zlo xl
        nvals = z0_nvals + [z0_nvars, z0_nvars, 2, z0_nvars, sum(z0_nvals)] * length

        transition = LogLinearLatentMarkovTransitionBase(z0_nvars, sum(z0_nvals), order=1, initializer=initializer) if transition is None else transition
        initial_factor = probability_tables.LogLinearProbabilityTable((z0_nvars,), [], initializer, requires_grad=requires_grad)

        factor_variables : list[tuple[int, ...]] = [(z0_nvars,)]
        factor_functions = nn.ModuleList([initial_factor])

        if separate_noise_distribution_per_state:
            if noise == constants.DiscreteNoise.UNIFORM:
                noise_distribution = probability_tables.LogLinearProbabilityTable((z0_nvars, z0_nvars), [0], constants.TensorInitializer.CONSTANT)
            else:
                raise NotImplementedError()
        else:
            if noise == constants.DiscreteNoise.UNIFORM:
                noise_distribution = probability_tables.LogLinearProbabilityTable((z0_nvars,), [], constants.TensorInitializer.CONSTANT)
            else:
                raise NotImplementedError()
        noise_switch = probability_tables.LogLinearProbabilityTable((2,), [], nn.Parameter(torch.tensor(mixture_ratio).log(), requires_grad=False))

        zt_mixture = switching_tables.SwitchingTable(nvars=2, nvals=[z0_nvars, z0_nvars], mode=constants.SwitchingMode.MIXTURE)
        zt_emit = switching_tables.SwitchingTable(z0_nvars, z0_nvals)
        for i in range(length):
            # add the noisy z', possibly depending on z
            factor_variables.append((z0_nvars + i * 5, z0_nvars + i * 5 + 1) if separate_noise_distribution_per_state else (z0_nvars + i * 5 + 1,)) # noisy z: z0_nvars zi (optional) zi'
            factor_functions.append(noise_distribution) # noisy z: z0_nvars zi (optional) zi'

            # add the switch zs
            factor_variables.append((z0_nvars + i * 5 + 2,))
            factor_functions.append(noise_switch)

            # add the output zo that depends on z, z' and zs
            factor_variables.append(tuple(z0_nvars + i * 5 + j for j in range(4)))
            factor_functions.append(zt_mixture)

            # add the x which depends on z0 and zo
            factor_variables.append((*tuple(range(z0_nvars)), z0_nvars + i * 5 + 3, z0_nvars + i * 5 + 4)) # deterministic emissions for x: z0_nvars zio xi
            factor_functions.append(zt_emit) # deterministic emissions for x: z0_nvars zio xi
            if i < length - 1:
                factor_variables.append((*sum(([z0_nvars + j * 5, z0_nvars + j * 5 + 4] if j >= i else [z0_nvars + j * 5 + 4] for j in range(i+1)), []), z0_nvars + (i+1) * 5)) # conditional transitinos: x<i zi xi zi+1
                factor_functions.append(transition.generate_wrapper(i+1)) # conditional transitinos: x<i zi xi zi+1
        super().__init__(nvars, nvals, factor_variables, factor_functions) # type: ignore
        self.transition = transition
        self.z0_nvars = z0_nvars
        self.z0_nvals = z0_nvals
        self.length = length
        self.noise = noise
        self.separate_noise_distribution_per_state = separate_noise_distribution_per_state
        self.__requires_grad = requires_grad

    def condition_on(self, observation: dict[int, int]):
        # this overrides the default fix variables that turns everything into potential tables which will not support the bayesnet interface anymore
        model = FixedLengthDirectedChainWithLogLinearTermFrequencyTransitionAndDeterministicEmissionWithNoiseInZ(self.z0_nvars, self.z0_nvals, self.length, transition=self.transition, noise=self.noise, separate_noise_distribution_per_state=self.separate_noise_distribution_per_state, requires_grad=self.__requires_grad)
        fv = self.conditional_factor_variables(observation)
        ff = self.conditional_factor_functions(observation)
        model._factor_variables, model._factor_functions = fv, nn.ModuleList(ff) # type:ignore
        return model

class FixedLengthDirectedChainWithLogLinearTermFrequencyTransitionAndDeterministicEmissionWithNoiseInZOneCopyForEveryToken(factor_graphs.AutoRegressiveBayesNetMixin, factor_graphs.FactorGraph, pgm.BayesianNetwork):

    def __init__(self, z0_nvars: int, z0_nvals: list[int], length: int, initializer=constants.TensorInitializer.UNIFORM, transition=None, noise=constants.DiscreteNoise.UNIFORM, separate_noise_distribution_per_state=True, mixture_ratio=(4, 1), requires_grad=True):
        nvars = z0_nvars * length + length * 5 # z0 z0 z0 z0 z0 z0 z0 / z0 z0 z0 z0 z0 z0 / ... z1 z1' z1s z1o x1 ... zl zl' zls zlo xl
        nvals = z0_nvals * length + [z0_nvars, z0_nvars, 2, z0_nvars, sum(z0_nvals)] * length

        transition = LogLinearLatentMarkovTransitionBase(z0_nvars, sum(z0_nvals), order=1, initializer=initializer) if transition is None else transition
        initial_factor = probability_tables.LogLinearProbabilityTable((z0_nvars,), [], initializer, requires_grad=requires_grad)

        factor_variables : list[tuple[int, ...]] = [(z0_nvars*length,)]
        factor_functions = nn.ModuleList([initial_factor])

        if separate_noise_distribution_per_state:
            if noise == constants.DiscreteNoise.UNIFORM:
                noise_distribution = probability_tables.LogLinearProbabilityTable((z0_nvars, z0_nvars), [0], constants.TensorInitializer.CONSTANT)
            else:
                raise NotImplementedError()
        else:
            if noise == constants.DiscreteNoise.UNIFORM:
                noise_distribution = probability_tables.LogLinearProbabilityTable((z0_nvars,), [], constants.TensorInitializer.CONSTANT)
            else:
                raise NotImplementedError()
        noise_switch = probability_tables.LogLinearProbabilityTable((2,), [], nn.Parameter(torch.tensor(mixture_ratio).log(), requires_grad=False))

        zt_mixture = switching_tables.SwitchingTable(nvars=2, nvals=[z0_nvars, z0_nvars], mode=constants.SwitchingMode.MIXTURE)
        zt_emit = switching_tables.SwitchingTable(z0_nvars, z0_nvals)
        for i in range(length):
            # add the noisy z', possibly depending on z
            factor_variables.append((z0_nvars * length + i * 5, z0_nvars * length + i * 5 + 1) if separate_noise_distribution_per_state else (z0_nvars * length + i * 5 + 1,)) # noisy z: z0_nvars zi (optional) zi'
            factor_functions.append(noise_distribution) # noisy z: z0_nvars zi (optional) zi'

            # add the switch zs
            factor_variables.append((z0_nvars * length + i * 5 + 2,))
            factor_functions.append(noise_switch)

            # add the output zo that depends on z, z' and zs
            factor_variables.append(tuple(z0_nvars * length + i * 5 + j for j in range(4)))
            factor_functions.append(zt_mixture)

            # add the x which depends on z0 and zo
            factor_variables.append((*tuple(range(i * z0_nvars, (i+1) * z0_nvars)), z0_nvars * length + i * 5 + 3, z0_nvars * length + i * 5 + 4)) # deterministic emissions for x: z0_nvars zio xi
            factor_functions.append(zt_emit) # deterministic emissions for x: z0_nvars zio xi
            if i < length - 1:
                factor_variables.append((*sum(([z0_nvars * length  + j * 5, z0_nvars * length  + j * 5 + 4] if j >= i else [z0_nvars * length  + j * 5 + 4] for j in range(i+1)), []), z0_nvars * length  + (i+1) * 5)) # conditional transitinos: x<i zi xi zi+1
                factor_functions.append(transition.generate_wrapper(i+1)) # conditional transitions: x<i zi xi zi+1
        super().__init__(nvars, nvals, factor_variables, factor_functions) # type: ignore
        self.transition = transition
        self.z0_nvars = z0_nvars
        self.z0_nvals = z0_nvals
        self.length = length
        self.noise = noise
        self.separate_noise_distribution_per_state = separate_noise_distribution_per_state
        self.__requires_grad = requires_grad

    def condition_on(self, observation: dict[int, int]):
        # this overrides the default fix variables that turns everything into potential tables which will not support the bayesnet interface anymore
        model = FixedLengthDirectedChainWithLogLinearTermFrequencyTransitionAndDeterministicEmissionWithNoiseInZOneCopyForEveryToken(self.z0_nvars, self.z0_nvals, self.length, transition=self.transition, noise=self.noise, separate_noise_distribution_per_state=self.separate_noise_distribution_per_state, requires_grad=self.__requires_grad)
        fv = self.conditional_factor_variables(observation)
        ff = self.conditional_factor_functions(observation)
        model._factor_variables, model._factor_functions = fv, nn.ModuleList(ff) # type:ignore
        return model

if __name__ == "__main__":
    z0_nvars = 3
    z0_nvals = 2
    time = 5
    # basice inference over zs and xt+1
    z0 = markov_networks.TreeMRF(z0_nvars, z0_nvals, constants.TensorInitializer.CONSTANT)
    ztxt = FixedLengthDirectedChainWithLogLinearTermFrequencyTransitionAndDeterministicEmission(z0_nvars, [z0_nvals]*z0_nvars, time, constants.TensorInitializer.CONSTANT)
    ztxt.transition.layer.bias.data = torch.tensor([[1,-math.inf,1],[1,1,-math.inf],[-math.inf, 1, 1]]).reshape(-1)
    model = factor_graphs.FactorGraph.join(z0.to_factor_graph_model(), ztxt, {0:0, 1:1, 2:2})
    try:
        model.unnormalized_probability((0,0,0,1,2,0,0,2,4,2,4))
    except NotImplementedError:
        print("correctly refused to evaluate...")
    print(model.factor_variables())
    bp = belief_propagation.FactorGraphBeliefPropagation()
    inference_results = bp.inference(model, {4:2,6:0,8:4,10:2}, [0,1,2,3,5,7,9,11,12])
    for table in inference_results.query_marginals:
        print(table.probability_table())

    # try sampling
    seeding.seed(42)
    z0 = markov_networks.TreeMRF(z0_nvars, z0_nvals, constants.TensorInitializer.CONSTANT).to_probability_table().to_bayesian_network()
    ztxt = FixedLengthDirectedChainWithLogLinearTermFrequencyTransitionAndDeterministicEmission(z0_nvars, [z0_nvals]*z0_nvars, time, constants.TensorInitializer.CONSTANT)
    ztxt.transition.layer.bias.data = torch.tensor([[1,-math.inf,1],[1,1,-math.inf],[-math.inf, 1, 1]]).reshape(-1)
    model = bayesian_networks.BayesianNetwork.join(z0, ztxt, {0:0, 1:1, 2:2})
    ancestral_sampler = ancestral.AncestralSamplerWithPotentialTables()
    print(model.local_parents())
    print(model.local_children())
    print(model.local_distributions())
    print(model.topological_order())
    samples = ancestral_sampler.sample(500, model)
    print((samples[:, 3] == 0).sum(), (samples[:, 3] == 1).sum(), (samples[:, 3] == 2).sum())
