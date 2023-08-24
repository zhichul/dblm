from __future__ import annotations
from dblm.core.interfaces import pgm
from dblm.core.modeling import constants, factor_graphs, probability_tables, switching_tables
import torch.nn as nn
import torch

class NoisyMixture(factor_graphs.AutoRegressiveBayesNetMixin, factor_graphs.FactorGraph, pgm.BayesianNetwork):

    def __init__(self, nvars:int, nvals:list[int], noise=constants.DiscreteNoise.UNIFORM, mixture_ratio=(4.0,1.0)) -> None:
        self._nvars = nvars * 4 # nvars, nvars noise, nvars switch, nvars out
        self._nvals = nvals + nvals + [2] * nvars + nvals

        factor_variables:list[tuple[int,...]] = [None] * (nvars * 3) #type:ignore
        factor_functions:list[pgm.ProbabilityTable] = [None] * (nvars * 3) # type:ignore

        noise_switch = probability_tables.LogLinearProbabilityTable((2,), [], nn.Parameter(torch.tensor(mixture_ratio).log(), requires_grad=False))

        for var in range(nvars):
            # add noise (noise is currently independent of the true value of the variable)
            if noise == constants.DiscreteNoise.UNIFORM:
                noise_distribution = probability_tables.LogLinearProbabilityTable((nvals[var],), [], constants.TensorInitializer.CONSTANT)
            else:
                raise NotImplementedError()
            factor_variables[var] = (nvars * 1 + var,)
            factor_functions[var] = noise_distribution

            # add switch
            factor_variables[var + nvars * 1] = (nvars * 2 + var,)
            factor_functions[var + nvars * 1] = noise_switch

            # add out
            factor_variables[var + nvars * 2] = (var, nvars * 1 + var, nvars * 2 + var, nvars * 3 + var)
            factor_functions[var + nvars * 2] = switching_tables.SwitchingTable(nvars=2, nvals=[nvals[var], nvals[var]], mode=constants.SwitchingMode.MIXTURE)

        super().__init__(self._nvars, self._nvals, factor_variables, factor_functions) # type:ignore

