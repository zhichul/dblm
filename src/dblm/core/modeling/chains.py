import json
import math
import os

import torch
from dblm.core.interfaces import pgm
from dblm.core.modeling import factor_graphs, probability_tables
from dblm.core.modeling import constants
from dblm.core import graph
import torch.nn as nn

class FixedLengthDirectedChain(nn.Module, pgm.BayesianNetwork):
    #TODO this is actully not locally normalized due to the last factor so is technically not a BayesianNetwork

    def __init__(self, nvars: int, nvals: int, initializer: constants.TensorInitializer, chain=None, requires_grad=True) -> None: # type: ignore
        super().__init__()
        self._nvars = nvars
        self._nvals = [nvals] * nvars
        self._graph = chain if chain is not None else graph.Chain(nvars)
        initial_factor = probability_tables.LogLinearProbabilityTable((nvals,), [], initializer, requires_grad=requires_grad)
        transition_factor = probability_tables.LogLinearProbabilityTable((nvals, nvals), [0], initializer, requires_grad=requires_grad)
        final_factor = probability_tables.LogLinearProbabilityTable((nvals,), [], initializer, requires_grad=requires_grad)

        self._factor_variables : list[tuple[int]] = [(0,)]
        self._factor_functions = nn.ModuleList([initial_factor])

        for edge in self._graph.edges:
            n1, n2 = edge.nodes
            var1, var2 = min(n1.id, n2.id), max(n1.id, n2.id)
            self._factor_variables.append((var1, var2)) # type:ignore
            self._factor_functions.append(transition_factor)
        self._factor_variables.append((nvars-1,))
        self._factor_functions.append(final_factor)

    # ProbabilisticGraphicalModel
    def graph(self):
        return self._graph

    def to_probability_table(self) -> pgm.ProbabilityTable:
        return probability_tables.LogLinearProbabilityTable.joint_from_factors(self._nvars,
                                                                    self._nvals,
                                                                    self._factor_variables,
                                                                    self._factor_functions) # type: ignore
    def to_potential_table(self) -> pgm.PotentialTable:
        return probability_tables.LogLinearPotentialTable.joint_from_factors(self._nvars,
                                                                    self._nvals,
                                                                    self._factor_variables,
                                                                    self._factor_functions) # type: ignore
    def to_factor_graph_model(self) -> pgm.FactorGraphModel:
        return factor_graphs.FactorGraph(self._nvars, self._nvals, self._factor_variables, self._factor_functions) # type: ignore

    def fix_variables(self, observation):
        raise NotImplementedError()

    # BayesianNetwork
    def local_distributions(self):
        raise NotImplementedError()

    def local_variables(self):
        raise NotImplementedError()

    def local_parents(self):
        raise NotImplementedError()

    def local_children(self):
        raise NotImplementedError()

    def topological_order(self):
        raise NotImplementedError()

    # Self
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        self._graph.save(os.path.join(directory, constants.GRAPH_FILE))
        with open(os.path.join(directory, constants.FACTORS_FILE), "w") as f:
            json.dump({
                    constants.FACTOR_VARIABLES: self._factor_variables,
                    constants.FACTOR_FUNCTIONS: [table.log_potential_table().tolist() for table in self._factor_functions], # type: ignore
                    constants.NUMBER_OF_VARIABLES: self._nvars,
                    constants.NUMBER_OF_VALUES: self._nvals
                    }, f)

    @staticmethod
    def load(directory):
        with open(os.path.join(directory, constants.FACTORS_FILE)) as f:
            factors_dict = json.load(f)
        nvars = factors_dict[constants.NUMBER_OF_VARIABLES]
        nvals = factors_dict[constants.NUMBER_OF_VARIABLES]
        chain = graph.Graph.load(os.path.join(directory, constants.GRAPH_FILE))
        fc = FixedLengthDirectedChain(nvars, nvals, constants.TensorInitializer.CONSTANT, chain=chain)
        if fc._factor_variables != [tuple(vars) for vars in factors_dict[constants.FACTOR_VARIABLES]]:
            raise AssertionError
        for factor_fn, table in zip(fc._factor_functions, factors_dict[constants.FACTOR_FUNCTIONS]):
            factor_fn.logits.data = torch.tensor(table)
        return fc

    # LocallyNormalizedDistribution
    def likelihood_function(self, assignment):
        raise NotImplementedError()

    def log_likelihood_function(self, assignment):
        raise NotImplementedError()

if __name__ == "__main__":
    chain = FixedLengthDirectedChain(2, 3, constants.TensorInitializer.CONSTANT)
    chain._factor_functions[-1].logits.data = torch.tensor([[100.,-math.inf,-math.inf],[10,-math.inf,-math.inf],[1,-math.inf,-math.inf]])
    table = chain.to_probability_table().probability_table()
    print(table)
    factor_graph = chain.to_factor_graph_model()
