
from __future__ import annotations
from dblm.core import graph
from dblm.core.interfaces import pgm
from dblm.core.modeling import factor_graphs, probability_tables
from dblm.core.modeling import constants
import torch.nn as nn
import json
import os

import torch

class TreeMRF(nn.Module, pgm.MarkovRandomField):

    def __init__(self, nvars: int, nvals: int | list[int], initializer: constants.TensorInitializer, tree: graph.Graph=None, requires_grad=True) -> None: # type: ignore
        self.call_super_init = True
        super().__init__()
        if isinstance(nvals, int):
            nvals = [nvals] * nvars
        self._nvars = nvars
        self._nvals = nvals
        self._graph = graph.random_labeled_tree(nvars) if tree is None else tree
        self._factor_variables: list[tuple[int]] = []
        self._factor_functions = nn.ModuleList()
        for edge in self._graph.edges:
            n1, n2 = edge.nodes
            var1, var2 = min(n1.id, n2.id), max(n1.id, n2.id)
            val1, val2 = (nvals[var1], nvals[var2]) # type: ignore
            self._factor_variables.append((var1, var2)) # type: ignore
            self._factor_functions.append(probability_tables.LogLinearPotentialTable((val1, val2), initializer, requires_grad=requires_grad))

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

    def condition_on(self, observation):
        raise NotImplementedError()

    # MarkovRandomField
    def local_potentials(self):
        return self._factor_functions

    def local_variables(self):
        return self._factor_variables

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
        nvals = factors_dict[constants.NUMBER_OF_VALUES]
        tree = graph.Graph.load(os.path.join(directory, constants.GRAPH_FILE))
        mrf = TreeMRF(nvars, nvals, constants.TensorInitializer.CONSTANT, tree=tree)
        if mrf._factor_variables != [tuple(vars) for vars in factors_dict[constants.FACTOR_VARIABLES]]:
            raise AssertionError
        for factor_fn, table in zip(mrf._factor_functions, factors_dict[constants.FACTOR_FUNCTIONS]):
            factor_fn.logits.data = torch.tensor(table)
        return mrf

    # GloballyNormalizedDistribution
    def unnormalized_probability(self, assignment):
        raise NotImplementedError()

    def energy(self, assignment):
        raise NotImplementedError()

    def normalization_constant(self):
        return NotImplementedError()

    def log_normalization_constant(self):
        return NotImplementedError()

if __name__ == "__main__":
    # TODO move these into proper test files
    from dblm.utils import seeding

    seeding.seed(42)
    mrf = TreeMRF(nvars=4, nvals=[1,4,3,2], initializer=constants.TensorInitializer.CONSTANT)
    table = mrf.to_probability_table().probability_table()
    assert (table == 1/(1*4*3*2)).all()
    assert (table.size() == torch.Size([1,4,3,2]))
