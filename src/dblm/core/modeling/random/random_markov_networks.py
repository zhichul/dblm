import code
import json
import os
from typing import List, Union

import torch
from dblm.core import graph
from dblm.core.modeling import pgm
from dblm.core.modeling.random import constants, random_probability_tables
import torch.nn as nn

FACTOR_VARIABLES="factor_variables"
FACTOR_FUNCTIONS="factor_functions"
NUMBER_OF_VARIABLES="nvars"
NUMBER_OF_VALUES="nvals"

class TreeMRF(nn.Module, pgm.MarkovRandomField):

    graph_file = "graph.json"
    factors_file = "factors.json"

    def __init__(self, nvars: int, nvals: Union[int, List[int]], initializer: constants.TensorInitializer, tree: graph.Graph=None, requires_grad=True) -> None: # type: ignore
        super().__init__()
        if isinstance(nvals, int):
            nvals = [nvals] * nvars
        self.nvars = nvars
        self.nvals = nvals
        self._graph = graph.random_labeled_tree(nvars) if tree is None else tree
        self._factor_variables = []
        self._factor_functions = []
        for edge in self._graph.edges:
            n1, n2 = edge.nodes
            var1, var2 = min(n1.id, n2.id), max(n1.id, n2.id)
            val1, val2 = (nvals[var1], nvals[var2]) # type: ignore
            self._factor_variables.append([var1, var2])
            self._factor_functions.append(random_probability_tables.LogLinearTable((val1, val2), initializer, requires_grad=requires_grad))

    def graph(self):
        return self._graph

    def to_probability_table(self) -> pgm.ProbabilityTable:
        table = None
        for factor_vars, factor_fn in zip(self._factor_variables, self._factor_functions):
            size = [1] * self._graph.num_nodes
            local_factor = factor_fn.log_potential_table()
            for var in factor_vars:
                size[var] = self.nvals[var]
            local_factor = local_factor.reshape(*size)
            table = local_factor if table is None else table + local_factor
        return random_probability_tables.LogLinearTable(self.nvals, table) # type: ignore

    def likelihood_function(self, assignment):
        raise NotImplementedError()

    def log_likelihood_function(self, assignment):
        raise NotImplementedError()

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        self._graph.save(os.path.join(directory, self.graph_file))
        with open(os.path.join(directory, self.factors_file), "w") as f:
            json.dump({
                    FACTOR_VARIABLES: self._factor_variables,
                    FACTOR_FUNCTIONS: [table.log_potential_table().tolist() for table in self._factor_functions],
                    NUMBER_OF_VARIABLES: self.nvars,
                    NUMBER_OF_VALUES: self.nvals
                    }, f)

    @staticmethod
    def load(directory):
        with open(os.path.join(directory, TreeMRF.factors_file)) as f:
            factors_dict = json.load(f)
        nvars = factors_dict[NUMBER_OF_VARIABLES]
        nvals = factors_dict[NUMBER_OF_VALUES]
        tree = graph.Graph.load(os.path.join(directory, TreeMRF.graph_file))
        mrf = TreeMRF(nvars, nvals, constants.TensorInitializer.CONSTANT, tree=tree)
        if mrf._factor_variables != factors_dict[FACTOR_VARIABLES]:
            raise AssertionError
        for factor_fn, table in zip(mrf._factor_functions, factors_dict[FACTOR_FUNCTIONS]):
            factor_fn.logits.data = torch.tensor(table)
        return mrf
