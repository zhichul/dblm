from __future__ import annotations
import collections
import json
import os

import torch
from dblm.core import graph

from dblm.core.modeling import bayesian_networks, constants, factor_graphs, markov_networks, noise
from dblm.core.modeling.interleaved_models import log_linear_transitions
from dblm.utils import seeding

class TreeMrfNoiselessEmission:

    def __init__(self,
                w_nvars: int,
                w_nvals:int | list[int],
                w_param_initializer: constants.TensorInitializer,
                zt_sequence_length,
                zt_param_initializer: constants.TensorInitializer,
                w_graph: graph.Graph=None, # type:ignore
                seed=42): # type:ignore
        seeding.seed(seed)
        if isinstance(w_nvals, int):
            w_nvals = [w_nvals] * w_nvars
        assert isinstance(w_nvals, list)
        self.p_w =  markov_networks.TreeMRF(w_nvars, w_nvals, w_param_initializer, tree=w_graph)
        self.p_w_table = self.p_w.to_probability_table().to_bayesian_network()
        self.factor_graph_model = self.p_w.to_factor_graph_model()
        self.bayes_net_model = self.p_w_table
        self.p_zt_xt = log_linear_transitions.FixedLengthDirectedChainWithLogLinearTermFrequencyTransitionAndDeterministicEmission(
            w_nvars,
            w_nvals,
            zt_sequence_length,
            zt_param_initializer)
        bindings = list(range(w_nvars))
        # z0 z0 z0 z0 _ _ _ _, _ _ _ _, z0'1 z0'1 z0'1 z0'1 _ _ _ _, _ _ _ _, z0'2 z0'2 z0'2 z0'2 ...
        self.factor_graph_model = factor_graphs.FactorGraph.join(self.factor_graph_model, self.p_zt_xt.to_factor_graph_model(), dict(enumerate(bindings)))
        self.bayes_net_model = bayesian_networks.BayesianNetwork.join(self.bayes_net_model, self.p_zt_xt, dict(enumerate(bindings)))
        factor_offset_zt = w_nvars - 1
        var_offset_zt = factor_offset_zt + 1
        self.nested_model = factor_graphs.BPAutoregressiveIncompleteLikelihoodFactorGraph.from_factor_graph(self.factor_graph_model, [(factor_offset_zt + i * 2 + 1, var_offset_zt + i * 2 + 1)for i in range(zt_sequence_length)]) # offset_zt is the initial index of the first factor related to t>0, and +1 is the offset
        self.names = [f"u({i})" for i in range(w_nvars)]
        for t in range(zt_sequence_length):
            self.names += [f"z({t})", f"x({t})"]
        assert self.factor_graph_model.nvars == len(self.names)
        self.config = {
            "w_nvars": w_nvars,
            "w_nvals": w_nvals,
            "w_param_initializer": str(w_param_initializer),
            "zt_sequence_length": zt_sequence_length,
            "zt_param_initializer": str(zt_param_initializer),
            "seed": seed
        }

    def save(self, directory):
        self.p_w.graph().save(os.path.join(directory, "w_graph.json"))
        torch.save(self.p_zt_xt.state_dict(), os.path.join(directory, "p_zt_xt.pt"))
        torch.save(self.p_w.state_dict(), os.path.join(directory, "p_w.pt"))
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(self.config, f)
        with open(os.path.join(directory, "mapping.json"), "w") as f:
            json.dump(collections.OrderedDict(
                (name, i) for i, name in enumerate(self.names)
            ), f, indent=4)
        with open(os.path.join(directory, "factor_variables.json"), "w") as f:
            json.dump(self.factor_graph_model.factor_variables(), f, indent=4)

    @staticmethod
    def load(directory):
        with open(os.path.join(directory, "config.json")) as f:
            config = json.load(f)
        w_graph = graph.Graph.load(os.path.join(directory, "w_graph.json"))
        model = TreeMrfNoiselessEmission(config["w_nvars"],
                             config["w_nvals"],
                             constants.TensorInitializer.CONSTANT,
                             config["zt_sequence_length"],
                             constants.TensorInitializer.CONSTANT,
                             w_graph=w_graph,
                             seed=config["seed"])
        model.p_w.load_state_dict(torch.load(os.path.join(directory, "p_w.pt")))
        model.p_w_table.load_state_dict(model.p_w.to_probability_table().to_bayesian_network().state_dict()) # type:ignore
        model.p_zt_xt.load_state_dict(torch.load(os.path.join(directory, "p_zt_xt.pt")))
        model.config["w_param_initializer"] = config["w_param_initializer"]
        model.config["zt_param_initializer"] = config["zt_param_initializer"]
        return model
