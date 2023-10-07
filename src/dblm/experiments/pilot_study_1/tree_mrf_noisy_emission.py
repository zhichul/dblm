from __future__ import annotations
import collections
import json
import os

import torch
from dblm.core import graph

from dblm.core.modeling import bayesian_networks, constants, factor_graphs, markov_networks, noise
from dblm.core.modeling.interleaved_models import log_linear_transitions
from dblm.utils import seeding

class TreeMrfNoisyEmission:

    def __init__(self,
                w_nvars: int,
                w_nvals:int | list[int],
                w_param_initializer: constants.TensorInitializer,
                w_noise_ratio: tuple[float,float], # first is signal second is noise
                zt_sequence_length,
                zt_param_initializer: constants.TensorInitializer,
                zt_noise_ratio: tuple[float,float], # first is signal second is noise
                w_graph: graph.Graph=None, # type:ignore
                seed=42): # type:ignore
        seeding.seed(seed)
        if isinstance(w_nvals, int):
            w_nvals = [w_nvals] * w_nvars
        assert isinstance(w_nvals, list)
        self.p_w =  markov_networks.TreeMRF(w_nvars, w_nvals, w_param_initializer, tree=w_graph)
        self.p_w_table = self.p_w.to_probability_table().to_bayesian_network()
        self.p_w_noise = noise.NoisyMixture(w_nvars, w_nvals, constants.DiscreteNoise.UNIFORM, mixture_ratio=w_noise_ratio)
        self.factor_graph_model = self.p_w.to_factor_graph_model()
        self.bayes_net_model = self.p_w_table

        for _ in range(zt_sequence_length):
            self.factor_graph_model = factor_graphs.FactorGraph.join(self.factor_graph_model, self.p_w_noise.to_factor_graph_model(), dict(enumerate(range(w_nvars))))
            self.bayes_net_model = bayesian_networks.BayesianNetwork.join(self.bayes_net_model, self.p_w_noise, dict(enumerate(range(w_nvars))))
        self.p_zt_xt = log_linear_transitions.FixedLengthDirectedChainWithLogLinearTermFrequencyTransitionAndDeterministicEmissionWithNoiseInZOneCopyForEveryToken(
            w_nvars,
            w_nvals,
            zt_sequence_length,
            zt_param_initializer,
            noise=constants.DiscreteNoise.UNIFORM,
            mixture_ratio=zt_noise_ratio,
            separate_noise_distribution_per_state=False)
        bindings = [w_nvars + i * w_nvars * 3 + w_nvars * 2 + j for i in range(zt_sequence_length) for j in range(w_nvars)]
        # w w w w _ _ _ _, _ _ _ _, w'1 w'1 w'1 w'1 _ _ _ _, _ _ _ _, w'2 w'2 w'2 w'2 ...
        self.factor_graph_model = factor_graphs.FactorGraph.join(self.factor_graph_model, self.p_zt_xt.to_factor_graph_model(), dict(enumerate(bindings)))
        self.bayes_net_model = bayesian_networks.BayesianNetwork.join(self.bayes_net_model, self.p_zt_xt, dict(enumerate(bindings)))
        factor_offset_zt = w_nvars - 1 + 3 * w_nvars * zt_sequence_length
        var_offset_zt = factor_offset_zt + 1
        self.nested_model = factor_graphs.BPAutoregressiveIncompleteLikelihoodFactorGraph.from_factor_graph(self.factor_graph_model, [(factor_offset_zt + i * 5 + 4, var_offset_zt + i * 5 + 4)for i in range(zt_sequence_length)]) # offset_zt is the initial index of the first factor related to t>0, and +1 is the offset
        self.names = [f"u({i})" for i in range(w_nvars)]
        for t in range(zt_sequence_length):
            self.names += [f"unoise({t},{i})" for i in range(w_nvars)] + [f"uswitch({t},{i})" for i in range(w_nvars)] + [f"u'({t},{i})" for i in range(w_nvars)]
        for t in range(zt_sequence_length):
            self.names += [f"z({t})", f"znoise({t})", f"zswitch({t})", f"z'({t})", f"x({t})"]
        assert self.factor_graph_model.nvars == len(self.names)
        self.config = {
            "w_nvars": w_nvars,
            "w_nvals": w_nvals,
            "w_param_initializer": str(w_param_initializer),
            "w_noise_ratio": w_noise_ratio,
            "zt_sequence_length": zt_sequence_length,
            "zt_param_initializer": str(zt_param_initializer),
            "zt_noise_ratio": zt_noise_ratio,
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
        model = TreeMrfNoisyEmission(config["w_nvars"],
                             config["w_nvals"],
                             constants.TensorInitializer.CONSTANT,
                             config["w_noise_ratio"],
                             config["zt_sequence_length"],
                             constants.TensorInitializer.CONSTANT,
                             config["zt_noise_ratio"],
                             w_graph=w_graph,
                             seed=config["seed"])
        model.p_w.load_state_dict(torch.load(os.path.join(directory, "p_w.pt")))
        model.p_w_table.load_state_dict(model.p_w.to_probability_table().to_bayesian_network().state_dict()) # type:ignore
        model.p_zt_xt.load_state_dict(torch.load(os.path.join(directory, "p_zt_xt.pt")))
        model.config["w_param_initializer"] = config["w_param_initializer"]
        model.config["zt_param_initializer"] = config["zt_param_initializer"]
        return model
