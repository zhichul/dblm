import math
import random
import torch
from typing import Any

import tqdm
from dblm.core.interfaces import pgm
from dblm.core import graph
from dblm.core.modeling import bayesian_networks, constants, factor_graphs, markov_networks, noise
from dblm.core.modeling.interleaved_models import log_linear_transitions

def generate_model(random=False, directed=False):
    sizez0 = 3
    nvals = [2,4,1] # 7 values for emission
    time = 5
    tree = graph.Graph(3)
    tree.add_edge(0,1)
    tree.add_edge(0,2)
    pgmz0 = markov_networks.TreeMRF(sizez0, nvals, constants.TensorInitializer.UNIFORM, tree=tree)
    if not random:
        pgmz0._factor_functions[0].logits.data = torch.tensor([[0., 10., 20., 30.], [0, 0, 10, 20]]).log()
        pgmz0._factor_functions[1].logits.data = torch.tensor([[10.], [1.]]).log()
    pgmz0_noise = noise.NoisyMixture(sizez0, nvals, constants.DiscreteNoise.UNIFORM)
    pgmztxt = log_linear_transitions.FixedLengthDirectedChainWithLogLinearTermFrequencyTransitionAndDeterministicEmissionWithNoiseInZ(sizez0, nvals, time, constants.TensorInitializer.CONSTANT, noise=constants.DiscreteNoise.UNIFORM, separate_noise_distribution_per_state=True)
    if not random:
        pgmztxt._factor_functions[0].logits.data.zero_() # type:ignore
        pgmztxt.transition.layer.bias.data = torch.tensor([[1,-math.inf,1],[1,1,-math.inf],[-math.inf, 1, 1]]).reshape(-1)
        pgmztxt.transition.layer.weight.data = torch.tensor([[1000., 0., 3., 4., 5., 6., 7.], # 0 -> 0 continue talk about z0_0 if z0_0 = 0 and talked a lot
                                                                    [1., 2., 3., 4., 5., 6., 7.], # 0 -> 1 this transition is ruled out by bias so don't care
                                                                    [0., 1000., 3., 4., 5., 6., 7.], # 0 -> 2 switch to talk about z0_2 if z0_0 = 1 and talked a lot
                                                                    [1., 2., 1000., 1000., 5., 6., 7.], # 1 -> 0
                                                                    [1., 2., 3., 4., 1000, 1000, 7.], # 1 -> 1
                                                                    [1., 2., 3., 4., 5., 6., 7.], # 1 -> 2 this transition is ruled out by bias so don't care
                                                                    [1., 2., 3., 4., 5., 6., 7.], # 2 -> 0 this transition is ruled out by bias so don't care
                                                                    [1., 2., 3., 4., 5., 6., 7.], # 2 -> 1
                                                                    [1., 2., 3., 4., 5., 6., 7.], # 2 -> 2
                                                                    ])
    if directed:
        # directed representation for ancestral sampling interface
        directed_model = bayesian_networks.BayesianNetwork.join(pgmz0.to_probability_table().to_bayesian_network(), pgmz0_noise, {0:0,1:1,2:2})
        directed_model = bayesian_networks.BayesianNetwork.join(directed_model, pgmztxt, {0:9, 1:10, 2:11})
        return directed_model, [(15,16),(20,21),(25,26),(30,31),(35,36)]
    else:
        # factor graph representation for BP interface
        factor_model = factor_graphs.FactorGraph.join(pgmz0.to_factor_graph_model(), pgmz0_noise, {0:0,1:1,2:2})
        factor_model = factor_graphs.FactorGraph.join(factor_model, pgmztxt, {0:9, 1:10, 2:11})
        return factor_model, [(15,16),(20,21),(25,26),(30,31),(35,36)]
