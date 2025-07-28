
import math
import unittest

import torch
from dblm.core import graph
from dblm.core.inferencers import belief_propagation

from dblm.core.modeling import bayesian_networks, constants, factor_graphs, markov_networks, noise
from dblm.core.modeling.interleaved_models import log_linear_transitions


class TestTreeLogLinearTransitionWithNoiseOneForEachToken(unittest.TestCase):

    def setUp(self):
        self.sizez0 = 3
        self.nvals = [2,4,1] # 7 values for emission
        self.time = 5
        self.tree = graph.Graph(3)
        self.tree.add_edge(0,1)
        self.tree.add_edge(0,2)
        self.pgmz0 = markov_networks.TreeMRF(self.sizez0, self.nvals, constants.TensorInitializer.CONSTANT, tree=self.tree)
        self.pgmz0._factor_functions[0]._logits.data = torch.tensor([[0., 10., 20., 30.], [0, 0, 10, 20]]).log()
        self.pgmz0._factor_functions[1]._logits.data = torch.tensor([[10.], [1.]]).log()
        self.pgmz0_noise = noise.NoisyMixture(self.sizez0, self.nvals, constants.DiscreteNoise.UNIFORM)
        self.pgmztxt = log_linear_transitions.FixedLengthDirectedChainWithLogLinearTermFrequencyTransitionAndDeterministicEmissionWithNoiseInZOneCopyForEveryToken(self.sizez0, self.nvals, self.time, constants.TensorInitializer.CONSTANT, noise=constants.DiscreteNoise.UNIFORM, separate_noise_distribution_per_state=True)
        self.pgmztxt._factor_functions[0]._logits.data.zero_() # type:ignore
        self.pgmztxt.transition.layer.bias.data = torch.tensor([[1,-math.inf,1],[1,1,-math.inf],[-math.inf, 1, 1]]).reshape(-1)
        self.pgmztxt.transition.layer.weight.data = torch.tensor([[1000., 0., 3., 4., 5., 6., 7.], # 0 -> 0 continue talk about z0_0 if z0_0 = 0 and talked a lot
                                                                  [1., 2., 3., 4., 5., 6., 7.], # 0 -> 1 this transition is ruled out by bias so don't care
                                                                  [0., 1000., 3., 4., 5., 6., 7.], # 0 -> 2 switch to talk about z0_2 if z0_0 = 1 and talked a lot
                                                                  [1., 2., 1000., 1000., 5., 6., 7.], # 1 -> 0
                                                                  [1., 2., 3., 4., 1000, 1000, 7.], # 1 -> 1
                                                                  [1., 2., 3., 4., 5., 6., 7.], # 1 -> 2 this transition is ruled out by bias so don't care
                                                                  [1., 2., 3., 4., 5., 6., 7.], # 2 -> 0 this transition is ruled out by bias so don't care
                                                                  [1., 2., 3., 4., 5., 6., 7.], # 2 -> 1
                                                                  [1., 2., 3., 4., 5., 6., 7.], # 2 -> 2
                                                                  ])
        model = factor_graphs.FactorGraph.join(self.pgmz0.to_factor_graph_model(), self.pgmz0_noise.to_factor_graph_model(), {0:0,1:1,2:2})
        model = factor_graphs.FactorGraph.join(model, self.pgmz0_noise.to_factor_graph_model(), {0:0,1:1,2:2})
        model = factor_graphs.FactorGraph.join(model, self.pgmz0_noise.to_factor_graph_model(), {0:0,1:1,2:2})
        model = factor_graphs.FactorGraph.join(model, self.pgmz0_noise.to_factor_graph_model(), {0:0,1:1,2:2})
        model = factor_graphs.FactorGraph.join(model, self.pgmz0_noise.to_factor_graph_model(), {0:0,1:1,2:2})
        model = factor_graphs.FactorGraph.join(model, self.pgmztxt.to_factor_graph_model(), dict(enumerate([9,10,11,18,19,20,27,28,29,36,37,38,45,46,47])))
        self.model = model

    def test_join(self):
        self.assertEqual(3  + 9 * 5 + 5 * 5, self.model.nvars)
        self.assertEqual([2,4,1,2,4,1,2,2,2,2,4,1,2,4,1,2,2,2,2,4,1,2,4,1,2,2,2,2,4,1,2,4,1,2,2,2,2,4,1,2,4,1,2,2,2,2,4,1,3,3,2,3,7,3,3,2,3,7,3,3,2,3,7,3,3,2,3,7,3,3,2,3,7], self.model.nvals)
        fvars_ref = [(0,1), (0,2), # z0
                           (3,), (4,), (5,), # noise
                           (6,), (7,), (8,), # switch
                           (0,3,6,9), (1,4,7,10), (2,5,8,11), # output
                           (12,), (13,), (14,), # noise
                           (15,), (16,), (17,), # switch
                           (0,12,15,18), (1,13,16,19), (2,14,17,20), # output
                           (21,), (22,), (23,), # noise
                           (24,), (25,), (26,), # switch
                           (0,21,24,27), (1,22,25,28), (2,23,26,29), # output
                           (30,), (31,), (32,), # noise
                           (33,), (34,), (35,), # switch
                           (0,30,33,36), (1,31,34,37), (2,32,35,38), # output
                           (39,), (40,), (41,), # noise
                           (42,), (43,), (44,), # switch
                           (0,39,42,45), (1,40,43,46), (2,41,44,47), # output
                            (48,), (48, 49), (50,), (48, 49, 50, 51),(9, 10, 11, 51, 52),
                            (48, 52, 53), (53, 54), (55,), (53, 54, 55, 56),(18, 19, 20, 56, 57),
                            (52, 53, 57, 58), (58, 59), (60,), (58, 59, 60, 61), (27, 28, 29, 61, 62),
                            (52, 57, 58, 62, 63), (63, 64), (65,), (63, 64, 65, 66), (36, 37, 38, 66, 67),
                            (52, 57, 62, 63, 67, 68), (68, 69), (70,), (68, 69, 70, 71), (45, 46, 47, 71, 72)]
        self.assertEqual(fvars_ref, self.model.factor_variables()) # chain factors
        self.assertEqual(72, len(self.model.factor_functions()))
        self.assertEqual(4 + (3 + 3 + 4 + 4 + 4) * 5
                         + 1 + 2 + 1 + 4 + 5
                         + 3 + 2 + 1 + 4 + 5
                         + 4 + 2 + 1 + 4 + 5
                         + 5 + 2 + 1 + 4 + 5
                         + 6 + 2 + 1 + 4 + 5, self.model.graph().num_edges)
        self.assertEqual([0,0,1,1,2,3,4,5,6,7,8,8,8,8,9,9,9,9,10,10,10,10,
                          11, 12, 13, 14, 15, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19,
                          20, 21, 22, 23, 24, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28,
                          29, 30, 31, 32, 33, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37,
                          38, 39, 40, 41, 42, 43, 44, 44, 44, 44, 45, 45, 45, 45, 46, 46, 46, 46,
                          47, 48, 48, 49, 50, 50, 50, 50, 51, 51, 51, 51, 51,
                          52, 52, 52, 53, 53, 54, 55, 55, 55, 55, 56, 56, 56, 56, 56,
                          57, 57, 57, 57, 58, 58, 59, 60, 60, 60, 60, 61, 61, 61, 61, 61,
                          62, 62, 62, 62, 62, 63, 63, 64, 65, 65, 65, 65, 66, 66, 66, 66, 66,
                          67, 67, 67, 67, 67, 67, 68, 68, 69, 70, 70, 70, 70, 71, 71, 71, 71, 71], [edge.factor_id for edge in self.model.graph().edges])
        self.assertEqual(list(sum(fvars_ref, tuple())), [edge.variable_id for edge in self.model.graph().edges]) # type:ignore


if __name__ == "__main__":
    unittest.main()
