
import math
import unittest

import torch
from dblm.core import graph
from dblm.core.inferencers import belief_propagation

from dblm.core.modeling import constants, factor_graphs, markov_networks
from dblm.core.modeling.interleaved_models import log_linear_transitions


class TestTreeLogLinearTransition(unittest.TestCase):

    def setUp(self):
        self.sizez0 = 3
        self.nvals = [2,4,1] # 7 values for emission
        self.time = 5
        self.tree = graph.Graph(3)
        self.tree.add_edge(0,1)
        self.tree.add_edge(0,2)
        self.pgmz0 = markov_networks.TreeMRF(self.sizez0, self.nvals, constants.TensorInitializer.CONSTANT, tree=self.tree)
        self.pgmz0._factor_functions[0].logits.data = torch.tensor([[0., 10., 20., 30.], [0, 0, 10, 20]]).log()
        self.pgmz0._factor_functions[1].logits.data = torch.tensor([[10.], [1.]]).log()
        self.pgmztxt = log_linear_transitions.FixedLengthDirectedChainWithLogLinearTermFrequencyTransitionAndDeterministicEmission(self.sizez0, self.nvals, self.time, constants.TensorInitializer.CONSTANT)
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
        model = factor_graphs.FactorGraph.join(self.pgmz0.to_factor_graph_model(), self.pgmztxt.to_factor_graph_model(), {0:0,1:1,2:2})
        self.model = model

    def test_join(self):
        self.assertEqual(13, self.model.nvars)
        self.assertEqual([2,4,1,3,7,3,7,3,7,3,7,3,7], self.model.nvals)
        self.assertEqual([(0,1), (0,2), (3,), (0,1,2,3,4), (3,4,5), (0,1,2,5,6), (4,5,6,7), (0,1,2,7,8), (4,6,7,8,9), (0,1,2,9,10), (4,6,8,9,10,11), (0,1,2,11,12)], self.model.factor_variables()) # chain factors
        self.assertEqual(12, len(self.model.factor_functions()))
        self.assertEqual(48, self.model.graph().num_edges)
        self.assertEqual([0,0,1,1,2,3,3,3,3,3,4,4,4,5,5,5,5,5,6,6,6,6,7,7,7,7,7,8,8,8,8,8,9,9,9,9, 9,10,10,10,10,10,10,11,11,11,11,11], [edge.factor_id for edge in self.model.graph().edges])
        self.assertEqual([0,1,0,2,3,0,1,2,3,4,3,4,5,0,1,2,5,6,4,5,6,7,0,1,2,7,8,4,6,7,8,9,0,1,2,9,10, 4, 6, 8, 9,10,11, 0, 1, 2,11,12], [edge.variable_id for edge in self.model.graph().edges])

    def test_unnormalized_likelihood(self):
        # for the initialization above
        # 0 1 0 has potential 100,
        # 0 2 0 200
        # 1 2 0 10
        # 0 3 0 300
        # 1 3 0 20
        # the factor 0-1 is max(v1 - v0, 0) * 10
        # the factor 0-2 is 10 if v0 == v2, else 1
        # the chain is conditioned on EOS, so still need global renormalization.
        self.assertAlmostEqual(0, self.model.fix_variables({4:2,6:0,8:6,10:2,12:0}).unnormalized_likelihood_function((0,0,0, 1,2, 0,0, 2,6, 1,2, 0,0)).item()) # type:ignore impossible z0
        self.assertAlmostEqual(0, self.model.fix_variables({4:5,6:6,8:6,10:5,12:1}).unnormalized_likelihood_function((1,3,0, 1,5, 2,6, 2,6, 1,5, 0,1)).item()) # type:ignore impossible zt
        self.assertAlmostEqual(0, self.model.fix_variables({4:5,6:1,8:6,10:5,12:2}).unnormalized_likelihood_function((1,3,0, 1,5, 0,1, 2,6, 1,5, 0,2)).item()) # type:ignore impossible xt | z0 zt

        self.assertAlmostEqual(math.log(20)
                               + math.log(1/3)
                               + 7 - torch.logsumexp(torch.tensor([7.,1001.]), dim=-1).item()
                               + 504 - torch.logsumexp(torch.tensor([4.,504.]), dim=-1).item()
                               + math.log(0.5)
                               + (1 + 2/4 + 6/2 + 7/4) - torch.logsumexp(torch.tensor([1 + 2/4 + 3 + 7/4, 1 + 2/4 + 500 + 7/4]), dim=-1).item(), self.model.fix_variables({4:5,6:1,8:6,10:5,12:1}).log_unnormalized_likelihood_function((1,3,0, 1,5, 0,1, 2,6, 1,5, 0,1)).item(), 3) # type:ignore

    def test_inference(self):
        bp = belief_propagation.FactorGraphBeliefPropagation()
        inference_results = bp.inference(self.model, {4:5,6:1,8:6,10:5,12:1}, [0,1,2, 3,5,7,9,11], iterations=10, return_messages=True, renormalize=True)
        torch.testing.assert_close(inference_results.query_marginals[0].probability_table(), torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(inference_results.query_marginals[1].probability_table(), torch.tensor([0.0,0.0,0.0,1.0]))
        torch.testing.assert_close(inference_results.query_marginals[2].probability_table(), torch.tensor([1.0]))
        torch.testing.assert_close(inference_results.query_marginals[3].probability_table(), torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(inference_results.query_marginals[4].probability_table(), torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(inference_results.query_marginals[5].probability_table(), torch.tensor([0.0, 0.0, 1.0]))
        torch.testing.assert_close(inference_results.query_marginals[6].probability_table(), torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(inference_results.query_marginals[7].probability_table(), torch.tensor([1.0, 0.0, 0.0]))

if __name__ == "__main__":
    unittest.main()
