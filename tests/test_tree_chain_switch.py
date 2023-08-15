import math
import unittest

import torch
from dblm.core import graph

from dblm.core.modeling import chains, constants, markov_networks, switching_tables, factor_graphs

class TestTreeChainSwitch(unittest.TestCase):

    def setUp(self):
        self.sizez0 = 3
        self.nvals = [2,4,1]
        self.time = 5
        self.tree = graph.Graph(3)
        self.tree.add_edge(0,1)
        self.tree.add_edge(0,2)
        self.pgmz0 = markov_networks.TreeMRF(self.sizez0, self.nvals, constants.TensorInitializer.CONSTANT, tree=self.tree)
        self.pgmz0._factor_functions[0].logits.data = torch.tensor([[0., 10., 20., 30.], [0, 0, 10, 20]]).log()
        self.pgmz0._factor_functions[1].logits.data = torch.tensor([[10.], [1.]]).log()
        self.pgmzt = chains.FixedLengthDirectedChain(self.time, self.sizez0, constants.TensorInitializer.CONSTANT)
        self.pgmzt._factor_functions[0].logits.data = torch.tensor([-math.inf, 0, -math.inf]) # initial one set to be 3
        self.pgmzt._factor_functions[-1].logits.data = torch.tensor([-math.inf, 0, -math.inf]) # final one set to be 3
        self.pgmzt._factor_functions[1].logits.data = torch.tensor([[1., 1., 1.],[10,10,10],[1000,1000,1000]]) # transition (SHARED) set to be uniform, scale shouldn't matter as locally normalized
        self.pgmxt = switching_tables.BatchedSwitchingTables(self.sizez0, self.nvals, self.time)
        model = factor_graphs.FactorGraph.join(self.pgmz0.to_factor_graph_model(), self.pgmzt.to_factor_graph_model(), dict())
        model = factor_graphs.FactorGraph.join(model, self.pgmxt,
                                               dict([(i, i) for i in range(self.sizez0)] +
                                                    [(j,j) for j in range(self.sizez0, self.sizez0 + self.time)]))
        self.model = model

    def test_join(self):
        self.assertEqual(13, self.model.nvars())
        self.assertEqual([2,4,1,3,3,3,3,3,7,7,7,7,7], self.model.nvals())
        self.assertEqual([(3,), (3,4), (4,5), (5,6), (6,7), (7,)], self.model.factor_variables()[2:2+6]) # chain factors
        self.assertEqual([(0,1,2,3,8), (0,1,2,4,9), (0,1,2,5,10), (0,1,2,6,11), (0,1,2,7,12)], self.model.factor_variables()[2+6:2+6+5]) # switch factors
        self.assertEqual(13, len(self.model.factor_functions()))

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
        self.assertAlmostEqual(0, self.model.unnormalized_likelihood_function((0,0,0, 1,2,0,2,1, 6,2,0,2,6)).item()) # type:ignore impossible z0
        self.assertAlmostEqual(0, self.model.unnormalized_likelihood_function((1,3,0, 1,2,0,2,2, 5,6,1,6,6)).item()) # type:ignore impossible zt
        self.assertAlmostEqual(0, self.model.unnormalized_likelihood_function((1,3,0, 1,2,0,2,1, 5,6,1,5,5)).item()) # type:ignore impossible xt | z0 zt

        self.assertAlmostEqual(20 * 1/81, self.model.unnormalized_likelihood_function((1,3,0, 1,2,0,2,1, 5,6,1,6,5)).item()) # type:ignore
        self.assertAlmostEqual(math.log(200 * 1/81), self.model.log_unnormalized_likelihood_function((0,2,0, 1,2,0,2,1, 4,6,0,6,4)).item()) # type:ignore

    def test_likelihood(self):
        table = self.model.to_probability_table()
        self.assertAlmostEqual(20/630 * 1/27,table.likelihood_function((1,3,0, 1,2,0,2,1, 5,6,1,6,5)).item())
        self.assertAlmostEqual(200/630 * 1/27,table.likelihood_function((0,2,0, 1,2,0,2,1, 4,6,0,6,4)).item())

if __name__ == '__main__':
    unittest.main()
