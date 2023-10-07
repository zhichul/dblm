import math
import unittest

import torch
from dblm.core import graph
from dblm.core.inferencers import belief_propagation

from dblm.core.modeling import batched_switching_tables, chains, constants, markov_networks, factor_graphs

class TestTreeChainSwitch(unittest.TestCase):

    def setUp(self):
        self.sizez0 = 3
        self.nvals = [2,4,1]
        self.time = 5
        self.tree = graph.Graph(3)
        self.tree.add_edge(0,1)
        self.tree.add_edge(0,2)
        self.pgmz0 = markov_networks.TreeMRF(self.sizez0, self.nvals, constants.TensorInitializer.CONSTANT, tree=self.tree)
        self.pgmz0._factor_functions[0]._logits.data = torch.tensor([[0., 10., 20., 30.], [0, 0, 10, 20]]).log()
        self.pgmz0._factor_functions[1]._logits.data = torch.tensor([[10.], [1.]]).log()
        self.pgmzt = chains.FixedLengthDirectedChain(self.time, self.sizez0, constants.TensorInitializer.CONSTANT)
        self.pgmzt._factor_functions[0]._logits.data = torch.tensor([-math.inf, 0, -math.inf]) # initial one set to be 3
        self.pgmzt._factor_functions[-1]._logits.data = torch.tensor([-math.inf, 0, -math.inf]) # final one set to be 3
        self.pgmzt._factor_functions[1]._logits.data = torch.tensor([[1., 1., 1.],[10,10,10],[1000,1000,1000]]) # transition (SHARED) set to be uniform, scale shouldn't matter as locally normalized
        self.pgmxt = batched_switching_tables.BatchedSwitchingTables(self.sizez0, self.nvals, self.time)
        model = factor_graphs.FactorGraph.join(self.pgmz0.to_factor_graph_model(), self.pgmzt.to_factor_graph_model(), dict())
        model = factor_graphs.FactorGraph.join(model, self.pgmxt,
                                               dict([(i, i) for i in range(self.sizez0)] +
                                                    [(j,j) for j in range(self.sizez0, self.sizez0 + self.time)]))
        self.model = model

    def test_join(self):
        self.assertEqual(13, self.model.nvars)
        self.assertEqual([2,4,1,3,3,3,3,3,7,7,7,7,7], self.model.nvals)
        self.assertEqual([(3,), (3,4), (4,5), (5,6), (6,7), (7,)], self.model.factor_variables()[2:2+6]) # chain factors
        self.assertEqual([(0,1,2,3,8), (0,1,2,4,9), (0,1,2,5,10), (0,1,2,6,11), (0,1,2,7,12)], self.model.factor_variables()[2+6:2+6+5]) # switch factors
        self.assertEqual(13, len(self.model.factor_functions()))
        self.assertEqual(39, self.model.graph().num_edges)
        self.assertEqual([0,0,1,1,2,3,3,4,4,5,5,6,6,7,8,8,8,8,8,9,9,9,9,9,10,10,10,10,10,11,11,11,11,11,12,12,12,12,12], [edge.nodes[0].id for edge in self.model.graph().edges])
        self.assertEqual([0,1,0,2,3,3,4,4,5,5,6,6,7,7,0,1,2,3,8,0,1,2,4,9, 0, 1, 2, 5,10, 0, 1, 2, 6,11, 0, 1, 2, 7,12], [edge.nodes[1].id for edge in self.model.graph().edges])

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
        self.assertAlmostEqual(0, self.model.unnormalized_probability((0,0,0, 1,2,0,2,1, 6,2,0,2,6)).item()) # type:ignore impossible z0
        self.assertAlmostEqual(0, self.model.unnormalized_probability((1,3,0, 1,2,0,2,2, 5,6,1,6,6)).item()) # type:ignore impossible zt
        self.assertAlmostEqual(0, self.model.unnormalized_probability((1,3,0, 1,2,0,2,1, 5,6,1,5,5)).item()) # type:ignore impossible xt | z0 zt

        self.assertAlmostEqual(20 * 1/81, self.model.unnormalized_probability((1,3,0, 1,2,0,2,1, 5,6,1,6,5)).item()) # type:ignore
        self.assertAlmostEqual(math.log(200 * 1/81), self.model.energy((0,2,0, 1,2,0,2,1, 4,6,0,6,4)).item()) # type:ignore

    def test_likelihood(self):
        table = self.model.to_potential_table()
        self.assertAlmostEqual(math.log(200 * 1/81), table.log_potential_value((0,2,0, 1,2,0,2,1, 4,6,0,6,4)).item()) # type:ignore
        table = self.model.to_probability_table()
        self.assertAlmostEqual(20/630 * 1/27,table.probability((1,3,0, 1,2,0,2,1, 5,6,1,6,5)).item())
        self.assertAlmostEqual(200/630 * 1/27,table.probability((0,2,0, 1,2,0,2,1, 4,6,0,6,4)).item())

    def test_tree_bp(self):
        bp = belief_propagation.FactorGraphBeliefPropagation()
        inference_results = bp.inference(self.model, {8:4,9:6,10:0,11:6,12:4}, [0,1,2, 3,4,5,6,7], iterations=10, return_messages=True, renormalize=False, materialize_switch=True)

        # first factor message
        torch.testing.assert_close(inference_results.messages_to_variables[0][0].potential_table(), torch.tensor([60., 30])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][1].potential_table(), torch.tensor([0., 10, 30, 50])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][2].potential_table(), torch.tensor([10., 1])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][3].potential_table(), torch.tensor([11.])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][4].potential_table(), torch.tensor([0., 1., 0.])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][5].potential_table(), torch.tensor([1.0, 1.0, 1.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][6].potential_table(), torch.tensor([1.0, 1.0, 1.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][7].potential_table(), torch.tensor([1.0, 1.0, 1.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][8].potential_table(), torch.tensor([1.0, 1.0, 1.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][9].potential_table(), torch.tensor([1.0, 1.0, 1.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][10].potential_table(), torch.tensor([1.0, 1.0, 1.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][11].potential_table(), torch.tensor([1.0, 1.0, 1.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][12].potential_table(), torch.tensor([1.0, 1.0, 1.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][13].potential_table(), torch.tensor([0, 1.0, 0])) # type:ignore

        # the switching variable messages
        torch.testing.assert_close(inference_results.messages_to_variables[0][14].potential_table(), torch.tensor([1.0, 1.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][15].potential_table(), torch.tensor([0.0, 0.0, 2.0, 0.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][16].potential_table(), torch.tensor([2.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][17].potential_table(), torch.tensor([0.0, 2.0, 0.0])) # type:ignore
        self.assertIs(inference_results.messages_to_variables[0][18], None) # type:ignore

        torch.testing.assert_close(inference_results.messages_to_variables[0][19].potential_table(), torch.tensor([4.0, 4.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][20].potential_table(), torch.tensor([2.0, 2.0, 2.0, 2.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][21].potential_table(), torch.tensor([8.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][22].potential_table(), torch.tensor([0.0, 0.0, 8.0])) # type:ignore
        self.assertIs(inference_results.messages_to_variables[0][23], None) # type:ignore

        torch.testing.assert_close(inference_results.messages_to_variables[0][24].potential_table(), torch.tensor([4.0, 0.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][25].potential_table(), torch.tensor([1.0, 1.0, 1.0, 1.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][26].potential_table(), torch.tensor([4.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][27].potential_table(), torch.tensor([4.0, 0.0, 0.0])) # type:ignore
        self.assertIs(inference_results.messages_to_variables[0][28], None) # type:ignore

        torch.testing.assert_close(inference_results.messages_to_variables[0][29].potential_table(), torch.tensor([4.0, 4.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][30].potential_table(), torch.tensor([2.0, 2.0, 2.0, 2.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][31].potential_table(), torch.tensor([8.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][32].potential_table(), torch.tensor([0.0, 0.0, 8.0])) # type:ignore
        self.assertIs(inference_results.messages_to_variables[0][33], None) # type:ignore

        torch.testing.assert_close(inference_results.messages_to_variables[0][34].potential_table(), torch.tensor([1.0, 1.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][35].potential_table(), torch.tensor([0.0, 0.0, 2.0, 0.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][36].potential_table(), torch.tensor([2.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_variables[0][37].potential_table(), torch.tensor([0.0, 2.0, 0.0])) # type:ignore
        self.assertIs(inference_results.messages_to_variables[0][38], None) # type:ignore

        # second variable message
        torch.testing.assert_close(inference_results.messages_to_factors[1][0].potential_table(), torch.tensor([640., 0.])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][1].potential_table(), torch.tensor([0., 0., 16., 0.])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][2].potential_table(), torch.tensor([3840., 0.])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][3].potential_table(), torch.tensor([1024.])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][4].potential_table(), torch.tensor([0., 2., 0.])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][5].potential_table(), torch.tensor([0.0, 2., 0.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][6].potential_table(), torch.tensor([0.0, 0.0, 8.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][7].potential_table(), torch.tensor([0.0, 0.0, 8.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][8].potential_table(), torch.tensor([4., 0, 0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][9].potential_table(), torch.tensor([4., 0, 0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][10].potential_table(), torch.tensor([0.0, 0.0, 8.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][11].potential_table(), torch.tensor([0.0, 0.0, 8.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][12].potential_table(), torch.tensor([0.0, 2., 0.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][13].potential_table(), torch.tensor([0.0, 2., 0.0])) # type:ignore
        # # the switching variable messages
        torch.testing.assert_close(inference_results.messages_to_factors[1][14].potential_table(), torch.tensor([38400, 0.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][15].potential_table(), torch.tensor([0.0, 0.0, 480/2.0, 0.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][16].potential_table(), torch.tensor([11264 / 2.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][17].potential_table(), torch.tensor([0.0, 1.0, 0.0])) # type:ignore
        self.assertIs(inference_results.messages_to_factors[1][18], None) # type:ignore

        torch.testing.assert_close(inference_results.messages_to_factors[1][19].potential_table(), torch.tensor([38400 / 4, 0.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][20].potential_table(), torch.tensor([0.0, 0.0, 480/2, 0.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][21].potential_table(), torch.tensor([11264 / 8.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][22].potential_table(), torch.tensor([1.0, 1.0, 1.0])) # type:ignore
        self.assertIs(inference_results.messages_to_factors[1][23], None) # type:ignore

        torch.testing.assert_close(inference_results.messages_to_factors[1][24].potential_table(), torch.tensor([38400 / 4.0, 480])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][25].potential_table(), torch.tensor([0.0, 0.0, 480, 0.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][26].potential_table(), torch.tensor([11264 / 4.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][27].potential_table(), torch.tensor([1.0, 1.0, 1.0])) # type:ignore
        self.assertIs(inference_results.messages_to_factors[1][28], None) # type:ignore

        torch.testing.assert_close(inference_results.messages_to_factors[1][29].potential_table(), torch.tensor([38400 / 4.0, 0.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][30].potential_table(), torch.tensor([0.0, 0.0, 480/2, 0.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][31].potential_table(), torch.tensor([11264/ 8.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][32].potential_table(), torch.tensor([1.0, 1.0, 1.0])) # type:ignore
        self.assertIs(inference_results.messages_to_factors[1][33], None) # type:ignore

        torch.testing.assert_close(inference_results.messages_to_factors[1][34].potential_table(), torch.tensor([38400, 0.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][35].potential_table(), torch.tensor([0.0, 0.0, 480/2.0, 0.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][36].potential_table(), torch.tensor([11264/ 2.0])) # type:ignore
        torch.testing.assert_close(inference_results.messages_to_factors[1][37].potential_table(), torch.tensor([0.0, 1.0, 0.0])) # type:ignore
        self.assertIs(inference_results.messages_to_factors[1][38], None) # type:ignore

        torch.testing.assert_close(inference_results.query_marginals[0].probability_table(), torch.tensor([1.0, 0]))
        torch.testing.assert_close(inference_results.query_marginals[1].probability_table(), torch.tensor([0.0,0.0,1.0,0.0]))
        torch.testing.assert_close(inference_results.query_marginals[2].probability_table(), torch.tensor([1.0]))
        torch.testing.assert_close(inference_results.query_marginals[3].probability_table(), torch.tensor([0.0, 1.0, 0]))
        torch.testing.assert_close(inference_results.query_marginals[4].probability_table(), torch.tensor([0.0, 0.0, 1.0]))
        torch.testing.assert_close(inference_results.query_marginals[5].probability_table(), torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(inference_results.query_marginals[6].probability_table(), torch.tensor([0.0, 0.0, 1.0]))
        torch.testing.assert_close(inference_results.query_marginals[7].probability_table(), torch.tensor([0.0, 1.0, 0]))
        self.assertEqual(inference_results.messages_to_factors[9][34].potential_table()[0].item(), math.inf) # type:ignore this will overflow


        inference_results = bp.inference(self.model, {8:4,9:6,10:0,11:6,12:4}, [0,1,2, 3,4,5,6,7], iterations=10, return_messages=True, renormalize=True)
        torch.testing.assert_close(inference_results.query_marginals[0].probability_table(), torch.tensor([1.0, 0]))
        torch.testing.assert_close(inference_results.query_marginals[1].probability_table(), torch.tensor([0.0,0.0,1.0,0.0]))
        torch.testing.assert_close(inference_results.query_marginals[2].probability_table(), torch.tensor([1.0]))
        torch.testing.assert_close(inference_results.query_marginals[3].probability_table(), torch.tensor([0.0, 1.0, 0]))
        torch.testing.assert_close(inference_results.query_marginals[4].probability_table(), torch.tensor([0.0, 0.0, 1.0]))
        torch.testing.assert_close(inference_results.query_marginals[5].probability_table(), torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(inference_results.query_marginals[6].probability_table(), torch.tensor([0.0, 0.0, 1.0]))
        torch.testing.assert_close(inference_results.query_marginals[7].probability_table(), torch.tensor([0.0, 1.0, 0]))
        self.assertLess(inference_results.messages_to_factors[9][34].potential_table()[0], math.inf) # type:ignore this will not overflow

if __name__ == '__main__':
    unittest.main()
