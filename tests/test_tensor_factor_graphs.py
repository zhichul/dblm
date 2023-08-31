
import math
import unittest

import torch
from dblm.core import graph
from dblm.core.inferencers import belief_propagation

from dblm.core.modeling import chains, constants, factor_graphs, markov_networks, switching_tables

class TestTensorFactorGraphs(unittest.TestCase):

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
        self.pgmxt = switching_tables.BatchedSwitchingTables(self.sizez0, self.nvals, self.time)
        model = factor_graphs.FactorGraph.join(self.pgmz0.to_factor_graph_model(), self.pgmzt.to_factor_graph_model(), dict())
        model = factor_graphs.FactorGraph.join(model, self.pgmxt,
                                               dict([(i, i) for i in range(self.sizez0)] +
                                                    [(j,j) for j in range(self.sizez0, self.sizez0 + self.time)]))
        self.model = model

    def test_tensor_bp(self):
        bp = belief_propagation.FactorGraphBeliefPropagation()
        observation = torch.tensor(
            [[4,6,0,6,4],
             [4,6,0,6,4],
             [3,0,6,6,3],]
        )
        inference_results = bp.inference(self.model,
                                            observation={8:observation[:,0],9:observation[:,1],10:observation[:,2],11:observation[:,3],12:observation[:,4]},
                                            query=[0,1,2, 3,4,5,6,7],
                                            iterations=10,
                                            return_messages=True,
                                            renormalize=True)
        torch.testing.assert_close(inference_results.query_marginals[0].probability_table(), torch.tensor([[1.0, 0],[1.0, 0],[1.0, 0]]))
        torch.testing.assert_close(inference_results.query_marginals[1].probability_table(), torch.tensor([[0.0,0.0,1.0,0.0], [0.0,0.0,1.0,0.0], [0.0,1.0,0.0,0.0]]))
        torch.testing.assert_close(inference_results.query_marginals[2].probability_table(), torch.tensor([[1.0], [1.0], [1.0]]))
        torch.testing.assert_close(inference_results.query_marginals[3].probability_table(), torch.tensor([[0.0, 1.0, 0], [0.0, 1.0, 0], [0.0, 1.0, 0]]))
        torch.testing.assert_close(inference_results.query_marginals[4].probability_table(), torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))
        torch.testing.assert_close(inference_results.query_marginals[5].probability_table(), torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
        torch.testing.assert_close(inference_results.query_marginals[6].probability_table(), torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]))
        torch.testing.assert_close(inference_results.query_marginals[7].probability_table(), torch.tensor([[0.0, 1.0, 0], [0.0, 1.0, 0], [0.0, 1.0, 0]]))

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
        impossible_assignments = torch.tensor([(0,0,0, 1,2,0,2,1, 6,2,0,2,6),(1,3,0, 1,2,0,2,2, 5,6,1,6,6),(1,3,0, 1,2,0,2,1, 5,6,1,5,5)])
        torch.testing.assert_close(torch.tensor([0.,0,0]), self.model.unnormalized_probability(tuple(impossible_assignments[:,i] for i in range(impossible_assignments.size(1))))) # type:ignore impossible z0
        possible_assignments = torch.tensor([(1,3,0, 1,2,0,2,1, 5,6,1,6,5), (0,2,0, 1,2,0,2,1, 4,6,0,6,4)])
        torch.testing.assert_close(torch.tensor([20. * 1/81,200 * 1/81]), self.model.unnormalized_probability(tuple(possible_assignments[:,i] for i in range(impossible_assignments.size(1))))) # type:ignore

    def test_likelihood(self):
        table = self.model.to_potential_table()
        assignments = torch.tensor([(0,2,0, 1,2,0,2,1, 4,6,0,6,4)])
        torch.testing.assert_close(torch.tensor([math.log(200 * 1/81)]), table.log_potential_value(tuple(assignments[:,i] for i in range(assignments.size(1))))) # type:ignore
        table = self.model.to_probability_table()
        assignments = torch.tensor([(1,3,0, 1,2,0,2,1, 5,6,1,6,5), (0,2,0, 1,2,0,2,1, 4,6,0,6,4)])
        torch.testing.assert_close(torch.tensor([20/630 * 1/27, 200/630 * 1/27]), table.probability(tuple(assignments[:,i] for i in range(assignments.size(1))))) # type:ignore

if __name__ == "__main__":
    unittest.main()
