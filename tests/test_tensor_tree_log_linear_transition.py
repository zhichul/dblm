
import math
import unittest

import torch
from dblm.core import graph
from dblm.core.modeling import bayesian_networks, constants, factor_graphs, markov_networks
from dblm.core.modeling.interleaved_models import log_linear_transitions


class TestTensorTreeLogLinearTransition(unittest.TestCase):

    def setUp(self):
        self.sizez0 = 3
        self.nvals = [2,4,1] # 7 values for emission
        self.time = 5
        self.tree = graph.Graph(3)
        self.tree.add_edge(0,1)
        self.tree.add_edge(0,2)
        self.pgmz0 = markov_networks.TreeMRF(self.sizez0, self.nvals, constants.TensorInitializer.UNIFORM, tree=self.tree)
        self.pgmz0._factor_functions[0]._logits.data = torch.tensor([[0., 10., 20., 30.], [0, 0, 10, 20]]).log()
        self.pgmz0._factor_functions[1]._logits.data = torch.tensor([[10.], [1.]]).log()
        self.pgmztxt = log_linear_transitions.FixedLengthDirectedChainWithLogLinearTermFrequencyTransitionAndDeterministicEmission(self.sizez0, self.nvals, self.time, constants.TensorInitializer.UNIFORM)
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
        model = factor_graphs.FactorGraph.join(self.pgmz0.to_factor_graph_model(), self.pgmztxt.to_factor_graph_model(), {0:0,1:1,2:2})
        self.model = model

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
        assignments = torch.tensor([(0,0,0, 1,2, 0,0, 2,6, 1,2, 0,0), (1,3,0, 1,5, 2,6, 2,6, 1,5, 0,1),(1,3,0, 1,5, 0,1, 2,6, 1,5, 0,2)])
        torch.testing.assert_close(torch.zeros(3).fill_(-math.inf), self.model.energy(tuple(assignments[:,i] for i in range(assignments.size(1))))) # type:ignore impossible z0


        reference_log_unnormalized_likelihood = (math.log(20)
                               + math.log(1/3)
                               + 7 - torch.logsumexp(torch.tensor([7.,1001.]), dim=-1).item()
                               + 504 - torch.logsumexp(torch.tensor([4.,504.]), dim=-1).item()
                               + math.log(0.5)
                               + (1 + 2/4 + 6/2 + 7/4) - torch.logsumexp(torch.tensor([1 + 2/4 + 3 + 7/4, 1 + 2/4 + 500 + 7/4]), dim=-1).item())
        reference_log_likelihood = reference_log_unnormalized_likelihood - math.log(630) # that's the log partition of the z0 model, everything else is locally normalized
        # the default model is unnormalized
        assignment = torch.tensor([(1,3,0, 1,5, 0,1, 2,6, 1,5, 0,1)])
        self.assertAlmostEqual(reference_log_unnormalized_likelihood, self.model.energy(tuple(assignment[:,i] for i in range(assignment.size(1)))).item(), 3) # type:ignore
        # turn z0 into a table first, and create a bayesian network that enforces local normalization
        directed_model = bayesian_networks.BayesianNetwork.join(self.pgmz0.to_probability_table().to_bayesian_network(), self.pgmztxt, shared={0:0,1:1,2:2})
        self.assertAlmostEqual(reference_log_likelihood, directed_model.log_probability(tuple(assignment[:,i] for i in range(assignment.size(1)))).item(), 3) # type:ignore
        # turn z0 into table first, then the resulting factor graph is happens to be locally normalized
        undirected_normalized_model = factor_graphs.FactorGraph.join(self.pgmz0.to_probability_table().to_factor_graph_model(), self.pgmztxt, shared={0:0,1:1,2:2})
        assignments = torch.tensor([(1,3,0, 1,5, 0,1, 2,6, 1,5, 0,1),(1,3,0, 1,5, 0,1, 2,6, 1,5, 0,1)])
        torch.testing.assert_close(torch.tensor([reference_log_likelihood, reference_log_likelihood]), undirected_normalized_model.condition_on({4:torch.tensor([5,5]),6:torch.tensor([1,1]),8:torch.tensor([6,6]),10:torch.tensor([5,5]),12:torch.tensor([1,1])}).energy(tuple(assignments[:,i] for i in range(assignments.size(1))))) # type:ignore
        torch.testing.assert_close(torch.tensor([reference_log_likelihood, reference_log_likelihood]), factor_graphs.BPAutoregressiveIncompleteLikelihoodFactorGraph.from_factor_graph(self.model, [(3,4),(5,6),(7,8),(9,10),(11,12)]).log_marginal_probability([(4,torch.tensor([5,5])),(6,torch.tensor([1,1])),(8,torch.tensor([6,6])),(10,torch.tensor([5,5])),(12,torch.tensor([1,1]))]))

if __name__ == "__main__":
    unittest.main()
