from __future__ import annotations
from collections import defaultdict
from dblm.core import graph
from dblm.rva.discrete import dist, tree_belief_propagation
import itertools
import torch
import math
import numpy as np

class TreeFactorGraph(dist.Distribution):

    def __init__(self,
                 tree: graph.Graph,
                 nT: int,
                 log_binary_potentials: list[torch.Tensor], # synced in order with tree.edges
                 log_unary_potentials: list[torch.Tensor]): # synced in order with tree.nodes
        super().__init__(tree.num_nodes, [nT] * tree.num_nodes)
        self.tree = tree
        self.nT = nT
        self.log_binary_potentials = log_binary_potentials # |E| x nT x nT
        self.log_unary_potentials = log_unary_potentials # |V| x nT x nT
        self.tbp = tree_belief_propagation.TreeBeliefPropagation()
        self.bp_done = False
        self.conditional_bp_done = False

    @torch.no_grad()
    def memoized_bp(self):
        if not self.bp_done:
            self.var2factors = defaultdict(list)
            for i in range(self.nvars):
                self.var2factors[i].append(i)
            for j, edge in enumerate(self.tree.edges):
                self.var2factors[edge.first_node_id].append(self.nvars + j)
                self.var2factors[edge.second_node_id].append(self.nvars + j)
            adjacency_matrix = torch.zeros((self.nvars, self.nvars + self.tree.num_edges), dtype=torch.long)
            for i in range(self.nvars):
                for fj in self.var2factors[i]:
                    adjacency_matrix[i, fj] = 1
            log_phis = self.log_unary_potentials + self.log_binary_potentials
            results = self.tbp.infer(adjacency_matrix, self.nvals, log_phis)
            self.adjacency_matrix = adjacency_matrix
            self.log_phis = log_phis
            self.bp_results = results
            self._log_marginals_vec = torch.cat(self.tbp.marginals(self.bp_results))
            self.bp_done = True

    @torch.no_grad()
    def memoized_conditional_bp(self):
        self.memoized_bp()
        if not self.conditional_bp_done:
            self.conditional_bp_results = []
            _log_marginals_conditioned_on_one = []
            for i in range(self.nvars):
                for val in range(self.nvals[i]):
                    new_log_phi = torch.zeros(self.nT).fill_(-math.inf)
                    new_log_phi[val] = 0
                    new_adj_col = torch.zeros(self.nvars, dtype=torch.long)
                    new_adj_col[i] = 1
                    new_adjacency_matrix = torch.cat([self.adjacency_matrix, new_adj_col[:,None]], dim=1)
                    results = self.tbp.infer(new_adjacency_matrix, self.nvals, [*self.log_phis, new_log_phi])
                    self.conditional_bp_results.append(results)
                    _log_marginals_conditioned_on_one.append(torch.cat(self.tbp.marginals(results)))
            self._log_marginals_conditioned_on_one = torch.stack(_log_marginals_conditioned_on_one)
            self.conditional_bp_done = True

    @torch.no_grad()
    def sample(self, n: int, encoding="binary"):
        self.memoized_bp()
        samples = [None] * self.nvars
        order = self.tree.dfs(leaf_as_root=True)
        edges = self.tree.node2edge_index()
        # print(order)
        for i, parent in order:
            # print(i, parent, [j for j in self.var2factors[i] if parent is None or j != edges[(i, parent)].id + self.nvars])
            messages_in = sum(self.bp_results.msg_v_from_f[i, j] for j in self.var2factors[i] if parent is None or j != edges[(i, parent)].id + self.nvars).expand(n, self.nT) #  type:ignore
            if parent is None:
                parent_message = 0
            elif parent < i:
                parent_message = self.log_binary_potentials[edges[(i, parent)].id][samples[parent], :] # n, nT
            else:
                parent_message = self.log_binary_potentials[edges[(i, parent)].id][:, samples[parent]].t() # n, nT
            m = torch.distributions.Categorical(logits=messages_in+parent_message)
            samples[i] = m.sample() # type:ignore
        if encoding == "binary":
            return self.int_to_bin(torch.stack(samples,dim=1).numpy()) # type:ignore
        elif encoding == "integer":
            return torch.stack(samples,dim=1).numpy() # type:ignore
        else:
            raise NotImplementedError(encoding)

    @torch.no_grad()
    def log_probability_table(self, encoding="binary"):
        expansion_size = [self.nT] * self.nvars
        cum = []
        for i, edge in enumerate(self.tree.edges):
            shape = [1] * self.nvars
            shape[edge.first_node_id] = self.nT
            shape[edge.second_node_id] = self.nT
            cum.append(self.log_binary_potentials[i].view(shape).expand(expansion_size))
        for i in range(self.nvars):
            shape = [1] * self.nvars
            shape[i] = self.nT
            cum.append(self.log_unary_potentials[i].view(shape).expand(expansion_size))
        prob_vec = torch.stack(cum).sum(dim=0).reshape(-1).log_softmax(-1).numpy()
        assignment_mat = []
        for assignments in itertools.product(*[range(nval) for nval in self.nvals]):
            assignment_mat.append(assignments)
        assignment_mat = np.array(assignment_mat)
        if encoding == "binary":
            return prob_vec, self.int_to_bin(assignment_mat)
        elif encoding == "integer":
            return prob_vec, assignment_mat
        else:
            raise NotImplementedError(encoding)

    @torch.no_grad()
    def log_marginals(self):
        self.memoized_bp()
        return self._log_marginals_vec.numpy()

    @torch.no_grad()
    def log_marginals_conditioned_on_one(self):
        self.memoized_conditional_bp()
        return self._log_marginals_conditioned_on_one.numpy()

    @torch.no_grad()
    def backoff_log_marginals_conditioned_on_one(self):
        out = np.broadcast_to(self.log_marginals()[None,:], (self.nv, self.nv)).copy() # type:ignore
        for i in range(self.nvars):
            s, e = self.start_end(i)
            out[s:e, s:e] = np.log(np.eye(self.nvals[i]))
        return out

    def log_marginals_conditioned_on_two(self):
        raise NotImplementedError()

if __name__ == "__main__":
    g = graph.Graph(6)
    nT = 2
    g.add_edge(0, 2)
    g.add_edge(2, 1)
    g.add_edge(2, 3)
    g.add_edge(4, 3)
    g.add_edge(3, 5)
    log_binary_potentials = [
        [[0.23, 0.77],
         [0.65, 0.35]],
        [[0.81, 0.19],
         [0.76, 0.3]],
        [[0.9, 0.0],
         [0.1, 0.9]],
        [[1.1, 2.2],
         [3.3, 4.4]],
        [[1.5, 2.5],
          [0.5, 0.1]]
    ]
    log_unary_potentials = [
        [5, 6.],
        [1., 2.],
        [2., 1.],
        [3., 4.],
        [1., 3.],
        [3., 8.]
    ]
    log_binary_potentials = [torch.tensor(p).log() for p in log_binary_potentials]
    log_unary_potentials = [torch.tensor(p).log() for p in log_unary_potentials]
    t = TreeFactorGraph(g, nT, log_binary_potentials, log_unary_potentials)
    log_prob_table = t.log_probability_table()[0]
    logups = []
    Z0s = [[] for _ in range(6)]
    Z1s = [[] for _ in range(6)]
    for assignment in itertools.product(*([0,1] for _ in range(6))):
        logup = 0
        logup += log_binary_potentials[0][assignment[0], assignment[2]].item()
        logup += log_binary_potentials[1][assignment[1], assignment[2]].item()
        logup += log_binary_potentials[2][assignment[2], assignment[3]].item()
        logup += log_binary_potentials[3][assignment[3], assignment[4]].item()
        logup += log_binary_potentials[4][assignment[3], assignment[5]].item()

        logup += log_unary_potentials[0][assignment[0]].item()
        logup += log_unary_potentials[1][assignment[1]].item()
        logup += log_unary_potentials[2][assignment[2]].item()
        logup += log_unary_potentials[3][assignment[3]].item()
        logup += log_unary_potentials[4][assignment[4]].item()
        logup += log_unary_potentials[5][assignment[5]].item()
        logups.append(logup)
        for j, ass in enumerate(assignment):
            if assignment[j] == 0:
                Z0s[j].append(logup)
            else:
                Z1s[j].append(logup)
    logZ = torch.logsumexp(torch.tensor(logups), -1).item()
    logZ0s = [torch.tensor(logZ0j).logsumexp(-1).item() for logZ0j in Z0s]
    logZ1s = [torch.tensor(logZ1j).logsumexp(-1).item() for logZ1j in Z1s]
    logmarginal0s = np.array([logZ0j - logZ for logZ0j in logZ0s])
    logmarginal1s = np.array([logZ1j - logZ for logZ1j in logZ1s])
    logps = np.array([logup - logZ  for logup in logups])
    np.testing.assert_allclose(logps, log_prob_table, rtol=1e-3, atol=1e-3)
    lms = t.log_marginals()
    np.testing.assert_allclose(lms[[0,2,4,6,8,10]], logmarginal0s, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(lms[[1,3,5,7,9,11]], logmarginal1s, rtol=1e-3, atol=1e-3)
    # marginals_from_samples = t.sample(10_000_000).mean(axis=0)
    # np.testing.assert_allclose(np.log(marginals_from_samples), lms, rtol=1e-3, atol=1e-3)




    # test conditional marginals
    lmo = t.log_marginals_conditioned_on_one()
    for i in range(6):
        for val in [0, 1]:
            logups = []
            Z0s = [[] for _ in range(6)]
            Z1s = [[] for _ in range(6)]
            for assignment in itertools.product(*([0,1] for _ in range(6))):
                if assignment[i] != val: continue
                logup = 0
                logup += log_binary_potentials[0][assignment[0], assignment[2]].item()
                logup += log_binary_potentials[1][assignment[1], assignment[2]].item()
                logup += log_binary_potentials[2][assignment[2], assignment[3]].item()
                logup += log_binary_potentials[3][assignment[3], assignment[4]].item()
                logup += log_binary_potentials[4][assignment[3], assignment[5]].item()

                logup += log_unary_potentials[0][assignment[0]].item()
                logup += log_unary_potentials[1][assignment[1]].item()
                logup += log_unary_potentials[2][assignment[2]].item()
                logup += log_unary_potentials[3][assignment[3]].item()
                logup += log_unary_potentials[4][assignment[4]].item()
                logup += log_unary_potentials[5][assignment[5]].item()
                logups.append(logup)
                for j, ass in enumerate(assignment):
                    if assignment[j] == 0:
                        Z0s[j].append(logup)
                    else:
                        Z1s[j].append(logup)
            logZ = torch.logsumexp(torch.tensor(logups), -1).item()
            logZ0s = [torch.tensor(logZ0j).logsumexp(-1).item() for logZ0j in Z0s]
            logZ1s = [torch.tensor(logZ1j).logsumexp(-1).item() for logZ1j in Z1s]
            logmarginal0s = np.array([logZ0j - logZ for logZ0j in logZ0s])
            logmarginal1s = np.array([logZ1j - logZ for logZ1j in logZ1s])
            np.testing.assert_allclose(lmo[2*i + val,[0,2,4,6,8,10]], logmarginal0s, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(lmo[2*i + val,[1,3,5,7,9,11]], logmarginal1s, rtol=1e-3, atol=1e-3)
