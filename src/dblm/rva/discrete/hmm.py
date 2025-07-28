from __future__ import annotations
import itertools
import math
import numpy as np
import torch

from dblm.rva.discrete import dist
from dblm.rva.discrete import tree_belief_propagation


class HMMPosterior(dist.Distribution):

    def __init__(self,
                 nvars: int,
                 nT: int,
                 log_transition: torch.Tensor, # internal transitions
                 log_emissions: list[torch.Tensor],
                 log_bos_transition: torch.Tensor,
                 log_eos_transition: torch.Tensor) -> None:
        super().__init__(nvars, [nT] * nvars)
        self.nT = nT
        self.log_transition = log_transition # nT x nT
        self.log_emissions = log_emissions # nvar x [nT]
        self.log_bos_transition = log_bos_transition
        self.log_eos_transition = log_eos_transition
        self.tbp = tree_belief_propagation.TreeBeliefPropagation()
        self.bp_done = False
        self.conditional_bp_done = False

    @torch.no_grad()
    def memoized_bp(self):
        if not self.bp_done:
            adjacency_matrix = torch.zeros((self.nvars, self.nvars + 1 + self.nvars), dtype=torch.long)
            for i in range(self.nvars):
                adjacency_matrix[i][i] = 1
                adjacency_matrix[i][i+1] = 1
                adjacency_matrix[i][self.nvars+1+i] = 1
            log_phis = [self.log_bos_transition] + [self.log_transition] * (self.nvars -1)+ [self.log_eos_transition] + self.log_emissions
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
        samples = []
        for i in range(self.nvars):
            l2r = self.bp_results.msg_v_from_f[0, 0].expand(n, self.nT) if i == 0 else self.log_transition[samples[-1]]
            r2l = self.bp_results.msg_f_from_v[i,i]
            m = torch.distributions.Categorical(logits=r2l+l2r)
            samples.append(m.sample())
        if encoding == "binary":
            return self.int_to_bin(torch.stack(samples,dim=1).numpy())
        elif encoding == "integer":
            return torch.stack(samples,dim=1).numpy()
        else:
            raise NotImplementedError(encoding)

    @torch.no_grad()
    def log_probability_table(self, encoding="binary"):
        expansion_size = [self.nT] * self.nvars
        cum = []
        for i in range(self.nvars-1):
            shape = [1] * self.nvars
            shape[i] = self.nT
            shape[i+1] = self.nT
            cum.append(self.log_transition.view(shape).expand(expansion_size))
        cum.append(self.log_bos_transition.view(-1, *([1] * (self.nvars-1))).expand(expansion_size))
        cum.append(self.log_eos_transition.view(*([1] * (self.nvars-1)), -1).expand(expansion_size))
        for i in range(self.nvars):
            shape = [1] * self.nvars
            shape[i] = self.nT
            cum.append(self.log_emissions[i].view(shape).expand(expansion_size))
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
    log_transition_matrix = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.2],
                         [0.3, 0.4, 0.1, 0.0],
                         [0.2, 0.2, 0.1, 0.2],
                         [0.1, 0.0, 0.5, 0.3],])) #  sample from Delta^nT+1 then slice the first nT columns
    log_eos_transition = torch.log(torch.tensor([0.2, 0.2, 0.3, 0.1]))
    log_bos_transition = torch.log(torch.tensor([0.3, 0.1, 0.1, 0.4]))
    log_emission_matrix = torch.tensor([[0.1, 0.3, 0.5, 0.0, 0.1],
                         [0.3, 0.5, 0.1, 0.0, 0.1],
                         [0.2, 0.2, 0.2, 0.2, 0.2],
                         [0.4, 0.1, 0.4, 0.1, 0.0],])
    log_emissions = [log_emission_matrix[:,i] for i in [1,2,3,4,0]]
    dist = HMMPosterior(5, 4, log_transition_matrix, log_emissions, log_bos_transition, log_eos_transition)
    mat_int = dist.sample(3, encoding="integer")
    mat_bin = dist.int_to_bin(mat_int)
    mat_int_again = dist.bin_to_int(mat_bin)
    print(mat_int)
    print(mat_bin)
    print(mat_int_again)
    print(np.round(np.exp(dist.log_marginals()), 2))
    print(np.round(np.exp(dist.log_marginals_conditioned_on_one()), 2))

    alpha0 = log_bos_transition + log_emissions[0]
    alpha1 = (alpha0[:, None] + log_transition_matrix).logsumexp(dim=0) + log_emissions[1]
    alpha2 = (alpha1[:, None] + log_transition_matrix).logsumexp(dim=0) + log_emissions[2]
    alpha3 = (alpha2[:, None] + log_transition_matrix).logsumexp(dim=0) + log_emissions[3]
    alpha4 = (alpha3[:, None] + log_transition_matrix).logsumexp(dim=0) + log_emissions[4]
    logZa = (alpha4 + log_eos_transition).logsumexp(-1)
    beta4 = log_eos_transition + log_emissions[-1]
    beta3 = (beta4[None, :] + log_transition_matrix).logsumexp(dim=1) + log_emissions[-2]
    beta2 = (beta3[None, :] + log_transition_matrix).logsumexp(dim=1) + log_emissions[-3]
    beta1 = (beta2[None, :] + log_transition_matrix).logsumexp(dim=1) + log_emissions[-4]
    beta0 = (beta1[None, :] + log_transition_matrix).logsumexp(dim=1) + log_emissions[-5]
    logZb = (beta0 + log_bos_transition).logsumexp(-1)
    marginals = [((alpha0 + beta0 - log_emissions[0])-logZa).exp().tolist(),
                 ((alpha1 + beta1 - log_emissions[1])-logZa).exp().tolist(),
                 ((alpha2 + beta2 - log_emissions[2])-logZa).exp().tolist(),
                 ((alpha3 + beta3 - log_emissions[3])-logZa).exp().tolist(),
                 ((alpha4 + beta4 - log_emissions[4])-logZa).exp().tolist()]
    print(np.round(np.array(sum(marginals, [])), 2))
    alpha0 = log_bos_transition + log_emissions[0]
    alpha1 = (alpha0[:, None] + log_transition_matrix).logsumexp(dim=0) + log_emissions[1]
    alpha2 = (alpha1[:, None] + log_transition_matrix).logsumexp(dim=0) + torch.log(torch.tensor([0,0,1,0.]))
    alpha3 = (alpha2[:, None] + log_transition_matrix).logsumexp(dim=0) + log_emissions[3]
    alpha4 = (alpha3[:, None] + log_transition_matrix).logsumexp(dim=0) + log_emissions[4]
    logZa = (alpha4 + log_eos_transition).logsumexp(-1)
    beta4 = log_eos_transition + log_emissions[-1]
    beta3 = (beta4[None, :] + log_transition_matrix).logsumexp(dim=1) + log_emissions[-2]
    beta2 = (beta3[None, :] + log_transition_matrix).logsumexp(dim=1) + torch.log(torch.tensor([0,0,1,0.]))
    beta1 = (beta2[None, :] + log_transition_matrix).logsumexp(dim=1) + log_emissions[-4]
    beta0 = (beta1[None, :] + log_transition_matrix).logsumexp(dim=1) + log_emissions[-5]
    logZb = (beta0 + log_bos_transition).logsumexp(-1)
    print(logZa, logZb)
    marginals = [((alpha0 + beta0 - log_emissions[0])-logZa).exp().tolist(),
                 ((alpha1 + beta1 - log_emissions[1])-logZa).exp().tolist(),
                 ((alpha2 + beta2 - torch.log(torch.tensor([0,0,1,0.])))-logZa).exp().tolist(),
                 ((alpha3 + beta3 - log_emissions[3])-logZa).exp().tolist(),
                 ((alpha4 + beta4 - log_emissions[4])-logZa).exp().tolist()]
    print(np.round(np.array(sum(marginals, [])), 2))

    pv, ass = dist.log_probability_table()
    torch.testing.assert_close((-np.exp(pv) * pv)[~np.isnan(-np.exp(pv) * pv)].sum(), dist.tbp.entropy(dist.adjacency_matrix, dist.log_phis, dist.bp_results))

