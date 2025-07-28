from __future__ import annotations
import bisect
from collections import defaultdict
import dataclasses
import itertools
import math
import sys
import numpy as np
import torch
import tqdm


def int_to_bin(nvars, nvals, mat):
    if not (len(mat.size()) == 2 and mat.size(1) == nvars):
        raise ValueError(mat.size(), nvars)
    cols = []
    for i in range(nvars):
        cols_bin = torch.zeros((mat.size(0), nvals[i]))
        cols_bin[torch.arange(mat.size(0)), mat[:,i]] = 1
        cols.append(cols_bin)
    return torch.cat(cols, dim=1)

def int_to_bin_batched(nvars, nvals, mat:torch.Tensor):
    if not (len(mat.size()) >= 2 and mat.size(-1) == nvars):
        raise ValueError(mat.size(), nvars)
    bin = mat.new_zeros((*tuple(mat.size()[:-1]), sum(nvals)), dtype=torch.bool)
    mat = mat.to(torch.long)
    offset = 0
    offsets = []
    for nval in nvals:
        offsets.append(offset)
        offset += nval
    if torch.iinfo(mat.dtype).max < offset:
        raise ValueError(f"Exiting due to overflow possibility: \n{torch.iinfo(mat.dtype)}\n{offset}\n{offsets}")
    offsets_mat = mat.new_tensor(offsets).expand_as(mat)
    mat = mat + offsets_mat
    bin.scatter_(-1, mat, 1)
    return bin

def topo_sort(V, E):
    s2t = defaultdict(list)
    t2s = defaultdict(list)
    for i, j  in E:
        s2t[i].append(j)
        t2s[j].append(i)
    in_degs = {i:len(t2s[i]) for i in V}
    topo = []
    frontier = [i for i in V if in_degs[i] == 0]
    while len(frontier) > 0:
        i = frontier.pop()
        topo.append(i)
        for j in s2t[i]:
            in_degs[j] -= 1
            if in_degs[j] == 0:
                frontier.append(j)
    if len(topo) < len(V):
        raise ValueError("Input graph is loopy.")
    return topo

@dataclasses.dataclass
class BPInferenceResult:

    msg_f_from_v: np.ndarray
    msg_v_from_f: np.ndarray

def observation_encoding(varname, obs, curr_adjacency_matrix, n, nvals):
    """
    var index,
    list of observations,
    current adjadency matrix  (encoding current factor graph),
    number of vars,
    number of vals for current var
    """
    observation_adj_column = curr_adjacency_matrix.new_zeros((n,1))
    observation_adj_column[varname, 0] = 1
    observation_log_phi = curr_adjacency_matrix.new_empty(obs.numel(), nvals, dtype=torch.float).fill_(-math.inf)
    observation_log_phi.scatter_(1, obs.unsqueeze(-1), 0.0)
    return observation_adj_column, observation_log_phi

class TreeBeliefPropagation:

    def energy(self, adjacency_matrix: torch.Tensor, nvals: list[int], log_phis: list[torch.Tensor], assignments: torch.Tensor):
        batch_dims = list(log_phis[0].size()[:-adjacency_matrix[:,0].sum().item()])
        if len(batch_dims) == 1:
            if batch_dims[0] != assignments.size(0):
                raise ValueError()
        elif len(batch_dims) == 0:
            pass
        else:
            raise ValueError()
        nvars = assignments.size(-1)
        if nvars != len(nvals):
            raise ValueError()
        out_size = assignments.size()[:-1] # drop last dim
        assignments = assignments.reshape(*(*batch_dims, -1, nvars)) # flatten each MRF's batch to 1dim
        energy = torch.tensor(0.0)
        for i, log_phi in enumerate(log_phis):
            index = []
            for j, is_in_factor in enumerate(adjacency_matrix[:, i]):
                if is_in_factor.item():
                    ass = assignments[:, :, j] if len(batch_dims) > 0 else assignments[:, j]
                    if (ass < 0).any() or (ass>=nvals[j]).any():
                        raise ValueError()
                    index.append(ass)
            index = ((torch.arange(batch_dims[0])[:, None].expand(batch_dims[0], assignments.size(1)),) if len(batch_dims) > 0 else tuple()) + tuple(index)
            energy = energy + log_phi[index]
        return energy.reshape(out_size)

    def sample(self, adjacency_matrix: torch.Tensor, nvals: list[int], log_phis: list[torch.Tensor], n:int, observations:dict[int, torch.Tensor]=None, device="cuda") -> torch.Tensor: # type:ignore
        log_phi_batch_dims = list(log_phis[0].size()[:-adjacency_matrix[:,0].sum().item()])
        if len(log_phi_batch_dims) > 0:
            raise NotImplementedError("Currently batching cannot be done in log_phis, do so in observations")
        if observations is not None and any(len(v.size()) > 1 for v in observations.values()):
            raise ValueError("observations must be zero or one dimensional")
        if observations is not None and len({v.numel() for v in observations.values()}) > 1:
            raise ValueError("variables must have same number of assignments, they're treated as batch ")
        batch_size = n
        if observations is not None:
            obs = observations
            observations = dict()
            for k, v in obs.items():
                observations[k] = v.to(device).repeat_interleave(n)
                batch_size = v.numel() * n
        else:
            observations = dict()
        log_phis = [lp.to(device).unsqueeze(0).expand(*(batch_size, *lp.size())) for lp in log_phis]
        adjacency_matrix = adjacency_matrix.to(device)

        # now we add conditioning
        # n =  adjacency_matrix.size(0)
        # obs_adjs = []
        # obs_log_phis = []
        # for varname, obs in enumerate(observations.items()):
        #     obs_adj, obs_log_phi = observation_encoding(varname, obs, adjacency_matrix, n, nvals[varname])
        #     obs_adjs.append(obs_adj)
        #     obs_log_phis.append(obs_log_phi)
        # adjacency_matrix = torch.cat([adjacency_matrix, *obs_adjs], dim=1)
        # log_phis = log_phis + obs_log_phis

        # now we can sample
        samples = []
        for i in range(adjacency_matrix.size(0)):
            if i in observations:
                samples.append(observations[i].unsqueeze(1))
            else:
                marginals = self.marginals(self.infer(adjacency_matrix, nvals, log_phis, observations=observations))[i]
                sample_i = torch.distributions.Categorical(logits=marginals).sample((1,)).reshape(-1) # type:ignore
                samples.append(sample_i.unsqueeze(1))
                observations[i] = sample_i
                del marginals
                # obs_adj, obs_log_phi = observation_encoding(i, sample_i, adjacency_matrix, n, nvals[i])
                # adjacency_matrix = torch.cat([adjacency_matrix, obs_adj], dim=1)
                # log_phis = [*log_phis, obs_log_phi]
        return torch.cat(samples, dim=1)

    def infer(self, adjacency_matrix: torch.Tensor, nvals: list[int], log_phis: list[torch.Tensor], observations=None):
        """`adjacency_matrix` is a vertex by factor matrix where A_{ij}
        is 0 if and only if Vi is connected to Fj.

        nvals is the number of values for each variable.

        log_phis is the log potential tables for the factor functions.
        """
        device = adjacency_matrix.device
        batch_dims = list(log_phis[0].size()[:-adjacency_matrix[:,0].sum().item()])
        n =  adjacency_matrix.size(0)
        if observations is not None and len(observations) > 0:
            if any(len(v.size()) > 1 for v in observations.values()):
                print("Observations must be zero or one dimensional", file=sys.stderr)
                breakpoint()
            if len({v.numel() for v in observations.values()}) > 1:
                print("variables must have same number of assignments, they're treated as batch", file=sys.stderr)
                breakpoint()
            if len(batch_dims) > 0 and not (len(batch_dims)==1 and batch_dims[0] == next(iter(observations.values())).numel()):
                print("Currently log_phis needs to either not have batch_dim or match observation", file=sys.stderr)
                breakpoint()
            # add conditioning to the factor graph
            if len(batch_dims) == 0:
                log_phis = [lp.unsqueeze(0).expand(*(next(iter(observations.values())).numel(), *lp.size())) for lp in log_phis]
            obs_adjs = []
            obs_log_phis = []
            for varname, obs in observations.items():
                obs_adj, obs_log_phi = observation_encoding(varname, obs, adjacency_matrix, n, nvals[varname])
                obs_adjs.append(obs_adj)
                obs_log_phis.append(obs_log_phi)
            adjacency_matrix = torch.cat([adjacency_matrix, *obs_adjs], dim=1)
            log_phis = log_phis + obs_log_phis
            batch_dims = list(log_phis[0].size()[:-adjacency_matrix[:,0].sum().item()])

        m =  adjacency_matrix.size(1)
        v2f = defaultdict(list)
        f2v = defaultdict(list)
        for vi, fj in torch.nonzero(adjacency_matrix).tolist():
            i = vi
            j = n + fj
            v2f[i].append(j)
            f2v[j].append(i)
        def msg2i(i, j): return i * (m+n) + j
        def ismsgv2f(msg): return (msg // (m+n)) < n
        def ismsgf2v(msg): return (msg // (m+n)) >= n
        def i2msg(i): return i//(m+n), i % (m+n)
        def msg2str(i, j): return f"{'v' if i < n else 'f'}{i if i < n else i - n} -> {'v' if j < n else 'f'}{j if j < n else j - n}"
        uniform_msg_memoizer = dict()
        def uniform_msg(k, normalize=True, memoize=True): 
            if (k, normalize, memoize) not in uniform_msg_memoizer:
                if normalize:
                    uniform_msg_memoizer[(k, normalize, memoize)] = torch.zeros(batch_dims + [k]).to(device).log_softmax(-1)
                else:
                    uniform_msg_memoizer[(k, normalize, memoize)] = torch.zeros(batch_dims + [k]).to(device)
            return uniform_msg_memoizer[(k, normalize, memoize)]
        messages = [msg2i(i, j) for (i, l) in v2f.items() for j in l] + [msg2i(j, i) for (i, l) in v2f.items() for j in l]
        dependencies = []
        for i in range(n):
            fs = v2f[i]
            for f, fprime in itertools.product(fs, fs):
                if f == fprime: continue
                dependencies.append((msg2i(fprime, i), msg2i(i, f)))
        for fj in range(m):
            j = n + fj
            vs = f2v[j]
            for v, vprime in itertools.product(vs, vs):
                if v == vprime: continue
                dependencies.append((msg2i(vprime, j), msg2i(j, v)))
        # print([(msg2str(*i2msg(i)),msg2str(*i2msg(j)))  for i, j in dependencies])
        # print([msg2str(*i2msg(i)) for i in topo_sort(messages, dependencies)])
        msg_2_dependency = defaultdict(list)
        msg_val = dict()
        for m1, m2 in dependencies:
            msg_2_dependency[m2].append(m1)
        bporder = topo_sort(messages, dependencies)
        for bpi, msg in tqdm.tqdm(enumerate(bporder), leave=False):
            in_msgs = sorted(msg_2_dependency[msg])
            if ismsgv2f(msg):
                i, j = i2msg(msg)
                k = nvals[i]
                if len(in_msgs) == 0:
                    msg_val[msg] = uniform_msg(k)
                else:
                    msg_val[msg] = torch.stack([msg_val[in_msg] for in_msg in in_msgs]).sum(dim=0).log_softmax(-1)
            else:
                assert ismsgf2v(msg)
                j, i = i2msg(msg)
                k = nvals[i]
                log_phi = log_phis[j-n]
                expansion_size = tuple(log_phi.size())
                pos = bisect.bisect(in_msgs, msg2i(i, j))
                if len(in_msgs) == 0:
                    msg_val[msg] = log_phi.log_softmax(-1)
                else:
                    product = []
                    for idx, in_msg in enumerate(in_msgs):
                        shape = batch_dims + [1] * (len(in_msgs) + 1)
                        shape[len(batch_dims) + idx + int(idx >= pos)] = -1
                        product.append(msg_val[in_msg].view(shape).expand(expansion_size))
                    product.append(log_phi)
                    msg_val[msg] = torch.stack(product, dim=0).sum(dim=0).transpose(len(batch_dims) + pos, -1).reshape(*batch_dims, -1, nvals[i]).logsumexp(-2).log_softmax(-1)
            # print(bpi, msg2str(*i2msg(msg)))
            # print(msg_val[msg].exp())
            # print("####")

        msg_v2f = np.ndarray((m,n), dtype=object)
        msg_f2v = np.ndarray((n,m), dtype=object)
        for msg in bporder:
            i, j = i2msg(msg)
            if ismsgv2f(msg):
                msg_v2f[j-n, i] = msg_val[msg]
            else:
                assert ismsgf2v(msg)
                msg_f2v[j, i-n] = msg_val[msg]
        return BPInferenceResult(msg_v2f, msg_f2v)

    def marginals(self, inference: BPInferenceResult):
        n, m = inference.msg_v_from_f.shape
        out = []
        for i in range(n):
            cum = []
            for j in range(m):
                msg = inference.msg_v_from_f[i,j]
                if msg is not None:
                    cum.append(msg)
            out.append(torch.stack(cum).sum(dim=0).log_softmax(-1)) # type:ignore
        return out

    def marginals_factors(self, adjacency_matrix: torch.Tensor, log_phis: list[torch.Tensor], inference: BPInferenceResult):
        batch_dims = list(log_phis[0].size()[:-adjacency_matrix[:,0].sum().item()])
        m, n = inference.msg_f_from_v.shape
        out = []
        for i in range(m):
            log_phi = log_phis[i]
            expansion_size = log_phi.size()
            cum = []
            idx = 0
            for j in range(n):
                msg = inference.msg_f_from_v[i, j]
                if msg is not None:
                    shape = batch_dims + [1] * (len(expansion_size) - len(batch_dims))
                    shape[len(batch_dims) + idx] = -1
                    idx += 1
                    cum.append(msg.view(shape).expand(expansion_size))
            cum.append(log_phi)
            out.append(torch.stack(cum).sum(dim=0).reshape(*batch_dims, -1).log_softmax(-1).reshape(expansion_size)) # type:ignore
        return out

    def entropy(self, adjacency_matrix: torch.Tensor, log_phis: list[torch.Tensor], inference: BPInferenceResult):
        batch_dims = list(log_phis[0].size()[:-adjacency_matrix[:,0].sum().item()])
        marginals = self.marginals(inference)
        marginals_factors = self.marginals_factors(adjacency_matrix, log_phis, inference)
        def entropy(logp):
            return torch.distributions.Categorical(logits=logp).entropy()
        v_entropies = [entropy(marginal) for marginal in marginals]
        v_degrees = adjacency_matrix.sum(dim=1).tolist()
        Hv = sum(v_ent * (1-v_deg) for v_ent, v_deg in zip(v_entropies, v_degrees))

        f_entropies = [entropy(marginal_factor.reshape(*batch_dims, -1)) for marginal_factor in marginals_factors]
        Hf = sum(f_ent for f_ent in f_entropies)
        if (len(Hf.size()) == 0): # type:ignore
            return (Hf + Hv).item() # type:ignore
        return Hf + Hv # type:ignore

    def log_partition(self, adjacency_matrix: torch.Tensor, log_phis: list[torch.Tensor], inference: BPInferenceResult):
        ent = self.entropy(adjacency_matrix, log_phis, inference)
        marginals_factors = self.marginals_factors(adjacency_matrix, log_phis, inference)

def test():
    adjacency_matrix = torch.tensor([
        [1,0,0,0],
        [1,1,0,0],
        [1,0,1,0],
        [0,0,1,1]
    ])
    f1 = torch.log(torch.tensor(
        [0.1,0.3,0.2,0.1,0.05,0.0,0.0,0.25] + [0.1,0.3,0.2,0.1,0.05,0.0,0.0,0.25]
    )).reshape(2,2,2,2)
    f2 = torch.log(torch.tensor(
        [1, 1.] + [1, 3.]
    )).reshape(2,2 )
    f3 = torch.log(torch.tensor(
        [4, 1., 1., 4] + [4, 1., 1., 4]
    )).reshape(2,2,2)
    f4 = torch.log(torch.tensor(
        [2 * 1.3, 8 * 1.3] + [2 * 1.3, 8 * 1.3]
    )).reshape(2, 2)
    tbp = TreeBeliefPropagation()
    log_phis = [f1, f2, f3, f4]
    results = tbp.infer(adjacency_matrix, [2,2,2,2], log_phis)
    torch.testing.assert_close(results.msg_v_from_f[3, 3].exp(),torch.tensor([[0.2000, 0.8000],
            [0.2000, 0.8000]]),atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(results.msg_f_from_v[2, 3].exp(),torch.tensor([[0.2000, 0.8000],
            [0.2000, 0.8000]]),atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(results.msg_v_from_f[2, 2].exp(),torch.tensor([[0.3200, 0.6800],
            [0.3200, 0.6800]]),atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(results.msg_f_from_v[0, 2].exp(),torch.tensor([[0.3200, 0.6800],
            [0.3200, 0.6800]]),atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(results.msg_v_from_f[1, 1].exp(),torch.tensor([[0.5000, 0.5000],
            [0.2500, 0.7500]]),atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(results.msg_f_from_v[0, 1].exp(),torch.tensor([[0.5000, 0.5000],
            [0.2500, 0.7500]]),atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(results.msg_v_from_f[0, 0].exp(),torch.tensor([[0.6643, 0.3357],
            [0.5458, 0.4542]]),atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(results.msg_f_from_v[0, 0].exp(),torch.tensor([[0.5000, 0.5000],
            [0.5000, 0.5000]]),atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(results.msg_v_from_f[2, 0].exp(),torch.tensor([[0.3500, 0.6500],
            [0.3571, 0.6429]]),atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(results.msg_f_from_v[2, 2].exp(),torch.tensor([[0.3500, 0.6500],
            [0.3571, 0.6429]]),atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(results.msg_v_from_f[3, 2].exp(),torch.tensor([[0.4100, 0.5900],
            [0.4143, 0.5857]]),atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(results.msg_f_from_v[3, 3].exp(),torch.tensor([[0.4100, 0.5900],
            [0.4143, 0.5857]]),atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(results.msg_v_from_f[1, 0].exp(),torch.tensor([[0.4549, 0.5451],
            [0.4549, 0.5451]]),atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(results.msg_f_from_v[1, 1].exp(),torch.tensor([[0.4549, 0.5451],
            [0.4549, 0.5451]]),atol=1e-4,rtol=1e-3)
    m = tbp.marginals(results)
    torch.testing.assert_close(m[0].exp(), torch.tensor([[0.6643, 0.3357],
        [0.5458, 0.4542]]),atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(m[1].exp(), torch.tensor([[0.4549, 0.5451],
            [0.2176, 0.7824]]),atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(m[2].exp(), torch.tensor([[0.2022, 0.7978],
            [0.2073, 0.7927]]),atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(m[3].exp(), torch.tensor([[0.1480, 0.8520],
            [0.1503, 0.8497]]),atol=1e-4,rtol=1e-3)
    log_phis_expanded = [log_phis[0][..., None].expand(2,2,2,2,2),
                         log_phis[1][..., None, :, None, None].expand(2,2,2,2,2),
                         log_phis[2][..., None, None, :, :].expand(2,2,2,2,2),
                         log_phis[3][..., None, None, None, :].expand(2,2,2,2,2)]
    ent_ref = torch.distributions.Categorical(logits=sum(log_phis_expanded).reshape(2, -1)).entropy() # type:ignore
    torch.testing.assert_close(tbp.entropy(adjacency_matrix, log_phis, results), ent_ref)

def test_sample():
    adjacency_matrix = torch.tensor([
        [1,0,0,0],
        [1,1,0,0],
        [1,0,1,0],
        [0,0,1,1]
    ])
    f1 = torch.log(torch.tensor(
        [0.1,0.3,0.2,0.1,0.05,0.0,0.0,0.25]
    )).reshape(2,2,2)
    f2 = torch.log(torch.tensor(
        [1, 1.]
    )).reshape(2)
    f3 = torch.log(torch.tensor(
        [4, 1., 1., 4]
    )).reshape(2,2)
    f4 = torch.log(torch.tensor(
        [2 * 1.3, 8 * 1.3]
    )).reshape(2)
    tbp = TreeBeliefPropagation()
    log_phis = [f1, f2, f3, f4]
    samples = tbp.sample(adjacency_matrix, [2,2,2,2], log_phis, 10_000000, dict()).detach().to("cpu")
    binary_samples = int_to_bin(4, [2,2,2,2], samples)
    marginals = binary_samples.mean(dim=0)
    results = tbp.infer(adjacency_matrix, [2,2,2,2], log_phis)
    m = tbp.marginals(results)
    torch.testing.assert_close(m[0].exp(), marginals[:2], atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(m[1].exp(), marginals[2:4], atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(m[2].exp(), marginals[4:6], atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(m[3].exp(), marginals[6:8], atol=1e-4,rtol=1e-3)

if __name__ == "__main__":
    # V = [1,2,3,4,5]
    # E = [(1,2), (2,3), (1,3), (2,5), (4,3), (3,5), (5,4)]
    # topo_sort(V, E)
    test()
    test_sample()
    print("tests passed!")
