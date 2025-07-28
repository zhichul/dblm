from __future__ import annotations
from dblm.rva.discrete import dist
import itertools
import math
import numpy as np


class Independent(dist.Distribution):

    def __init__(self,
                 nvars: int,
                 nvals: np.ndarray, # n
                 dists: list[np.ndarray]) -> None:
        super().__init__(nvars, nvals)
        self.dists = dists

    def sample(self, n: int, encoding="binary"):
        samples = []
        for i in range(self.nvars):
            sample = np.random.choice(np.arange(self.nvals[i]), size=n, p=self.dists[i])
            if encoding == "integer":
                samples.append(sample[:, None])
            elif encoding == "binary":
                sample_bin = np.zeros((n, self.nvals[i]))
                sample_bin[np.arange(n), sample] = 1
                samples.append(sample_bin)
            else:
                raise NotImplementedError(encoding)
        return np.concatenate(samples, axis=1)


    def log_probability_table(self, encoding="binary"):
        prob_vec = []
        assignment_mat = []
        for (probs, assignments) in zip(itertools.product(*self.dists), itertools.product(*[range(nval) for nval in self.nvals])):
            prob_vec.append(sum(math.log(prob) for prob in probs))
            assignment_mat.append(assignments)
        assignment_mat = np.array(assignment_mat)
        prob_vec = np.array(prob_vec)
        if encoding == "binary":
            return prob_vec, self.int_to_bin(assignment_mat)
        elif encoding == "integer":
            return prob_vec, assignment_mat
        else:
            raise NotImplementedError(encoding)

    def log_marginals(self):
        return np.log(np.array(list(itertools.chain(*self.dists))))

    def log_marginals_conditioned_on_one(self):
        out = np.broadcast_to(self.log_marginals()[None,:], (self.nv, self.nv)).copy() # type:ignore
        for i in range(self.nvars):
            s, e = self.start_end(i)
            out[s:e, s:e] = np.log(np.eye(self.nvals[i]))
        return out

    def backoff_log_marginals_conditioned_on_one(self):
        return self.log_marginals_conditioned_on_one()

    def log_marginals_conditioned_on_two(self):
        out = np.broadcast_to(self.log_marginals_conditioned_on_one()[None,:,:], (self.nv, self.nv, self.nv)).copy() # type:ignore
        for i in range(self.nvars):
            s, e = self.start_end(i)
            out[i, s:i] = -math.inf # impossible to condition otherwise
            out[i, i+1:e] = -math.inf
        return out

if __name__ == "__main__":
    dist = Independent(5, [5,4,3,2,3], [[0.2]*5, [0.25] *4, [1/3]*3, [0.5, 0.5], [1/3]*3]) # type:ignore
    mat_int = dist.sample(3, encoding="integer")
    mat_bin = dist.int_to_bin(mat_int)
    mat_int_again = dist.bin_to_int(mat_bin)
    print(mat_int)
    print(mat_bin)
    print(mat_int_again)
    print(dist.log_marginals())
    print(dist.log_marginals_conditioned_on_one())
