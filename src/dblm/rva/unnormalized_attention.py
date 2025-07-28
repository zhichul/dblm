from __future__ import annotations
from dblm.utils import seeding
from scipy.special import softmax
import numpy as np


class InterVar:

    def __init__(self, nvars: int,
                       nvals: np.ndarray, # n
                       scales_mu: np.ndarray, # n
                       corr_mu: np.ndarray, # n, n
                       scales_d: np.ndarray, #  n
                       corr_d: list[np.ndarray], # |S_i| x |S_i| for i in range(n)
    ):
        self.nvars = nvars
        self.nvals = nvals
        self.scales_mu = scales_mu
        self.corr_mu = corr_mu
        self.scales_d = scales_d
        self.corr_d = corr_d

    def sample(self, n: int):
        cov_mu = np.diag(self.scales_mu) @ self.corr_mu @ np.diag(self.scales_mu) 
        mu = np.random.multivariate_normal(np.zeros_like(self.scales_mu), cov_mu, size=(n))
        binary_reps = []
        for i in range(self.nvars):
            scale_i = self.scales_d[i]
            corr_i = self.corr_d[i]
            nvals_i = self.nvals[i]
            cov_i = scale_i**2 * corr_i
            d_i = np.random.multivariate_normal(np.zeros(nvals_i), cov_i, size=(n))
            binary_reps.append(d_i + mu[:, i:i+1])
        return np.concatenate(binary_reps, axis=1)

class IntraVar(InterVar):

    def __init__(self, nvars: int,
                       nvals: np.ndarray, # n
                       scales_mu: np.ndarray, # n
                       scales_d: np.ndarray, #  n
                       corr_d: list[np.ndarray], # |S_i| x |S_i| for i in range(n)
    ):
        super().__init__(nvars, nvals, scales_mu, np.eye(nvars), scales_d, corr_d)

if __name__ == "__main__":
    seeding.seed(12)
    nvars = 5
    nvals = np.array([3] * nvars)
    scales_mu = np.array([3] * nvars)
    scales_d = np.array([scales_mu[0] * 0.1] * nvars)
    corr_d = [np.eye(nvals[0])] * nvars
    q = IntraVar(nvars, nvals, scales_mu, scales_d, corr_d)
    print(softmax(q.sample(10), axis=1).round(2))
    import tensorflow_probability as tfp

    dist = tfp.distributions.CholeskyLKJ(
        dimension=5,
        concentration=1,
        validate_args=False,
        allow_nan_stats=True,
        name='CholeskyLKJ'
    )
    corr_mu_l = dist.sample().numpy()
    corr_mu = corr_mu_l @ corr_mu_l.T
    q = InterVar(nvars, nvals, scales_mu, corr_mu, scales_d, corr_d)
    print(softmax(q.sample(10), axis=1).round(2))
    breakpoint()
