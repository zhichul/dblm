from __future__ import annotations
import math
import sys
import numpy as np
from scipy.special import log_softmax, logsumexp, softmax
import torch
import tqdm

from dblm.utils import seeding

def expected_attention(log_unnormalized_attention_weights: np.ndarray, # n, |V|
                       log_probabilities, # T
                       assignments # T, |V|
                       ):
    n, V = log_unnormalized_attention_weights.shape
    T = log_probabilities.shape[0]
    uaw = np.broadcast_to(log_unnormalized_attention_weights[:, None, :], (n, T, V))
    mask = np.broadcast_to(1-assignments[None,:,:], (n,T,V))
    uaw_masked = np.ma.array(uaw, mask=mask)
    uaw_filled = uaw_masked.filled(-math.inf)
    log_attention = log_softmax(uaw_filled, axis=-1)
    expected_attention = np.exp(logsumexp(log_attention + log_probabilities[None, :, None], axis=1))
    return expected_attention

def albo(log_unnormalized_attention_weights: np.ndarray,  # n, |V|
        log_marginals: np.ndarray, # |V|,
        log_marginals_conditioned_on_one: np.ndarray # |V| x |V|
        ):
    log_numerator = log_unnormalized_attention_weights + log_marginals[None, :]
    log_denominator = logsumexp(log_unnormalized_attention_weights[:,None,:] + log_marginals_conditioned_on_one[None, :, :], axis=-1)
    return np.exp(log_numerator - log_denominator)

def mult_gate(log_unnormalized_attention_weights: np.ndarray,  # n, |V|
        log_marginals: np.ndarray, # |V|,
        ):
    log_numerator = log_unnormalized_attention_weights + log_marginals[None, :]
    return softmax(log_numerator, axis=1)

def albo_b(log_unnormalized_attention_weights: np.ndarray,  # n, |V|
        log_marginals: np.ndarray, # |V|,
        ):
    log_marginals_conditioned_on_one = np.broadcast_to(log_marginals[None, :], (log_marginals.shape[0], log_marginals.shape[0]))
    return albo(log_unnormalized_attention_weights, log_marginals, log_marginals_conditioned_on_one)

def cut(n, V, k, MAX):
    if n * k * V * 4 > MAX:
        increment = MAX // (n * V * 4)
        left = k
        sizes = []
        while left > 0:
            sizes.append(min(increment, left))
            left = left - increment
    else:
        sizes = [k]
    return sizes

def sample_k(log_unnormalized_attention_weights: np.ndarray, sampler, k, log_marginals=None):
    MAX = 10_000_000_000
    n, V = log_unnormalized_attention_weights.shape
    with torch.no_grad():
        log_probs = None
        attempts = 0
        max_attempts = 100
        while log_probs is None and attempts < max_attempts:
            try:
                sizes = cut(n, V, k, MAX)
                log_probs = None
                counts = torch.tensor(0.0).to("cuda")
                for size in tqdm.tqdm(sizes, leave=False):  # type:ignore
                    sample = sampler(size) # s x |V|
                    counts = counts + torch.from_numpy(sample.sum(axis=0)).to("cuda")
                    # sample = np.broadcast_to(sample[None,:,:], (n, size, V))
                    # log_uaw = np.broadcast_to(log_unnormalized_attention_weights[:,None,:], (n,size,V))
                    # masked_weights = np.ma.array(log_uaw, mask=1-sample).filled(-math.inf)
                    # masked_log_probs = log_softmax(masked_weights, axis=-1)
                    # log_probs.append(logsumexp(masked_log_probs, axis=1))
                    sample = torch.from_numpy(sample).to(torch.bool)[None,:,:].to("cuda").expand((n, size, V))
                    log_uaw = torch.from_numpy(log_unnormalized_attention_weights[:,None,:]).to("cuda").expand((n,size,V))
                    masked_weights = log_uaw.masked_fill(~sample, -math.inf)
                    masked_log_probs = masked_weights.log_softmax(-1)
                    if log_probs is None:
                        log_probs = masked_log_probs.logsumexp(1)
                    else:
                        log_probs = [torch.logaddexp(log_probs, masked_log_probs.logsumexp(1)) for _ in range(1)][0]
            except RuntimeError as e:
                log_probs = None
                attempts += 1
                MAX = int(MAX * 0.8)
        # log_probs = logsumexp(np.array(log_probs), axis=0)
        # log_probs = log_probs - math.log(k) # type:ignore
        # return np.exp(log_probs)
        log_probs = log_probs.cpu().numpy() # type:ignore
        if log_marginals is None:
            log_probs = log_probs - math.log(k) # type:ignore
        else:
            counts = counts.clamp(min=1)
            log_probs = log_probs - counts.reshape(1,-1).log().cpu().numpy() + log_marginals.reshape(1,-1)
        return np.exp(log_probs)

if __name__ == "__main__":
    from dblm.rva import unnormalized_attention
    from dblm.rva.discrete import independent
    seeding.seed(12)
    nvars = 5
    nvals = np.array([5,4,3,2,3])
    scales_mu = np.array([3] * nvars)
    scales_d = np.array([scales_mu[0] * 0.1] * nvars)
    corr_d = [np.eye(nvals[i]) for i in range(nvars)]
    q = unnormalized_attention.IntraVar(nvars, nvals, scales_mu, scales_d, corr_d)
    uaw = q.sample(10)

    dist = independent.Independent(5, [5,4,3,2,3], [[0.2]*5, [0.25] *4, [1/3]*3, [0.5, 0.5], [1/3]*3]) # type:ignore

    logp, ass  = dist.log_probability_table()
    ea  = expected_attention(uaw, logp, ass)
    print(ea.round(3))
    albo_ea  = albo(uaw, dist.log_marginals(), dist.log_marginals_conditioned_on_one())
    mult_gate_ea  = mult_gate(uaw, dist.log_marginals())
    sample_10_ea  = sample_k(uaw, dist.sample, 10)
    sample_100_ea  = sample_k(uaw, dist.sample, 100)
    sample_1000_ea  = sample_k(uaw, dist.sample, 1000)
    sample_10000_ea  = sample_k(uaw, dist.sample, 10000)
    print("albo", np.abs((ea - albo_ea)).mean())
    print("softmax", np.abs((ea - softmax(uaw, axis=-1))).mean())
    print("mult_gate", np.abs((ea - mult_gate_ea)).mean())
    print("sample_10", np.abs((ea - sample_10_ea)).mean())
    print("sample_100", np.abs((ea - sample_100_ea)).mean())
    print("sample_1000", np.abs((ea - sample_1000_ea)).mean())
    print("sample_10000", np.abs((ea - sample_10000_ea)).mean())
